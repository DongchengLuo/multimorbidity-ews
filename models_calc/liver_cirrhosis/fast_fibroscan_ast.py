import pandas as pd
import numpy as np
import re
import math

# ==========================================
# 1. Field mapping (FAST Score indicators)
# ==========================================
FIELD_MAPPING = {
    # --- Required FibroScan indicators ---
    'LSM': ['LSM', 'liver_stiffness_noninvasive', 'liver_stiffness', 'stiffness_value'],  # unit: kPa
    'CAP': ['CAP', 'fat_attenuation', 'liver_stiffness_noninvasive'],  # unit: dB/m

    # --- Required biochemical indicators ---
    'AST': ['AST'],
}

# Gauge reference ticks (low | medium | high)
GAUGE_REF_STR = "0-0.35|0.35-0.67|0.67-1.0"


# ==========================================
# 2. Basic cleaning and utility functions
# ==========================================
def normalize_str(s):
    if pd.isna(s): return ""
    return re.sub(r'[\(\)\（\）\[\]\【\】\{\}\s]', '', str(s)).lower()


def get_category(name):
    """Match the standard label from the field name (exact match)"""
    norm_name = normalize_str(name)
    for key, patterns in FIELD_MAPPING.items():
        for p in patterns:
            if normalize_str(p) == norm_name:
                return key
    return None


def clean_numeric(val):
    """Generic pure-numeric extraction: take the first number that appears"""
    if pd.isna(val) or str(val).strip() == "": return np.nan
    try:
        match = re.search(r"[-+]?\d*\.\d+|\d+", str(val))
        return float(match.group()) if match else np.nan
    except:
        return np.nan


def clean_lsm(val, allow_fallback=True):
    """Extract liver stiffness (LSM) with precise keyword and kPa unit recognition"""
    if pd.isna(val) or str(val).strip() == "": return np.nan
    s = str(val)
    try:
        match = re.search(r'(?:liver stiffness|E value|LSM)[^\d]*(\d+(?:\.\d+)?)', s, re.IGNORECASE)
        if match: return float(match.group(1))

        match_kpa = re.search(r'(\d+(?:\.\d+)?)\s*kpa', s, re.IGNORECASE)
        if match_kpa: return float(match_kpa.group(1))

        if allow_fallback:
            return clean_numeric(s)
        return np.nan
    except:
        return np.nan


def clean_cap(val, allow_fallback=True):
    """Extract fat attenuation (CAP) with precise keyword and dB/m unit recognition"""
    if pd.isna(val) or str(val).strip() == "": return np.nan
    s = str(val)
    try:
        match = re.search(r'(?:CAP|fat attenuation)[^\d]*(\d+(?:\.\d+)?)', s, re.IGNORECASE)
        if match: return float(match.group(1))

        match_db = re.search(r'(\d+(?:\.\d+)?)\s*db/m', s, re.IGNORECASE)
        if match_db: return float(match_db.group(1))

        if allow_fallback:
            return clean_numeric(s)
        return np.nan
    except:
        return np.nan


# ==========================================
# 3. FAST Score core computation logic
# ==========================================
def calculate_fast_score(lsm, cap, ast):
    """
    Formula reference: Newsome PN, et al. Lancet Gastroenterol Hepatol 2020.
    L = -1.65 + 1.07 * ln(LSM) + 2.66*10^-8 * CAP^3 - 63.3 * AST^-1
    FAST = exp(L) / (1 + exp(L))
    """
    if any(pd.isna(x) for x in [lsm, cap, ast]) or lsm <= 0 or ast == 0:
        return None, "-", "-", "-"

    try:
        l_val = -1.65 + 1.07 * math.log(lsm) + 2.66e-8 * (cap ** 3) - 63.3 * (1 / ast)
        score = math.exp(l_val) / (1 + math.exp(l_val))

        if score <= 0.35:
            risk_level = "low risk"
            code = "K74.5.1"
            suggestion = "Excludes significant active fibrotic MASH (90% sensitivity)"
        elif score >= 0.67:
            risk_level = "high risk"
            code = "K74.5.3"
            suggestion = "Highly suspected active fibrotic MASH; biopsy or close monitoring advised"
        else:
            risk_level = "medium risk"
            code = "K74.5.2"
            suggestion = "Indeterminate zone; further clinical assessment advised"

        return score, risk_level, suggestion, code

    except Exception:
        return None, "-", "-", "-"


# ==========================================
# 4. Main data-processing routine
# ==========================================
def process_data(base_path, item_path, output_path):
    print("--- Starting FAST Score (fibrotic MASH) model ---")

    def read_csv_safe(path):
        for enc in ['utf-8-sig', 'gbk', 'utf-8']:
            try:
                return pd.read_csv(path, encoding=enc, low_memory=False)
            except:
                continue
        return pd.DataFrame()

    base_df = read_csv_safe(base_path)
    item_df = read_csv_safe(item_path)

    if base_df.empty or item_df.empty:
        print("Data file could not be read or is empty")
        return

    # 1. Pivot to wide table
    print("Aggregating exam indicators...")
    item_df['cat'] = item_df['res_itm_name'].apply(get_category)
    filtered = item_df.dropna(subset=['cat', 'res_value']).copy()

    records = []
    filtered['app_no'] = filtered['app_no'].astype(str).str.strip()
    filtered['pat_id'] = filtered['pat_id'].astype(str).str.strip()

    for (app_no, pat_id), group in filtered.groupby(['app_no', 'pat_id']):
        row = {'app_no': app_no, 'pat_id': pat_id}
        data_pool = {k: None for k in FIELD_MAPPING.keys()}

        for _, item in group.iterrows():
            cat = item['cat']
            # Keep the raw string, taking only the first value encountered (no averaging)
            if data_pool[cat] is None:
                data_pool[cat] = str(item['res_value'])

        raw_lsm = data_pool.get('LSM') or ""
        raw_cap = data_pool.get('CAP') or ""

        # [Core extraction logic - cross-validation]
        # Extract LSM: search LSM-matched content first; if absent, search CAP-matched content precisely (no generic extraction)
        lsm_val = clean_lsm(raw_lsm, allow_fallback=True)
        if pd.isna(lsm_val):
            lsm_val = clean_lsm(raw_cap, allow_fallback=False)

        # Extract CAP: search CAP-matched content first; if absent, search LSM-matched content precisely (no generic extraction)
        cap_val = clean_cap(raw_cap, allow_fallback=True)
        if pd.isna(cap_val):
            cap_val = clean_cap(raw_lsm, allow_fallback=False)

        row['LSM'] = lsm_val
        row['CAP'] = cap_val
        row['AST'] = clean_numeric(data_pool.get('AST', ""))

        records.append(row)

    wide_items = pd.DataFrame(records)

    # 2. Merge basic information
    base_df['app_no'] = base_df['app_no'].astype(str).str.strip()
    base_df['pat_id'] = base_df['pat_id'].astype(str).str.strip()

    if not wide_items.empty:
        final = pd.merge(base_df, wide_items, on=['app_no', 'pat_id'], how='left')
    else:
        final = base_df.copy()
        final['LSM'] = np.nan
        final['CAP'] = np.nan
        final['AST'] = np.nan

    # 3. Scoring and result generation
    results = []
    print("Computing risk scores...")

    for _, r in final.iterrows():
        lsm = r.get('LSM', np.nan)
        cap = r.get('CAP', np.nan)
        ast = r.get('AST', np.nan)

        score, risk_level, suggestion, code = calculate_fast_score(lsm, cap, ast)

        results.append({
            'app_no': r.get('app_no'),
            'pat_id': r.get('pat_id'),
            'pat_name': r.get('pat_name', 'unknown'),
            'pat_sex': r.get('pat_sex', "-"),

            # --- Raw feature input data ---
            'LSM (kPa)': round(lsm, 2) if pd.notna(lsm) else "-",
            'CAP (dB/m)': round(cap, 1) if pd.notna(cap) else "-",
            'AST': round(ast, 1) if pd.notna(ast) else "-",

            # --- Model output results ---
            'predicted_score': round(score, 3) if score is not None else "-",
            'risk_level': risk_level,
            'code': code,
            'gauge_reference': GAUGE_REF_STR,
            'model_name': 'Cirrhosis FAST Score (NASH)'
        })

    # 4. Export
    if results:
        res_df = pd.DataFrame(results)
        res_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"--- Assessment complete, exported to {output_path} ---")
        print(f"total exported: {len(res_df)}")
    else:
        print("--- WARNING: no valid data found ---")


if __name__ == "__main__":
    process_data(
        "base.csv",
        "item_result.csv",
        "fast_nash_result.csv"
    )