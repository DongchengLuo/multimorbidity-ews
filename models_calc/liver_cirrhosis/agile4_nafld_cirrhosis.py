import pandas as pd
import numpy as np
import re
import math

# ==========================================
# 1. Field mapping (see the precise field-matching reference table)
# ==========================================
FIELD_MAPPING = {
    # --- Required basic fields ---
    'gender': ['pat_sex', 'sex'],

    # --- Past medical history ---
    'has_diabetes': ['prior_diabetes', 'type2_diabetes', 'diabetes_history', 'past_history_diabetes'],

    # --- Laboratory test indicators ---
    'ast': ['AST'],
    'alt': ['ALT'],
    'plt': ['PLT', 'platelet_count', 'B-PLT'],

    # --- Imaging / special examinations ---
    'lsm': ['liver_stiffness', 'LSM', 'liver_stiffness_noninvasive', 'liver_elastography']
}


# ==========================================
# 2. Basic cleaning and matching logic
# ==========================================
def normalize_str(s):
    if pd.isna(s): return ""
    return re.sub(r'[\(\)\（\）\[\]\【\】\{\}\s]', '', str(s)).lower()


def get_category(name):
    norm_name = normalize_str(name)
    for key, patterns in FIELD_MAPPING.items():
        for p in patterns:
            # Change: switch to exact matching
            if normalize_str(p) == norm_name: return key
    return None


def clean_numeric(val):
    """Generic numeric extraction: take the first number in the string"""
    if pd.isna(val) or str(val).strip() == "": return np.nan
    try:
        match = re.search(r"[-+]?\d*\.\d+|\d+", str(val))
        return float(match.group()) if match else np.nan
    except:
        return np.nan


def clean_lsm(val):
    """
    Dedicated cleaning of complex liver-stiffness (LSM) strings.
    Handle cases such as "CAP:171dB/m, liver stiffness:5.7kPa".
    """
    if pd.isna(val) or str(val).strip() == "":
        return np.nan

    s = str(val)

    try:
        # Priority 1: look for explicit keywords "liver stiffness", "E value" or "LSM" and take the following number
        match = re.search(r'(?:liver stiffness|E value|LSM)[^\d]*(\d+(?:\.\d+)?)', s, re.IGNORECASE)
        if match:
            return float(match.group(1))

        # Priority 2: if "liver stiffness" is not stated but a kPa unit appears, take the number before kPa
        match_kpa = re.search(r'(\d+(?:\.\d+)?)\s*kpa', s, re.IGNORECASE)
        if match_kpa:
            return float(match_kpa.group(1))

        # Priority 3 (fallback): if only a bare number is present (e.g. "5.7"), fall back to basic extraction
        return clean_numeric(s)
    except:
        return np.nan


def parse_gender_to_num(val):
    """Male=1, Female=0"""
    s = str(val).lower()
    if 'male' in s or 'm' in s or '1' == s: return 1
    return 0


def parse_yes_no(val):
    """Yes=1, No=0"""
    s = str(val).lower()
    if any(x in s for x in ['yes', 'present', 'y', '1']): return 1
    return 0


# ==========================================
# 3. Agile 4 core scoring logic
# ==========================================
def calculate_agile4(sex_num, diabetes_num, lsm, ast, alt, plt):
    """
    Agile 4 Formula (Liver Cirrhosis Prediction)
    Formula: P = exp(logit) / (1 + exp(logit))
    """
    debug_info = {}

    # 1. Completeness check: return "-" without computing when items are missing
    if any(pd.isna(x) or x <= 0 for x in [lsm, ast, alt, plt]):
        return None, "-", "-", "-", {}

    try:
        aar = ast / alt

        # Logit formula
        logit = 7.50139 - \
                (15.42498 * (1.0 / math.sqrt(lsm))) - \
                (0.01378 * plt) - \
                (1.41149 * (1.0 / aar)) - \
                (0.53281 * sex_num) + \
                (0.41741 * diabetes_num)

        score = math.exp(logit) / (1 + math.exp(logit))
        debug_info['AAR'] = round(aar, 2)

        # Move risk determination inside the function for unified handling
        if score < 0.251:
            risk_level = "low risk"
            suggestion = "regular follow-up"
            risk_code = "K74.3.1"
        elif score > 0.565:
            risk_level = "high risk"
            suggestion = "see a specialist promptly"
            risk_code = "K74.3.3"
        else:
            risk_level = "medium risk"
            suggestion = "further examination advised"
            risk_code = "K74.3.2"

        return score, risk_level, risk_code, suggestion, debug_info

    except Exception as e:
        return None, "-", "-", "-", {}


# ==========================================
# 4. Main data-processing routine
# ==========================================
def process_data(base_path, item_path, output_path):
    print("--- Starting Agile 4 (cirrhosis) model ---")

    def read_csv_safe(path):
        for enc in ['gbk', 'utf-8-sig', 'utf-8']:
            try:
                return pd.read_csv(path, encoding=enc, low_memory=False)
            except:
                continue
        return pd.DataFrame()

    base_df = read_csv_safe(base_path)
    item_df = read_csv_safe(item_path)

    if base_df.empty:
        print(f"error: base data file unreadable or empty")
        return

    # 1. Pivot to wide table
    print("cleaning and aggregating exam data...")
    records = []
    if not item_df.empty:
        item_df['cat'] = item_df['res_itm_name'].apply(get_category)
        filtered = item_df.dropna(subset=['cat', 'res_value']).copy()

        for (app_no, pat_id), group in filtered.groupby(['app_no', 'pat_id']):
            row = {'app_no': str(app_no).strip(), 'pat_id': str(pat_id).strip()}
            data_pool = {}

            # Change: drop np.nanmean and assign directly in the loop
            for _, item in group.iterrows():
                data_pool[item['cat']] = str(item['res_value'])

            # [Core change]: use the dedicated clean_lsm to extract liver stiffness
            row['lsm'] = clean_lsm(data_pool.get('lsm', np.nan))

            row['ast'] = clean_numeric(data_pool.get('ast', np.nan))
            row['alt'] = clean_numeric(data_pool.get('alt', np.nan))
            row['plt'] = clean_numeric(data_pool.get('plt', np.nan))

            dia_raw = data_pool.get('has_diabetes', "")
            row['diabetes_num'] = parse_yes_no(dia_raw)

            records.append(row)

    if records:
        wide_items = pd.DataFrame(records)
    else:
        wide_items = pd.DataFrame(columns=['app_no', 'pat_id', 'lsm', 'ast', 'alt', 'plt', 'diabetes_num'])

    # 2. Merge
    base_df['app_no'] = base_df['app_no'].astype(str).str.strip()
    base_df['pat_id'] = base_df['pat_id'].astype(str).str.strip()
    if 'app_no' not in base_df.columns:
        base_df['app_no'] = ""

    # Change: left join with base_df as the base (how='left')
    final = pd.merge(base_df, wide_items, on=['app_no', 'pat_id'], how='left')

    # 3. Scoring and result generation
    print("running model computation...")
    results = []
    GAUGE_REF = "0-0.251|0.251-0.565|0.565-1"

    for _, r in final.iterrows():
        sex_raw = r.get('pat_sex', '')
        sex_num = parse_gender_to_num(sex_raw)

        lsm = r.get('lsm')
        ast = r.get('ast')
        alt = r.get('alt')
        plt = r.get('plt')
        # Treat NaN from the left join as empty / no diabetes
        diabetes_num = r.get('diabetes_num')
        diabetes_num = int(diabetes_num) if pd.notna(diabetes_num) else 0

        # Compute the score (return a placeholder when missing)
        score, risk_level, risk_code, suggestion, info = calculate_agile4(sex_num, diabetes_num, lsm, ast, alt, plt)

        # Handle missing data and format output
        results.append({
            'app_no': r.get('app_no'),
            'pat_id': r.get('pat_id'),
            'pat_name': r.get('pat_name', 'unknown'),

            # --- Raw feature input data ---
            'pat_sex': sex_raw if pd.notna(sex_raw) and str(sex_raw).strip() != "" else "-",
            'prior_diabetes': diabetes_num,
            'AST': round(ast, 2) if pd.notna(ast) else "-",
            'ALT': round(alt, 2) if pd.notna(alt) else "-",
            'PLT': round(plt, 2) if pd.notna(plt) else "-",
            'liver_stiffness': round(lsm, 2) if pd.notna(lsm) else "-",

            # --- Intermediate derived indicators ---
            'AAR': info.get('AAR', "-"),

            # --- Model output results ---
            'predicted_score': round(score, 4) if score is not None else "-",
            'risk_level': risk_level,
            'code': risk_code,
            'gauge_reference': GAUGE_REF,
            'suggestion': suggestion,
            'model_name': 'Agile 4'
        })

    # 4. Export
    if results:
        res_df = pd.DataFrame(results)
        res_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"--- Assessment complete: {len(res_df)} results generated, exported to {output_path} ---")
    else:
        print("--- Assessment complete, but no valid results were generated ---")


if __name__ == "__main__":
    process_data(
        "base.csv",
        "item_result.csv",
        "agile4_result.csv"
    )