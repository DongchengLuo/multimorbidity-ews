import pandas as pd
import numpy as np
import re
import math

# ==========================================
# 1. Field mapping
# ==========================================
FIELD_MAPPING = {
    'AST': ['AST'],
    'scr': ['creatinine (Cr)', 'creatinine'],
    'lsm': ['liver_stiffness', 'liver_stiffness_noninvasive'],
    'age': ['app_pat_age'],
    'gender': ['pat_sex']
}


def normalize_str(s):
    if pd.isna(s): return ""
    return re.sub(r'[\(\)\（\）\[\]\【\】\{\}\s]', '', str(s)).lower()


def get_category(name):
    norm_name = normalize_str(name)
    for key, patterns in FIELD_MAPPING.items():
        for p in patterns:
            # [Exact match]
            if normalize_str(p) == norm_name: return key
    return None


def clean_numeric(val):
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


# ==========================================
# 2. acFibroMASH core algorithm
# ==========================================
def calculate_acfibromash(ast, scr, lsm):
    """
    Compute the acFibroMASH index
    """
    # Completeness check: return the placeholder "-" without computing when items are missing
    if any(pd.isna(x) or x <= 0 for x in [ast, scr, lsm]):
        return None, "-", "-", "-", "-", "-"

    try:
        # Compute acMASH
        ac_mash = (ast / scr) * 10

        # Compute probability
        linear_pred = -3.956 + (0.305 * lsm) + (0.065 * ac_mash)
        try:
            prob = math.exp(linear_pred) / (1 + math.exp(linear_pred))
        except OverflowError:
            prob = 1.0

        # Risk tiering and encoding
        if prob < 0.15:
            risk_diag = "Rule-out"
            risk_5yr = "low risk"
            code = "K74.1.1"
            suggestion = "Fibrotic MASH excluded; low 5-year liver event risk."
        elif 0.15 <= prob <= 0.39:
            risk_diag = "Gray zone"
            risk_5yr = "medium risk"
            code = "K74.1.2"
            suggestion = "Diagnosis uncertain; further clinical/imaging assessment advised."
        else:  # > 0.39
            risk_diag = "Rule-in"
            risk_5yr = "high risk"
            code = "K74.1.3"
            suggestion = "Highly suspected fibrotic MASH; markedly elevated 5-year liver event risk; consult a specialist."

        return prob, ac_mash, risk_diag, risk_5yr, code, suggestion

    except Exception:
        return None, "-", "-", "-", "-", "-"


# ==========================================
# 3. Main data-processing routine
# ==========================================
def process_acfibromash(base_path, item_path, output_path):
    print("--- Starting acFibroMASH model ---")

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
        print("error: cannot read base data file or it is empty.")
        return

    # 1. Extract indicators
    records = []
    if not item_df.empty:
        item_df['cat'] = item_df['res_itm_name'].apply(get_category)
        filtered = item_df.dropna(subset=['cat', 'res_value']).copy()

        # Group by the key IDs used for matching
        for (app_no, pat_id), group in filtered.groupby(['app_no', 'pat_id']):
            row = {'app_no': str(app_no).strip(), 'pat_id': str(pat_id).strip()}

            data_pool = {}
            for _, item in group.iterrows():
                data_pool[item['cat']] = str(item['res_value'])

            # [Core change]: use the new dedicated extractor for lsm and clean_numeric for other indicators
            row['lsm'] = clean_lsm(data_pool.get('lsm', np.nan))
            row['AST'] = clean_numeric(data_pool.get('AST', np.nan))
            row['scr'] = clean_numeric(data_pool.get('scr', np.nan))

            records.append(row)

    if records:
        wide_items = pd.DataFrame(records)
    else:
        # Produce an empty DataFrame for merging when there are no lab indicators
        wide_items = pd.DataFrame(columns=['app_no', 'pat_id', 'AST', 'scr', 'lsm'])

    # 2. Merge basic information
    base_df['app_no'] = base_df['app_no'].astype(str).str.strip()
    base_df['pat_id'] = base_df['pat_id'].astype(str).str.strip()

    # Change: left join with base_df as the base (how='left')
    final = pd.merge(base_df, wide_items, on=['app_no', 'pat_id'], how='left')

    results = []
    MODEL_NAME = "acFibroMASH (fibrotic MASH diagnosis and prediction)"
    # Fixed gauge reference ranges
    GAUGE_REF = "0-0.15|0.15-0.39|0.39-1.0"

    for _, r in final.iterrows():
        ast = r.get('AST')
        scr = r.get('scr')
        lsm = r.get('lsm')

        # Fetch age and sex for display
        age = r.get('app_pat_age')
        sex = r.get('pat_sex')

        # Compute the score (return a placeholder when missing)
        prob, ac_mash, risk_diag, risk_5yr, code, suggestion = calculate_acfibromash(ast, scr, lsm)

        # ---------------------------------------------------------
        # Handle missing data and format the output string
        # ---------------------------------------------------------
        results.append({
            'model_name': MODEL_NAME,
            'app_no': r.get('app_no'),
            'pat_id': r.get('pat_id'),
            'pat_name': r.get('pat_name', r.get('name', 'unknown')),
            'sex': sex if pd.notna(sex) and str(sex).strip() != "" else "-",
            'age': int(clean_numeric(age)) if pd.notna(clean_numeric(age)) else "-",

            # --- Raw feature input data ---
            'AST(U/L)': round(ast, 2) if pd.notna(ast) else "-",
            'serum_creatinine (umol/L)': round(scr, 2) if pd.notna(scr) else "-",
            'liver_stiffness_LSM (kPa)': round(lsm, 2) if pd.notna(lsm) else "-",

            # --- Intermediate derived indicators ---
            'acMASH_index': round(ac_mash, 2) if ac_mash != "-" else "-",

            # --- Model output results ---
            'acFibroMASH_probability': round(prob, 4) if prob is not None else "-",
            'code': code,
            'gauge_reference': GAUGE_REF,
            'current_diagnosis_category': risk_diag,
            '5y_liver_event_risk': risk_5yr,
            'recommendation': suggestion
        })

    # 3. Save results
    if results:
        res_df = pd.DataFrame(results)
        # Reordered output for a more readable result report
        res_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"--- Assessment complete: {len(res_df)} results generated, exported to {output_path} ---")
    else:
        print("--- Assessment complete, but no valid results were generated ---")


if __name__ == "__main__":
    process_acfibromash(
        "base.csv",
        "item_result.csv",
        "acFibroMASH_evaluation_results.csv"
    )