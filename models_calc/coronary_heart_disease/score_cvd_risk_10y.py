import pandas as pd
import numpy as np
import re

# ==========================================
# 1. Field mapping
# ==========================================
FIELD_MAPPING = {
    'age': ['app_pat_age'],
    'gender': ['pat_sex'],
    'smoke': ['smoking_history'],
    'sbp': ['systolic_bp', 'blood_pressure'],
    'TC': ['total_cholesterol', 'CHOL'],
    'HDL-C': ['HDL_cholesterol', 'HDL', 'HDL-C']
}


# ==========================================
# 2. Basic helper functions
# ==========================================
def normalize_str(s):
    if pd.isna(s): return ""
    return re.sub(r'[\(\)\（\）\[\]\【\】\{\}\s]', '', str(s)).lower()


def get_category(name):
    norm_name = normalize_str(name)
    for key, patterns in FIELD_MAPPING.items():
        if key in ['age', 'gender']: continue
        for p in patterns:
            if normalize_str(p) in norm_name: return key
    return None


def clean_numeric(val):
    if pd.isna(val) or str(val).strip() == "": return np.nan
    try:
        match = re.search(r"[-+]?\d*\.\d+|\d+", str(val))
        return float(match.group()) if match else np.nan
    except:
        return np.nan


def check_smoke_status(val_list):
    text = " ".join([str(v) for v in val_list])
    if any(kw in text for kw in ['yes', 'present', 'smoking', '1', 'smoking_years']):
        if 'absent' not in text and 'no' not in text and 'never' not in text:
            return 1
    return 0


# ==========================================
# 3. SCORE2 core algorithm and risk tiering
# ==========================================
def get_risk_info(risk_percentage):
    """Return the tier name and code from the percentage"""
    if pd.isna(risk_percentage):
        return "-", ""

    if risk_percentage < 5.0:
        return "low risk", "I25.5.1"
    elif 5.0 <= risk_percentage < 10.0:
        return "medium risk", "I25.5.2"
    else:
        return "high risk", "I25.5.3"


def calculate_score2_china(age, sex, smoker, sbp, total_chol, hdl_chol):
    if pd.isna(age) or pd.isna(sbp) or pd.isna(total_chol) or pd.isna(hdl_chol):
        return np.nan, "-", ""

    # Variable centering and coefficients (High Risk Region - China)
    center_age, center_sbp, center_tc, center_hdl = 60, 120, 6, 1.3
    scale_age, scale_sbp, scale_tc, scale_hdl = 5, 20, 1, 0.5

    c_age = (age - center_age) / scale_age
    c_sbp = (sbp - center_sbp) / scale_sbp
    c_tc = (total_chol - center_tc) / scale_tc
    c_hdl = (hdl_chol - center_hdl) / scale_hdl

    is_male = str(sex).strip() in ['male', '1', 'Male', 'M']

    if is_male:
        beta_age, beta_smoker, beta_sbp, beta_tc, beta_hdl = 0.3742, 0.6012, 0.2777, 0.1458, -0.2698
        i_age_smoker, i_age_sbp, i_age_tc, i_age_hdl = -0.0755, -0.0255, -0.0281, 0.0426
        baseline_survival = 0.9605
        scale1, scale2 = 0.778231091, 0.844985356
    else:
        beta_age, beta_smoker, beta_sbp, beta_tc, beta_hdl = 0.4648, 0.7744, 0.3131, 0.1002, -0.2606
        i_age_smoker, i_age_sbp, i_age_tc, i_age_hdl = -0.1088, -0.0277, -0.0226, 0.0613
        baseline_survival = 0.9776
        scale1, scale2 = 0.611474287, 0.703624072

    lp = (beta_age * c_age) + (beta_smoker * smoker) + (beta_sbp * c_sbp) + \
         (beta_tc * c_tc) + (beta_hdl * c_hdl) + \
         (i_age_smoker * c_age * smoker) + (i_age_sbp * c_age * c_sbp) + \
         (i_age_tc * c_age * c_tc) + (i_age_hdl * c_age * c_hdl)

    risk_uncalibrated = max(0.00001, min(0.99999, 1 - (baseline_survival ** np.exp(lp))))
    transformed_risk = np.log(-np.log(1 - risk_uncalibrated))
    risk_final = 1 - np.exp(-np.exp(scale1 + (scale2 * transformed_risk)))

    risk_pct = risk_final * 100
    level_name, risk_code = get_risk_info(risk_pct)

    return risk_pct, level_name, risk_code


# ==========================================
# 4. Main data-processing routine
# ==========================================
def process_data(base_path, item_path, output_path):
    print("--- Starting SCORE2 (China) risk assessment ---")

    def read_csv_safe(path):
        for enc in ['gbk', 'utf-8-sig', 'utf-8']:
            try:
                return pd.read_csv(path, encoding=enc, low_memory=False)
            except:
                continue
        return pd.DataFrame()

    # Added internal helper: get the first valid number in a list
    def get_first_valid(val_list):
        for v in val_list:
            num = clean_numeric(v)
            if not pd.isna(num):
                return num
        return np.nan

    base_df = read_csv_safe(base_path)
    item_df = read_csv_safe(item_path)

    if base_df.empty or item_df.empty:
        print("Error: cannot read the data file")
        return

    # Process the item table
    item_df['cat'] = item_df['res_itm_name'].apply(get_category)
    filtered = item_df.dropna(subset=['cat', 'res_value']).copy()

    records = []
    # Group by app_no (application/visit number) and pat_id (patient ID) to identify the same exam
    for (app_no, pat_id), group in filtered.groupby(['app_no', 'pat_id']):
        row = {'app_no': str(app_no).strip(), 'pat_id': str(pat_id).strip()}
        data_pool = {k: [] for k in FIELD_MAPPING.keys() if k not in ['age', 'gender']}

        for _, item in group.iterrows():
            if item['cat']: data_pool[item['cat']].append(str(item['res_value']))

        # === Core change: no averaging; take the first valid value ===
        row['sbp'] = get_first_valid(data_pool.get('sbp', []))
        row['TC'] = get_first_valid(data_pool.get('TC', []))
        row['HDL-C'] = get_first_valid(data_pool.get('HDL-C', []))
        row['smoke_val'] = check_smoke_status(data_pool.get('smoke', []))
        records.append(row)

    wide_items = pd.DataFrame(records)
    if wide_items.empty:
        print("no valid assessment indicators found")
        return

    # Merge basic information
    base_df['app_no'] = base_df['app_no'].astype(str).str.strip()
    base_df['pat_id'] = base_df['pat_id'].astype(str).str.strip()

    age_col, sex_col = FIELD_MAPPING['age'][0], FIELD_MAPPING['gender'][0]
    final = pd.merge(wide_items, base_df, on=['app_no', 'pat_id'], how='inner')

    # Compute and build output
    results = []
    # Fixed gauge reference values
    GAUGE_REF = "0%-5%|5%-10%|10%-100%"

    for _, r in final.iterrows():
        age = clean_numeric(r.get(age_col))
        sex = r.get(sex_col, '')
        smoke_status = r.get('smoke_val', 0)

        risk_val, risk_level, risk_code = calculate_score2_china(
            age, sex, smoke_status, r.get('sbp'), r.get('TC'), r.get('HDL-C')
        )

        suggestion = "-"
        if not pd.isna(risk_val):
            if risk_val >= 10.0:
                suggestion = "elevated cardiovascular risk; intensify intervention"
            elif risk_val >= 5.0:
                suggestion = "medium risk; monitor blood pressure and lipids"
            else:
                suggestion = "low risk; maintain a healthy lifestyle"

        results.append({
            'pat_id': r.get('pat_id'),
            'app_no': r.get('app_no'),
            'pat_name': r.get('pat_name', 'unknown'),
            # === All raw indicators used in the computation ===
            'sex': sex,
            'age': int(age) if not pd.isna(age) else '',
            'smoking_history': 'yes' if smoke_status == 1 else 'no',
            'systolic_bp': f"{r.get('sbp'):.0f}" if not pd.isna(r.get('sbp')) else "-",
            'total_cholesterol': f"{r.get('TC'):.2f}" if not pd.isna(r.get('TC')) else "-",
            'HDL-C': f"{r.get('HDL-C'):.2f}" if not pd.isna(r.get('HDL-C')) else "-",
            # ==================================
            'risk_probability': f"{risk_val:.2f}%" if not pd.isna(risk_val) else "-",
            'prediction_horizon': 'next 10 years',
            'risk_level': risk_level,
            'code': risk_code,
            'gauge_reference': GAUGE_REF,
            'recommendation': suggestion,
            'model_name': 'CVD-score'
        })

    if results:
        pd.DataFrame(results).to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"Processing complete, saved to: {output_path}")


if __name__ == "__main__":
    process_data(
        "base.csv",
        "item_result.csv",
        "score_china_result.csv"
    )
