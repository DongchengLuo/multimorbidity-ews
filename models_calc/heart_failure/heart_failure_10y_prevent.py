import pandas as pd
import numpy as np
import re

# ==========================================
# 1. Field mapping
# ==========================================
FIELD_MAPPING = {
    'age': ['app_pat_age'],
    'gender': ['pat_sex'],
    'sbp': ['systolic_bp', 'blood_pressure'],
    'BMI': ['(body mass index) BMI', 'bmi', 'BMI'],
    'egfr': ['eGFR (CKD-EPI formula)', 'eGFR'],
    'scr': ['creatinine (Cr)', 'creatinine'],  # <-- added: creatinine mapping
    'height': ['height'],
    'weight': ['weight'],
    'has_diabetes': ['prior_diabetes', 'type2_diabetes', 'diabetes', 'past_history', 'past_medical_history'],
    'smoke': ['smoking_history', 'smoking_status'],
    'bp_treatment': ['on_antihypertensive', 'antihypertensive_treatment', 'past_history', 'past_medical_history']
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
        for p in patterns:
            if normalize_str(p) == norm_name: return key  # change: exact match
    return None

def clean_numeric(val):
    if pd.isna(val) or str(val).strip() == "": return np.nan
    try:
        match = re.search(r"[-+]?\d*\.\d+|\d+", str(val))
        return float(match.group()) if match else np.nan
    except:
        return np.nan

def parse_sex(val):
    s = str(val).strip()
    if 'male' in s or 'M' in s.upper() or str(val) == '1':
        return 1  # 1 = male
    return 0      # 0 = female

def parse_history_boolean(key, val_list):
    combined_val = " ".join([str(v) for v in val_list if not pd.isna(v)])
    keywords = {
        'has_diabetes': ['diabetes', 'elevated_glucose', 'type_2'],
        'bp_treatment': ['antihypertensive', 'antihypertensive_drug', 'taking', 'hypertension', 'treatment'],
        'smoke': ['smoking', 'cigarette', 'smoke']
    }
    target_kws = keywords.get(key, [])
    is_positive = any(x in combined_val for x in ['yes', 'present', 'Yes', 'True', '1'])

    if target_kws:
        match = False
        for kw in target_kws:
            if kw in combined_val:
                match = True
                break
        if match:
            if 'absent' in combined_val and 'absent' + kw in combined_val:
                return 0
            return 1
    return 1 if is_positive else 0

# ==========================================
# 3. Core computation logic (pure computation)
# ==========================================

# Added: 2021 CKD-EPI eGFR estimation function
def calculate_ckd_epi_egfr(scr, age, sex):
    """
    Estimate eGFR from serum creatinine (2021 CKD-EPI, race-free)
    sex: 1=male, 0=female
    """
    if pd.isna(scr) or pd.isna(age) or pd.isna(sex):
        return np.nan
    
    # Smart unit conversion: umol/L is common domestically; treat values > 20 as umol/L and convert to mg/dL
    # (normal mg/dL range is about 0.5 - 1.5)
    scr_mgdl = scr / 88.4 if scr > 20 else scr
    
    if sex == 0:  # female
        k = 0.7
        alpha = -0.241
        multiplier = 1.012
    else:         # male
        k = 0.9
        alpha = -0.302
        multiplier = 1.0
        
    term1 = min(scr_mgdl / k, 1) ** alpha
    term2 = max(scr_mgdl / k, 1) ** -1.200
    term3 = 0.9938 ** age
    
    egfr = 142 * term1 * term2 * term3 * multiplier
    return egfr

def calculate_prevent_hf_risk(sex, age, sbp, has_diabetes, smoke, bmi, egfr, bp_treatment):
    if pd.isna(age) or pd.isna(sbp) or pd.isna(bmi) or pd.isna(egfr):
        return np.nan

    age = max(30, min(79, age))
    sbp = max(90, min(200, sbp))
    bmi = max(18.5, min(39.9, bmi))
    egfr = max(15, min(140, egfr))

    COEFFS = {
        'Intercept': [-4.310409, -3.946391],
        'Age_term': [0.8998235, 0.8972642],
        'SBP_min': [-0.4559771, -0.6811466],
        'SBP_max': [0.3576505, 0.3634461],
        'Diabetes': [1.038346, 0.923776],
        'Smoker': [0.583916, 0.5023736],
        'BMI_min': [-0.0072294, -0.0485841],
        'BMI_max': [0.2997706, 0.3726929],
        'eGFR_min': [0.7451638, 0.6926917],
        'eGFR_max': [0.0557087, 0.0251827],
        'Treated': [0.3534442, 0.2980922],
        'Treated_x_SBP_max': [-0.0981511, -0.0497731],
        'Age_x_SBP_max': [-0.0946663, -0.1289201],
        'Age_x_Diabetes': [-0.3581041, -0.3040924],
        'Age_x_Smoker': [-0.1159453, -0.1401688],
        'Age_x_BMI_max': [-0.003878, 0.0068126],
        'Age_x_eGFR_min': [-0.1884289, -0.1797778]
    }

    age_t = (age - 55) / 10.0
    sbp_min_t = (min(sbp, 110) - 110) / 20.0
    sbp_max_t = (max(sbp, 110) - 130) / 20.0
    bmi_min_t = (min(bmi, 30) - 25) / 5.0
    bmi_max_t = (max(bmi, 30) - 30) / 5.0
    egfr_min_t = (min(egfr, 60) - 60) / -15.0
    egfr_max_t = (max(egfr, 60) - 90) / -15.0

    idx = sex
    log_odds = COEFFS['Intercept'][idx]
    log_odds += COEFFS['Age_term'][idx] * age_t
    log_odds += COEFFS['SBP_min'][idx] * sbp_min_t
    log_odds += COEFFS['SBP_max'][idx] * sbp_max_t
    log_odds += COEFFS['Diabetes'][idx] * has_diabetes
    log_odds += COEFFS['Smoker'][idx] * smoke
    log_odds += COEFFS['BMI_min'][idx] * bmi_min_t
    log_odds += COEFFS['BMI_max'][idx] * bmi_max_t
    log_odds += COEFFS['eGFR_min'][idx] * egfr_min_t
    log_odds += COEFFS['eGFR_max'][idx] * egfr_max_t
    log_odds += COEFFS['Treated'][idx] * bp_treatment
    log_odds += COEFFS['Treated_x_SBP_max'][idx] * (bp_treatment * sbp_max_t)
    log_odds += COEFFS['Age_x_SBP_max'][idx] * (age_t * sbp_max_t)
    log_odds += COEFFS['Age_x_Diabetes'][idx] * (age_t * has_diabetes)
    log_odds += COEFFS['Age_x_Smoker'][idx] * (age_t * smoke)
    log_odds += COEFFS['Age_x_BMI_max'][idx] * (age_t * bmi_max_t)
    log_odds += COEFFS['Age_x_eGFR_min'][idx] * (age_t * egfr_min_t)

    risk = np.exp(log_odds) / (1 + np.exp(log_odds))
    return risk * 100

# ==========================================
# 4. Main data-processing routine
# ==========================================
def process_data(base_path, item_path, output_path):
    print("--- Starting AHA PREVENT heart failure risk assessment ---")

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
        print("Error: cannot read the file, or it is empty.")
        return

    # 1. Pivot to wide table
    item_df['cat'] = item_df['res_itm_name'].apply(get_category)
    filtered = item_df.dropna(subset=['cat', 'res_value']).copy()

    records = []
    grouped = filtered.groupby(['app_no', 'pat_id'])

    for (app_no, pat_id), group in grouped:
        row = {'app_no': str(app_no).strip(), 'pat_id': str(pat_id).strip()}
        data_pool = {k: [] for k in FIELD_MAPPING.keys()}
        for _, item in group.iterrows():
            data_pool[item['cat']].append(str(item['res_value']))

        # Numeric indicator extraction (one exam per day, take the value directly)
        height_vals = [clean_numeric(v) for v in data_pool.get('height', [])]
        height_vals = [v for v in height_vals if not pd.isna(v)]
        row['height'] = height_vals[0] if height_vals else np.nan

        weight_vals = [clean_numeric(v) for v in data_pool.get('weight', [])]
        weight_vals = [v for v in weight_vals if not pd.isna(v)]
        row['weight'] = weight_vals[0] if weight_vals else np.nan

        bmi_vals = [clean_numeric(v) for v in data_pool.get('BMI', [])]
        bmi_vals = [v for v in bmi_vals if not pd.isna(v)]
        row['bmi'] = bmi_vals[0] if bmi_vals else np.nan
        
        # Compute BMI from height and weight when missing
        if pd.isna(row['bmi']):
            h = row['height']
            w = row['weight']
            if pd.notna(h) and pd.notna(w):
                if h > 3: h /= 100.0
                if h > 0 and w > 0: row['bmi'] = w / (h * h)

        sbp_vals = [clean_numeric(v) for v in data_pool.get('sbp', [])]
        sbp_vals = [v for v in sbp_vals if not pd.isna(v)]
        row['sbp'] = sbp_vals[0] if sbp_vals else np.nan

        egfr_vals = [clean_numeric(v) for v in data_pool.get('egfr', [])]
        egfr_vals = [v for v in egfr_vals if not pd.isna(v)]
        row['egfr'] = egfr_vals[0] if egfr_vals else np.nan

        scr_vals = [clean_numeric(v) for v in data_pool.get('scr', [])]
        scr_vals = [v for v in scr_vals if not pd.isna(v)]
        row['scr'] = scr_vals[0] if scr_vals else np.nan
        
        row['has_diabetes'] = parse_history_boolean('has_diabetes', data_pool.get('has_diabetes', []))
        row['smoke'] = parse_history_boolean('smoke', data_pool.get('smoke', []))
        row['bp_treatment'] = parse_history_boolean('bp_treatment', data_pool.get('bp_treatment', []))
        
        records.append(row)

    wide_items = pd.DataFrame(records)
    base_df['app_no'] = base_df['app_no'].astype(str).str.strip()
    base_df['pat_id'] = base_df['pat_id'].astype(str).str.strip()

    if wide_items.empty:
        print("No valid indicator data extracted.")
        return

    # Merge basic information (map app_no to app_no)
    final = pd.merge(wide_items, base_df, left_on=['app_no', 'pat_id'], right_on=['app_no', 'pat_id'], how='inner')

    results = []
    model_name_str = "AHA PREVENT Heart Failure Risk Prediction"
    gauge_ref = "0-1%|1-5%|5-100%"

    for _, r in final.iterrows():
        age = clean_numeric(r.get('app_pat_age'))
        sex_val = parse_sex(r.get('pat_sex'))
        
        height = r.get('height', np.nan)
        weight = r.get('weight', np.nan)
        bmi = r.get('bmi', np.nan)
        sbp = r.get('sbp', np.nan)
        egfr = r.get('egfr', np.nan)
        scr = r.get('scr', np.nan)
        has_diabetes = r.get('has_diabetes', 0)
        smoke = r.get('smoke', 0)
        bp_treatment = r.get('bp_treatment', 0)

        # Compute eGFR when missing but creatinine, age and sex are available
        if pd.isna(egfr) and pd.notna(scr) and pd.notna(age):
            egfr = calculate_ckd_epi_egfr(scr=scr, age=age, sex=sex_val)

        risk = calculate_prevent_hf_risk(
            sex=sex_val, age=age, sbp=sbp,
            has_diabetes=has_diabetes, smoke=smoke,
            bmi=bmi, egfr=egfr, bp_treatment=bp_treatment
        )

        # Risk-tier determination
        code = ""
        risk_level = ""
        if pd.isna(risk):
            risk_str = "data_missing_cannot_compute"
            risk_level = "-"
            code = "-"
        else:
            risk_str = f"{risk:.2f}%"
            if risk < 1:
                risk_level = "low risk"
                code = "I50.1.1"
            elif 1 <= risk <= 5:
                risk_level = "medium risk"
                code = "I50.1.2"
            else:
                risk_level = "high risk"
                code = "I50.1.3"

        results.append({
            'pat_id': r.get('pat_id'),
            'app_no': r.get('app_no'),
            'pat_name': r.get('pat_name', 'unknown'),
            'sex': r.get('pat_sex', 'unknown'),
            'age': int(age) if pd.notna(age) else '-',
            
            # --- All raw indicators used in the computation, fully expanded ---
            'height': round(height, 2) if pd.notna(height) else '-',
            'weight': round(weight, 2) if pd.notna(weight) else '-',
            'BMI': round(bmi, 2) if pd.notna(bmi) else '-',
            'systolic_bp (SBP)': round(sbp, 2) if pd.notna(sbp) else '-',
            'creatinine (Scr)': round(scr, 2) if pd.notna(scr) else '-',
            'eGFR': round(egfr, 2) if pd.notna(egfr) else '-',
            'smoking': 'yes' if smoke == 1 else 'no',
            'diabetes': 'yes' if has_diabetes == 1 else 'no',
            'antihypertensive_treatment': 'yes' if bp_treatment == 1 else 'no',
            # -----------------------------------
            
            'model_name': model_name_str,
            'risk_probability': risk_str,
            'prediction_horizon':'next 10 years',
            'risk_level': risk_level,
            'code': code,
            'gauge_reference': gauge_ref
        })

    if results:
        res_df = pd.DataFrame(results)
        
        # Reorder columns to show creatinine (Scr)
        main_cols = ['pat_id', 'app_no', 'pat_name', 'sex', 'age', 
                     'height', 'weight', 'BMI', 'systolic_bp (SBP)', 'creatinine (Scr)', 'eGFR', 
                     'smoking', 'diabetes', 'antihypertensive_treatment',
                     'model_name', 'risk_probability', 'risk_level', 'code', 'gauge_reference']
        
        other_cols = [c for c in res_df.columns if c not in main_cols]
        res_df = res_df[main_cols + other_cols]

        res_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        
        success_count = res_df[res_df['risk_probability'] != 'data_missing_cannot_compute'].shape[0]
        skip_count = len(res_df) - success_count
        
        print(f"--- Assessment complete! ---")
        print(f"processed {len(res_df)}: {success_count} computed, {skip_count} skipped for missing indicators.")
        print(f"results exported to {output_path}")
    else:
        print("--- Assessment complete, no results ---")

if __name__ == "__main__":
    process_data(
        "base.csv",
        "item_result.csv",
        "aha_prevent_hf_result.csv"
    )

