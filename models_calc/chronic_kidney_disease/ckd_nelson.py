import pandas as pd
import numpy as np
import re

# ==========================================
# 1. Field mapping (unchanged)
# ==========================================
FIELD_MAPPING = {
    'age': ['app_pat_age', 'age'],
    'gender': ['pat_sex', 'sex'],
    'egfr': ['eGFR', 'eGFR (CKD-EPI)', 'egfr'],
    'scr': ['creatinine (Cr)', 'creatinine', 'serum_creatinine'],
    'acr': ['uacr', 'urine_albumin_creatinine_ratio', 'acr'],  
    'BMI': ['(body mass index) BMI', 'bmi', 'BMI'],
    'height': ['height'],  
    'weight': ['weight'],  
    'Hba1c': ['HbA1c', 'GHb', 'hba1c'],  
    'smoke': ['smoking_history', 'smoke', 'smoker'],
    'has_diabetes': ['past_history', 'type2_diabetes', 'diabetes_history', 'diabetes'],
    'has_cvd': ['has_cardiovascular_disease', 'past_history', 'cvd'],
    'has_hypertension': ['has_hypertension', 'hypertension_history', 'past_history'],
    'insulin': ['insulin', 'on_insulin', 'insulin_therapy'],  
    'oral_meds': ['oral_meds', 'oral_hypoglycemic', 'glucose_lowering_drug']  
}

# ==========================================
# 2. Basic utility functions (unchanged)
# ==========================================
def normalize_str(s):
    if pd.isna(s): return ""
    return re.sub(r'[\(\)\（\）\[\]\【\】\{\}\s]', '', str(s)).lower()

def check_target_with_negation(text, target_keywords, window_size=5):
    if pd.isna(text) or text == "": return 0
    text = normalize_str(text)
    negations = ['absent', 'no', 'not', 'exclude', 'deny', 'negative']
    for kw in target_keywords:
        kw = normalize_str(kw)
        if kw in text:
            idx = text.find(kw)
            prefix = text[max(0, idx - window_size):idx]
            if any(neg in prefix for neg in negations): continue
            return 1
    return 0

def calculate_egfr_ckd_epi(scr, age, sex):
    if any(pd.isna(x) for x in [scr, age, sex]): return np.nan
    s_cr = scr / 88.4 if scr > 50 else scr
    kappa = 0.7 if sex == 1 else 0.9  
    alpha = -0.329 if sex == 1 else -0.411
    egfr = 141 * min(s_cr/kappa, 1)**alpha * max(s_cr/kappa, 1)**-1.209 * (0.993**age)
    if sex == 1: egfr *= 1.018
    return egfr

def clean_numeric(val):
    if pd.isna(val) or str(val).strip() == "": return np.nan
    try:
        match = re.search(r"[-+]?\d*\.\d+|\d+", str(val))
        return float(match.group()) if match else np.nan
    except: return np.nan

def parse_bool(val):
    if pd.isna(val): return np.nan
    s = str(val).strip().lower()
    if s in ['1', '1.0', 'true', 'yes', 'y', 'present', 'positive', '+']: return 1
    if s in ['0', '0.0', 'false', 'no', 'n', 'absent', 'negative', '-']: return 0
    return np.nan

def parse_sex(val):
    if pd.isna(val): return np.nan
    s = str(val).strip().lower()
    if 'female' in s or 'female' in s or s == 'f' or s == '1': return 1
    if 'male' in s or 'male' in s or s == 'm' or s == '0': return 0
    return np.nan

# ==========================================
# 3. Nelson equation (original structure fully restored)
# ==========================================
def calculate_nelson_risk(inputs):
    age = inputs.get('age')
    sex = inputs.get('gender')  
    race = 0  
    egfr = inputs.get('egfr')
    acr = inputs.get('acr')
    bmi = inputs.get('BMI')
    diabetes = inputs.get('has_diabetes')
    cvd = inputs.get('has_cvd', 0)
    smoker = inputs.get('smoke', 0)
    htn = inputs.get('has_hypertension', 0)

    if any(pd.isna(x) for x in [age, sex, egfr, acr, bmi, diabetes]):
        return np.nan, "cannot_compute", "core_indicator_missing"

    acr_val = max(acr, 0.1)  
    log_acr = np.log10(acr_val)
    t_age = age / 5.0 - 11.0
    t_bmi = bmi / 5.0 - 5.4
    t_egfr1 = 15.0 - min(egfr, 90) / 5.0
    t_egfr2 = max(0, egfr - 90) / 5.0

    if diabetes == 0:
        exp_log_acr = (0.6754442 + 0.0222581 * t_age + 0.0459020 * sex - 0.0340495 * race
                       + 0.0085871 * t_egfr1 - 0.0275825 * t_egfr2 + 0.0495695 * cvd
                       + 0.0381086 * smoker + 0.1286836 * htn + 0.0218783 * t_bmi)
        xb = (-3.609661 + 0.2582196 * t_age + 0.1821665 * sex + 0.1808945 * race
              + 0.4581006 * t_egfr1 - 0.3159218 * t_egfr2 + 0.1953927 * cvd
              + 0.1213741 * smoker + 0.3543645 * htn + 0.0630538 * t_bmi
              + 0.3519087 * (log_acr - exp_log_acr))
        risk = 1.0 - np.exp(- (5 ** 1.055408) * np.exp(xb))
        return risk, "Nelson equation (non-diabetic)", ""
    else:
        hba1c = inputs.get('Hba1c')
        insulin = inputs.get('insulin', 0)
        oral_meds = inputs.get('oral_meds', 0)
        if pd.isna(hba1c): return np.nan, "cannot_compute", "diabetic patient missing HbA1c"
        no_dm_meds = 1 if (insulin == 0 and oral_meds == 0) else 0
        t_hba1c = hba1c - 7.0
        xb = (-2.647004 + 0.1351572 * t_age + 0.1381975 * sex + 0.0920208 * race
              + 0.3546697 * t_egfr1 - 0.1525133 * t_egfr2 + 0.1870637 * cvd
              + 0.0619679 * t_hba1c + 0.1078296 * insulin - 0.1509440 * no_dm_meds
              + 0.0239590 * t_hba1c * insulin + 0.0398424 * t_hba1c * no_dm_meds
              - 0.0008400 * smoker + 0.3653268 * htn + 0.0503060 * t_bmi
              + 0.3737905 * (log_acr - 1.0))
        risk = 1.0 - np.exp(- (5 ** 0.9766551) * np.exp(xb))
        return risk, "Nelson equation (diabetic)", ""

# ==========================================
# 4. Main routine (revised matching and missing-value logic)
# ==========================================
def process_data(base_path, item_path, output_path):
    def read_csv_safe(path):
        for enc in ['gbk', 'utf-8-sig', 'utf-8']:
            try: return pd.read_csv(path, encoding=enc, low_memory=False)
            except: continue
        return pd.DataFrame()

    base_df = read_csv_safe(base_path)
    item_df = read_csv_safe(item_path)
    if base_df.empty or item_df.empty: return

    for df in [base_df, item_df]:
        if 'app_no' in df.columns: df['app_no'] = df['app_no'].astype(str).str.strip()
        if 'pat_id' in df.columns: df['pat_id'] = df['pat_id'].astype(str).str.strip()

    # --- Core change: switch to exact match '==' ---
    item_df['cat'] = item_df['res_itm_name'].apply(
        lambda x: next((k for k, pats in FIELD_MAPPING.items() if any(normalize_str(p) == normalize_str(x) for p in pats)), None)
    )
    filtered = item_df.dropna(subset=['cat']).copy()

    records = []
    for (app_no, pat_id), group in filtered.groupby(['app_no', 'pat_id']):
        row = {'app_no': app_no, 'pat_id': pat_id}
        data_pool = {k: [] for k in FIELD_MAPPING.keys()}
        for _, item in group.iterrows():
            data_pool[item['cat']].append(item['res_value'])

        # ==========================================
        # Fix: drop averaging and take the first result
        # ==========================================
        for field in ['age', 'egfr', 'scr', 'acr', 'Hba1c', 'BMI', 'height', 'weight']:
            vals = data_pool.get(field, [])
            row[field] = clean_numeric(vals[0]) if vals else np.nan
        
        if pd.isna(row['BMI']):
            h, w = row.get('height'), row.get('weight')
            # Add np.isnan checks to avoid errors
            if not pd.isna(h) and not pd.isna(w) and h > 0 and w > 0:
                if h > 3: h /= 100.0
                row['BMI'] = w / (h * h)

        # ==========================================
        # Fix: for booleans too, skip iteration and take the first result
        # ==========================================
        for k in ['has_diabetes', 'has_cvd', 'smoke', 'has_hypertension', 'insulin', 'oral_meds']:
            raw_vals = data_pool.get(k, [])
            row[f'item_{k}'] = parse_bool(raw_vals[0]) if raw_vals else np.nan

        g_vals = data_pool.get('gender', [])
        row['sex_item'] = parse_sex(g_vals[0]) if g_vals else np.nan
        
        records.append(row)

    wide_items = pd.DataFrame(records)
    base_df.rename(columns={'app_no': 'app_no'}, inplace=True)
    final = pd.merge(base_df, wide_items, on=['app_no', 'pat_id'], how='left')

    results = []
    for _, r in final.iterrows():
        hist_text = str(r.get('past_history', '')) + str(r.get('disease_history', ''))
        age_val = r.get('age') if not pd.isna(r.get('age')) else clean_numeric(r.get('app_pat_age'))
        sex_val = r.get('sex_item') if not pd.isna(r.get('sex_item')) else parse_sex(r.get('pat_sex'))
        
        egfr_val = r.get('egfr')
        if pd.isna(egfr_val) and not pd.isna(r.get('scr')):
            egfr_val = calculate_egfr_ckd_epi(r.get('scr'), age_val, sex_val)

        def final_judge(key, kws):
            return 1 if (r.get(f'item_{key}') == 1 or check_target_with_negation(hist_text, kws)) else (0 if r.get(f'item_{key}') == 0 else 0)

        inputs = {
            'age': age_val,
            'gender': sex_val,
            'egfr': egfr_val,
            'acr': r.get('acr'),
            'BMI': r.get('BMI'),
            'has_diabetes': final_judge('has_diabetes', ['diabetes']),
            'has_cvd': final_judge('has_cvd', ['cardiovascular', 'coronary_heart_disease', 'cerebral_infarction', 'myocardial_infarction', 'cvd']),
            'smoke': final_judge('smoke', ['smoking', 'cigarette', 'smoke']),
            'has_hypertension': final_judge('has_hypertension', ['hypertension', 'high_bp']),
            'Hba1c': r.get('Hba1c'),
            'insulin': 1 if r.get('item_insulin') == 1 else 0,
            'oral_meds': 1 if r.get('item_oral_meds') == 1 else 0
        }

        risk, model_name, remarks = calculate_nelson_risk(inputs)

        # --- Restore the original result-dict structure ---
        results.append({
            'app_no': r['app_no'],
            'pat_id': r['pat_id'],
            'pat_name': r.get('pat_name', r.get('name', 'unknown')),
            'model_name': 'CKD-Nelson',
            # Fix: replace "-" with the clearer "missing" to match other scripts
            'risk_probability': f"{risk * 100:.2f}%" if not pd.isna(risk) else "missing",
            'prediction_horizon':'next 5 years',
            'note (abnormal reason)': remarks,
            'age': inputs['age'],
            'sex (1 female, 0 male)': inputs['gender'],
            'eGFR': inputs['egfr'],
            'ACR': inputs['acr'],
            'BMI': inputs['BMI'],
            'diabetes (1 yes, 0 no)': inputs['has_diabetes'],
            'cardiovascular_disease (1 yes, 0 no)': inputs['has_cvd'],
            'hypertension (1 yes, 0 no)': inputs['has_hypertension'],
            'smoking_history (1 yes, 0 no)': inputs['smoke'],
            'HbA1c': inputs['Hba1c'],
            'insulin (1 yes, 0 no)': inputs['insulin'],
            'oral_hypoglycemic (1 yes, 0 no)': inputs['oral_meds']
        })

    if results:
        pd.DataFrame(results).to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"Assessment complete (including records with missing indicators), exported to {output_path}")

if __name__ == "__main__":
    process_data("base.csv", "item_result.csv", "nelson_ckd_risk_result.csv")
