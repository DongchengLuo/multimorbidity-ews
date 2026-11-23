import pandas as pd
import numpy as np
import re
import math

# ==========================================
# 1. Field mapping (see the precise field-matching reference table)
# ==========================================
FIELD_MAPPING = {
    'age': ['app_pat_age', 'age'],
    'gender': ['pat_sex', 'sex'],
    'sbp': ['systolic_bp', 'blood_pressure'],
    'smoke': ['smoking_history'],
    'has_diabetes': ['past_history', 'past_medical_history', 'diabetes'],
    'has_cvd': ['past_history', 'past_medical_history', 'cardiovascular_disease', 'coronary_heart_disease'],
    'has_af': ['past_history', 'past_medical_history', 'atrial_fibrillation'],
    'has_lvh': ['left_ventricular_hypertrophy', 'lvh'],
    'bp_treatment': ['past_history', 'past_medical_history', 'antihypertensive_treatment', 'antihypertensive_drug']
}

# ==========================================
# 2. Basic helper functions
# ==========================================
def normalize_str(s):
    if pd.isna(s):
        return ""
    return re.sub(r'[\(\)\（\）\[\]\【\】\{\}\s]', '', str(s)).lower()

def get_category(name):
    norm_name = normalize_str(name)
    for key, patterns in FIELD_MAPPING.items():
        for p in patterns:
            # Switch to exact match (equal after stripping brackets and spaces)
            if normalize_str(p) == norm_name:
                return key
    return None

def clean_numeric(val):
    if pd.isna(val) or str(val).strip() == "":
        return np.nan
    try:
        # Support "150/90" blood pressure and take the max (systolic)
        if '/' in str(val):
            return max([float(x.strip()) for x in str(val).split('/') if x.strip()])
        match = re.search(r"[-+]?\d*\.\d+|\d+", str(val))
        return float(match.group()) if match else np.nan
    except:
        return np.nan

# [Change 1]: added an exclude_keywords parameter
def check_keyword(text, keywords, exclude_keywords=None):
    text = normalize_str(text)
    
    # Check for negation first
    if exclude_keywords:
        if any(normalize_str(ekw) in text for ekw in exclude_keywords):
            return False # any negation word means absent
            
    return any(normalize_str(kw) in text for kw in keywords)

# ==========================================
# 3. Framingham stroke core algorithm (1991)
# ==========================================
def calculate_framingham_stroke(row):
    """Returns (risk probability, risk tier, computation details, code, recommendation)"""
    
    # Extract basic indicators
    age = clean_numeric(row.get('app_pat_age'))
    sex_raw = str(row.get('pat_sex', ''))
    is_male = None
    if any(x in sex_raw.lower() for x in ['male', 'm', '1']):
        is_male = True
    elif any(x in sex_raw.lower() for x in ['female', 'f', '0']):
        is_male = False
        
    sbp = row.get('sbp_val')
    hyp_rx = row.get('hyp_rx', False)
    dm = row.get('dm', False)
    cigs = row.get('smoke', False)
    cvd = row.get('cvd', False)
    af = row.get('af', False)
    lvh = row.get('lvh', False)
    
    # 1. Required-indicator and range validation
    required_missing = []
    if pd.isna(age):
        required_missing.append('age')
    if is_male is None:
        required_missing.append('sex')
    if pd.isna(sbp):
        required_missing.append('systolic_bp')
        
    if required_missing:
        return None, "-", f"missing required indicators: {','.join(required_missing)}", "", "please provide complete exam indicators for an accurate assessment"
    
    if not (55 <= age <= 84):
        return None, "-", f"age {age} outside the Framingham range (55-84 only)", "", "age outside the model range"
    
    # 2. Core computation
    hyp_val = 1 if hyp_rx else 0
    dm_val = 1 if dm else 0
    cigs_val = 1 if cigs else 0
    cvd_val = 1 if cvd else 0
    af_val = 1 if af else 0
    lvh_val = 1 if lvh else 0
    
    try:
        if is_male:
            l = (0.0505 * age + 0.0140 * sbp + 0.3263 * hyp_val + 
                 0.3384 * dm_val + 0.5147 * cigs_val + 0.5195 * cvd_val + 
                 0.6061 * af_val + 0.8415 * lvh_val)
            m, s_10 = 5.6770, 0.9044
        else:
            interaction = sbp * hyp_val
            l = (0.0657 * age + 0.0197 * sbp + 2.5432 * hyp_val + 
                 (-0.0134 * interaction) + 0.5442 * dm_val + 
                 0.5294 * cigs_val + 0.4326 * cvd_val + 1.1497 * af_val + 
                 0.8488 * lvh_val)
            m, s_10 = 7.5766, 0.9353
            
        prob = (1 - (s_10 ** math.exp(l - m))) * 100
        
        # 3. Risk tiering and codes
        if prob < 10:
            level, code, suggest = "low risk", "I64.1.1", "maintain a healthy lifestyle; regular follow-up"
        elif 10 <= prob < 20:
            level, code, suggest = "medium risk", "I64.1.2", "control blood pressure and risk factors; consult a specialist"
        else:
            level, code, suggest = "high risk", "I64.1.3", "significant risk; seek medical assessment and intervention promptly"
            
        details = f"computed (linear predictor:{round(l, 4)}); antihypertensive:{hyp_val}, diabetes:{dm_val}, smoking:{cigs_val}, CHD:{cvd_val}, AF:{af_val}, LVH:{lvh_val}"
        
        return round(prob, 2), level, details, code, suggest
        
    except Exception as e:
        return None, "computation_error", str(e), "", "internal error during model computation"

# ==========================================
# 4. Main data-processing routine
# ==========================================
def process_data(base_path, item_path, output_path):
    print("--- Starting [Framingham 10-year stroke risk (1991)] model ---")
    
    def read_csv_safe(path):
        for enc in ['gbk', 'utf-8-sig', 'utf-8']:
            try:
                return pd.read_csv(path, encoding=enc, low_memory=False)
            except:
                continue
        return pd.DataFrame()
    
    base_df = read_csv_safe(base_path)
    item_df = read_csv_safe(item_path)
    
    if base_df.empty or item_df.empty:
        print("Error: cannot read the input file.")
        return
        
    # Preprocess the item table
    item_df['cat'] = item_df['res_itm_name'].apply(get_category)
    filtered = item_df.dropna(subset=['cat', 'res_value']).copy()
    
    records = []
    print("Analyzing indicator data...")
    
    # Common negation words
    neg_words = ['absent', 'no', 'not']
    
    # Aggregate data per person
    for (app_no, pat_id), group in filtered.groupby(['app_no', 'pat_id']):
        row = {'app_no': str(app_no).strip(), 'pat_id': str(pat_id).strip()}
        data_pool = {k: [] for k in FIELD_MAPPING.keys()}
        
        for _, item in group.iterrows():
            data_pool[item['cat']].append(str(item['res_value']))
        
        # Numeric and identifier extraction (read single values, no aggregation)
        sbp_vals = data_pool.get('sbp', [])
        row['sbp_val'] = clean_numeric(sbp_vals[0]) if sbp_vals else np.nan
        
        # Merge long history text
        history_text = " ".join(data_pool.get('has_diabetes', []) + 
                               data_pool.get('has_cvd', []) + 
                               data_pool.get('has_af', []) + 
                               data_pool.get('bp_treatment', []))
        
        # [Change 2]: add exclude_keywords to all boolean checks to prevent misclassification of other fields
        row['dm'] = check_keyword(history_text, ['diabetes', 'high_glucose'], exclude_keywords=neg_words)
        row['af'] = check_keyword(history_text, ['atrial_fibrillation', 'af'], exclude_keywords=neg_words)
        row['cvd'] = check_keyword(history_text, ['coronary_heart_disease', 'heart_failure', 'myocardial_infarction', 'intermittent_claudication', 'cvd'], exclude_keywords=neg_words)
        row['hyp_rx'] = check_keyword(history_text, ['antihypertensive_drug', 'antihypertensive_treatment', 'on_antihypertensive'], exclude_keywords=neg_words)
        row['smoke'] = check_keyword(" ".join(data_pool.get('smoke', [])), ['yes', 'present', 'smoking', '1'], exclude_keywords=neg_words)
        row['lvh'] = check_keyword(" ".join(data_pool.get('has_lvh', [])), ['yes', 'present', 'left_ventricular_hypertrophy', 'lvh'], exclude_keywords=neg_words)
        
        records.append(row)
        
    wide_items = pd.DataFrame(records)
    
    # Normalize field types for merging
    base_df['app_no'] = base_df['app_no'].astype(str).str.strip()
    base_df['pat_id'] = base_df['pat_id'].astype(str).str.strip()
    
    # Merge base-table information
    final = pd.merge(wide_items, base_df, on=['app_no', 'pat_id'], how='inner')
    
    results = []
    print("Generating final assessment results...")
    
    # Fixed gauge reference values
    GAUGE_REF = "0%-10%|10%-20%|20%-100%"
    
    for _, r in final.iterrows():
        # Run computation
        prob, risk_level, details, code, suggest = calculate_framingham_stroke(r)
        
        # Add all information to the result record
        results.append({
            'model_name': 'Framingham 10-Year Stroke Risk Model (1991)',
            'pat_id': r['pat_id'],
            'app_no': r['app_no'],
            'pat_name': r.get('pat_name', 'unknown'),
            'sex': r.get('pat_sex', ''),
            'age': r.get('app_pat_age', '-'),
            'risk_probability': f"{prob:.2f}%" if prob is not None else "-",
            'risk_level': risk_level,
            'prediction_horizon':'next 10 years',
            'code': code,
            'gauge_reference': GAUGE_REF,
            'determination_details': details,
            'recommendation': suggest,
            'prediction_horizon': 'next 10 years',
            # ====== Raw values of all inputs involved in the computation ======
            'systolic_bp (mmHg)': r['sbp_val'],
            'on_antihypertensive_drug': "yes" if r['hyp_rx'] else "no",
            'diabetes_history': "yes" if r['dm'] else "no",
            'smoking_history': "yes" if r['smoke'] else "no",
            'cardiovascular_history': "yes" if r['cvd'] else "no",
            'atrial_fibrillation_history': "yes" if r['af'] else "no",
            'left_ventricular_hypertrophy': "yes" if r['lvh'] else "no",
        })
        
    if results:
        res_df = pd.DataFrame(results)
        res_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"--- Exported {len(res_df)} records to {output_path} ---")
    else:
        print("No valid results generated.")

if __name__ == "__main__":
    process_data(
        "base.csv",
        "item_result.csv",
        "framingham_stroke_result.csv"
    )
