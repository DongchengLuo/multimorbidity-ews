import pandas as pd
import numpy as np
import re

# ==========================================
# 1. Field mapping (strictly per the Excel reference table)
# ==========================================
FIELD_MAPPING = {
    # --- Basic information ---
    'age': ['app_pat_age', 'age'],
    'gender': ['pat_sex', 'sex'],

    # --- Body measurements ---
    'BMI': ['(body mass index) BMI', 'bmi', 'BMI'],
    'height': ['height'],
    'weight': ['weight'],
    'wc': ['waist'],

    # --- Blood pressure ---
    'sbp': ['systolic_bp', 'blood_pressure'],
    'dbp': ['diastolic_bp', 'blood_pressure'],

    # --- Medical history and lifestyle ---
    'has_hypertension': ['past_history', 'past_medical_history'],
    'diabetes_family_history': ['diabetes_family_history', 'family_history'],
    'exercise': ['physical_exercise', 'physical_activity', 'exercise']
}

MODEL_NAME = "ADA-diabetes score"

# ==========================================
# 2. Basic cleaning and parsing utilities
# ==========================================
def normalize_str(s):
    if pd.isna(s): return ""
    return re.sub(r'[\(\)\（\）\[\]\【\】\{\}\s]', '', str(s)).lower()

def get_category(name):
    """Exact-match the standard label from the field name"""
    norm_name = normalize_str(name)
    for key, patterns in FIELD_MAPPING.items():
        for p in patterns:
            if normalize_str(p) == norm_name:
                return key
    return None

def clean_numeric(val):
    if pd.isna(val) or str(val).strip() == "": return np.nan
    try:
        match = re.search(r"[-+]?\d*\.\d+|\d+", str(val))
        return float(match.group()) if match else np.nan
    except:
        return np.nan

def parse_sex(val):
    s = str(val).strip().lower()
    if s in ['male', 'm', '1']: return 'Male'
    if s in ['female', 'f', '2', '0']: return 'Female'
    return None

def parse_bool_indicator(val_list, keywords=None):
    """Extract booleans: 1 = yes, 0 = no, NaN = missing"""
    if not val_list: return np.nan
    valid_vals = [str(v) for v in val_list if pd.notna(v) and str(v).strip() != '']
    if not valid_vals: return np.nan
    
    combined_str = " ".join(valid_vals).lower()
    
    # Exclude negations first
    if any(x in combined_str for x in ['no', 'absent', 'false', 'negative', 'neg', '-']): 
        return 0
    # Look for specific keywords
    if keywords and any(kw.lower() in combined_str for kw in keywords): 
        return 1
    # Generic affirmative words
    if any(x in combined_str for x in ['yes', 'present', 'true', 'positive', 'pos', '+']): 
        return 1
        
    return np.nan

def extract_smart_value(val_list, mode='mean'):
    """
    Extract values from compound data such as '120/80'
    """
    valid_nums = []
    for v in val_list:
        if pd.isna(v): continue
        s = str(v)
        nums = re.findall(r"(\d+\.?\d*)", s)
        nums = [float(x) for x in nums]
        if not nums: continue

        if len(nums) >= 2:
            if mode == 'max': valid_nums.append(max(nums[:2]))
            elif mode == 'min': valid_nums.append(min(nums[:2]))
            else: valid_nums.append(nums[0])
        else:
            valid_nums.append(nums[0])

    if not valid_nums: return np.nan
    return valid_nums[0]

# ==========================================
# 3. Diabetes risk score (ADA-diabetes score)
# ==========================================
def calculate_diabetes_risk_score(age, sex, family_history, hypertension_status, bmi, waist_cm, is_active):
    # Do not compute when core indicators (age, sex, BMI) are missing
    if pd.isna(age) or sex is None or pd.isna(bmi):
        return None, {}

    score = 0
    debug_info = {}

    # 1. Age
    s = 0
    if age >= 60: s = 3
    elif age >= 50: s = 2
    elif age >= 40: s = 1
    score += s

    # 2. Sex
    s = 1 if (sex == 'Male') else 0
    score += s

    # 3. Family history
    s = 1 if family_history else 0
    score += s

    # 4. Hypertension
    s = 1 if hypertension_status else 0
    score += s

    # 5. Obesity status
    waist_inch = waist_cm / 2.54 if (pd.notna(waist_cm) and waist_cm > 0) else 0
    s = 0
    
    is_extreme = (bmi >= 40) or \
                 (sex == 'Male' and waist_inch >= 50) or \
                 (sex != 'Male' and waist_inch >= 49)

    is_obese = (30 <= bmi < 40) or \
               (sex == 'Male' and 40 <= waist_inch < 50) or \
               (sex != 'Male' and 35 <= waist_inch < 49)

    is_overweight = (25 <= bmi < 30) or \
                    (sex == 'Male' and 37 <= waist_inch < 40) or \
                    (sex != 'Male' and 31.5 <= waist_inch < 35)

    if is_extreme: s = 3
    elif is_obese: s = 2
    elif is_overweight: s = 1

    score += s

    # 6. Physical activity
    s = -1 if is_active else 0
    score += s

    return score, debug_info

# ==========================================
# 4. Main data-processing routine
# ==========================================
def process_data(base_path, item_path, output_path):
    print(f"--- Starting {MODEL_NAME} screening model ---")

    def read_csv_safe(path):
        for enc in ['gbk', 'utf-8-sig', 'utf-8']:
            try: return pd.read_csv(path, encoding=enc, low_memory=False)
            except: continue
        return pd.DataFrame()

    base_df = read_csv_safe(base_path)
    item_df = read_csv_safe(item_path)

    if base_df.empty:
        print("Error: base data file could not be read or is empty")
        return

    # 1. Pivot to wide table
    records = []
    if not item_df.empty:
        item_df['cat'] = item_df['res_itm_name'].apply(get_category)
        filtered = item_df.dropna(subset=['cat', 'res_value']).copy()

        for (app_no, pat_id), group in filtered.groupby(['app_no', 'pat_id']):
            row = {'app_no': str(app_no).strip(), 'pat_id': str(pat_id).strip()}
            data_pool = {k: [] for k in FIELD_MAPPING.keys()}

            for _, item in group.iterrows():
                data_pool[item['cat']].append(str(item['res_value']))

            # Numeric extraction
            row['sbp'] = extract_smart_value(data_pool.get('sbp', []), mode='max')
            row['dbp'] = extract_smart_value(data_pool.get('dbp', []), mode='min')
            row['BMI'] = extract_smart_value(data_pool.get('BMI', []), mode='mean')
            row['wc'] = extract_smart_value(data_pool.get('wc', []), mode='mean')

            # Compute BMI if missing
            if pd.isna(row['BMI']):
                h = extract_smart_value(data_pool.get('height', []), mode='mean')
                w = extract_smart_value(data_pool.get('weight', []), mode='mean')
                if pd.notna(h) and pd.notna(w) and h > 0:
                    h_m = h / 100.0 if h > 3 else h
                    row['BMI'] = w / (h_m * h_m)

            # Status extraction
            row['hist_fam'] = parse_bool_indicator(data_pool.get('diabetes_family_history', []), ['diabetes'])
            row['hist_hyp'] = parse_bool_indicator(data_pool.get('has_hypertension', []), ['hypertension', 'antihypertensive_drug'])
            row['exercise_raw'] = parse_bool_indicator(data_pool.get('exercise', []), ['yes', 'present', 'active', 'often', 'exercise'])

            records.append(row)

    wide_items = pd.DataFrame(records) if records else pd.DataFrame(columns=['app_no', 'pat_id'])

    # 2. Merge information (left join)
    base_df['app_no'] = base_df['app_no'].astype(str).str.strip()
    base_df['pat_id'] = base_df['pat_id'].astype(str).str.strip()
    final = pd.merge(base_df, wide_items, on=['app_no', 'pat_id'], how='left')

    results = []
    gauge_reference_str = "0-3|4-4|5-10"

    for _, r in final.iterrows():
        age = clean_numeric(r.get('app_pat_age'))
        sex = parse_sex(r.get('pat_sex', ''))
        
        bmi = r.get('BMI')
        wc = r.get('wc')
        sbp = r.get('sbp')
        dbp = r.get('dbp')
        
        hist_fam = r.get('hist_fam')
        hist_hyp = r.get('hist_hyp')
        exe_raw = r.get('exercise_raw')

        # ----------------------------------------------------
        # Special-case handling
        # ----------------------------------------------------
        # Hypertension determination: history OR (SBP>=140) OR (DBP>=90)
        is_hypertensive = False
        if hist_hyp == 1: is_hypertensive = True
        if pd.notna(sbp) and sbp >= 140: is_hypertensive = True
        if pd.notna(dbp) and dbp >= 90: is_hypertensive = True
        
        # Physical exercise defaults to True when missing
        if pd.isna(exe_raw):
            is_active = True
            exe_display = "yes (defaulted to yes when unrecognized)"
        elif exe_raw == 1:
            is_active = True
            exe_display = "yes"
        else:
            is_active = False
            exe_display = "no"

        # Run score computation
        score, details = calculate_diabetes_risk_score(
            age, sex, (hist_fam == 1), is_hypertensive, bmi, wc, is_active
        )

        # Risk tiering and encoding
        if score is not None:
            if score >= 5:
                risk_level, suggestion, code = "high risk", "recommend immediate fasting glucose testing", "E14.5.3"
            elif score >= 4:
                risk_level, suggestion, code = "medium risk", "recommend glucose screening", "E14.5.2"
            else:
                risk_level, suggestion, code = "low risk", "maintain a healthy lifestyle", "E14.5.1"
        else:
            risk_level, suggestion, code = "-", "-", "-"

        # Format for display
        def fmt_in(v, is_num=False):
            if pd.isna(v) or v == "": return "missing"
            return f"{v:.1f}" if is_num else str(v)
            
        def fmt_bool(v):
            if pd.isna(v): return "missing"
            return "yes" if v == 1 else "no"

        results.append({
            'pat_id': r.get('pat_id', ''),
            'app_no': r.get('app_no', ''),
            'pat_name': r.get('pat_name', 'unknown'),
            'model_name': MODEL_NAME,
            'sex': "male" if sex == 'Male' else ("female" if sex == 'Female' else "missing"),
            'age': int(age) if pd.notna(age) else "missing",
            # --- Raw indicator section (missing shown as "missing") ---
            'BMI': fmt_in(bmi, is_num=True),
            'blood_pressure': f"{int(sbp)}/{int(dbp)}" if pd.notna(sbp) and pd.notna(dbp) and sbp > 0 else "missing",
            'hypertension_history': fmt_bool(hist_hyp),
            'diabetes_family_history': fmt_bool(hist_fam),
            'physical_exercise': exe_display,
            'hypertension_determination': 'yes' if is_hypertensive else ('missing' if (pd.isna(hist_hyp) and pd.isna(sbp) and pd.isna(dbp)) else 'no'),
            # --- Model computation results (missing shown as "-") ---
            'predicted_score': score if score is not None else "-",
            'risk_level': risk_level,
            'prediction_horizon': 'current',
            'code': code,
            'gauge_reference': gauge_reference_str if score is not None else "-",
            'recommendation': suggestion
        })

    if results:
        pd.DataFrame(results).to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"--- Assessment complete: {len(results)} records processed. Results saved to: {output_path} ---")
    else:
        print("--- Assessment complete, no valid results ---")

if __name__ == "__main__":
    process_data("base.csv", "item_result.csv", "ADA_diabetes_screening_result_E14.csv")
