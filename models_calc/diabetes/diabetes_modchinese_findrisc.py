import pandas as pd
import numpy as np
import re
import math

# ==========================================
# 1. Script metadata and configuration
# ==========================================
MODEL_NAME = "ModChinese Diabetes Risk Screening Model"
REFERENCE = "Zhang J, et al. Adaptation of finnish diabetes risk score for screening (PLoS One. 2025)"
GAUGE_REFERENCE = "0-6|7-7|8-15"

FIELD_MAPPING = {
    'age': ['age', 'app_pat_age'],
    'gender': ['sex', 'pat_sex', 'gender'],
    'bmi': ['BMI', 'bmi', '(body mass index) BMI'],
    'waist': ['waist', 'wc'],
    'has_hypertension': ['hypertension_history', 'has_hypertension', 'past_history'],
    'veg_less_300g': ['insufficient_vegetable_intake', 'daily_vegetables_lt_300g', 'veg_low'],
    'exercise_less_05h': ['physical_inactivity', 'daily_exercise_lt_0_5h', 'active_low'],
    'height': ['height'],
    'weight': ['weight']
}

# ==========================================
# 2. Basic utility functions
# ==========================================
def clean_numeric(val):
    if pd.isna(val) or str(val).strip() == "":
        return np.nan
    try:
        match = re.search(r"[-+]?\d*\.\d+|\d+", str(val))
        return float(match.group()) if match else np.nan
    except:
        return np.nan

def parse_bool(val):
    if pd.isna(val) or str(val).strip() == "":
        return None
    s = str(val).strip()
    if any(k in s for k in ['yes', 'present', '1', 'True', 'true', 'insufficient']):
        return True
    if any(k in s for k in ['no', 'absent', '0', 'False', 'false', 'normal']):
        return False
    return None

def normalize_str(s):
    if pd.isna(s):
        return ""
    return re.sub(r'[\(\)\（\）\[\]\【\】\{\}\s]', '', str(s)).lower()

def get_category(name):
    norm_name = normalize_str(name)
    for key, patterns in FIELD_MAPPING.items():
        for p in patterns:
            if normalize_str(p) == norm_name:
                return key
    return None

def safe_get_first(lst, default=np.nan):
    return lst[0] if (isinstance(lst, list) and len(lst) > 0) else default

# ==========================================
# 3. Core processing flow
# ==========================================
def process_data(base_path="base.csv", item_path="item_result.csv", output_path="FINDRISC_result_E14.csv"):
    print(f"--- Starting {MODEL_NAME} ---")
    
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
        print("Error: cannot read the base data file")
        return

    # 1. Wide-table indicator extraction
    records = []
    if not item_df.empty:
        item_df['cat'] = item_df['res_itm_name'].apply(get_category)
        filtered = item_df.dropna(subset=['cat', 'res_value']).copy()
        
        filtered['app_no'] = filtered['app_no'].astype(str).str.strip()
        filtered['pat_id'] = filtered['pat_id'].astype(str).str.strip()

        for (app_no, pat_id), group in filtered.groupby(['app_no', 'pat_id']):
            pool = {k: group[group['cat']==k]['res_value'].tolist() for k in FIELD_MAPPING.keys()}
            
            res = {
                'app_no': app_no,
                'pat_id': pat_id,
                'bmi_raw': clean_numeric(safe_get_first(pool.get('bmi', []))),
                'waist_raw': clean_numeric(safe_get_first(pool.get('waist', []))),
                'hypertension_raw': parse_bool(safe_get_first(pool.get('has_hypertension', []))),
                'veg_raw': parse_bool(safe_get_first(pool.get('veg_less_300g', []))),
                'exercise_raw': parse_bool(safe_get_first(pool.get('exercise_less_05h', []))),
                'height_raw': clean_numeric(safe_get_first(pool.get('height', []))),
                'weight_raw': clean_numeric(safe_get_first(pool.get('weight', [])))
            }
            
            # Auto-compute BMI when missing
            if pd.isna(res['bmi_raw']) and pd.notna(res['height_raw']) and pd.notna(res['weight_raw']):
                h = res['height_raw']
                w = res['weight_raw']
                if h > 0:
                    h_m = h / 100.0 if h > 3 else h
                    res['bmi_raw'] = w / (h_m * h_m)
            
            records.append(res)
    
    wide_items = pd.DataFrame(records) if records else pd.DataFrame(columns=['app_no', 'pat_id'])
    
    base_df['app_no'] = base_df['app_no'].astype(str).str.strip()
    base_df['pat_id'] = base_df['pat_id'].astype(str).str.strip()
    
    # Left join keeps everyone
    final_merged = pd.merge(base_df, wide_items, left_on=['app_no', 'pat_id'], right_on=['app_no', 'pat_id'], how='left')

    # 2. Row-by-row scoring
    results = []
    for _, r in final_merged.iterrows():
        try:
            # Prepare input parameters
            age = clean_numeric(r.get('app_pat_age'))
            gender_str = str(r.get('pat_sex', ''))
            gender = 'male' if 'male' in gender_str else ('female' if 'female' in gender_str else None)
            
            bmi = r.get('bmi_raw')
            waist = r.get('waist_raw')
            
            # Extract the raw value and check for missingness
            hyp_raw = r.get('hypertension_raw')
            veg_raw = r.get('veg_raw')
            exe_raw = r.get('exercise_raw')
            
            hyp_is_default = pd.isna(hyp_raw)
            veg_is_default = pd.isna(veg_raw)
            exe_is_default = pd.isna(exe_raw)
            
            # Default to False when missing
            hyp = False if hyp_is_default else hyp_raw
            veg = False if veg_is_default else veg_raw
            exe = False if exe_is_default else exe_raw
            
            # Core missing check (only age, gender, bmi, waist; hyp, veg, exe always have values)
            core_missing = any(pd.isna(x) for x in [age, gender, bmi, waist])
            
            if core_missing:
                total_score, risk, code = "-", "-", "-"
            else:
                # Run the scoring logic
                s_age = 0 if age < 35 else (2 if 35 <= age <= 44 else (3 if 45 <= age <= 54 else 4))
                s_bmi = 0 if bmi < 24 else (1 if 24 <= bmi < 28 else 2)
                s_gender = 2 if gender == "male" else 0
                
                if gender == "male":
                    s_waist = 0 if waist < 85 else (2 if 85 <= waist <= 95 else 3)
                else:
                    s_waist = 0 if waist < 80 else (2 if 80 <= waist <= 90 else 3)
                
                s_hyp = 2 if hyp else 0
                s_veg = 1 if veg else 0
                s_exe = 1 if exe else 0
                
                total_score = s_age + s_bmi + s_gender + s_waist + s_hyp + s_veg + s_exe
                
                if total_score >= 8:
                    risk, code = "high risk", "E14.4.3"
                elif total_score == 7:
                    risk, code = "medium risk", "E14.4.2"
                else:
                    risk, code = "low risk", "E14.4.1"
            
            # Formatting function
            def fmt_in(val): return val if (pd.notna(val) and val != "") else "missing"
            def fmt_bool(val): return "present/yes" if val is True else ("absent/no" if val is False else "missing")

            # Wrap results
            results.append({
                'pat_id': r['pat_id'],
                'app_no': r['app_no'],
                'pat_name': r.get('pat_name', 'unknown'),
                # --- Model results (missing shown as "-") ---
                'predicted_score': total_score,
                'risk_level': risk,
                'code': code,
                'gauge_reference': GAUGE_REFERENCE if total_score != "-" else "-",
                'prediction_horizon': 'current' if total_score != "-" else "-",
                'model_name': 'Diabetes Risk Prediction - ModChinese',
                # --- Input features (missing shown as "missing") ---
                'age': int(age) if pd.notna(age) else "missing",
                'sex': fmt_in(gender_str),
                'BMI': round(bmi, 2) if pd.notna(bmi) else "missing",
                'waist': waist if pd.notna(waist) else "missing",
                # Output format including default-value handling
                'hypertension_history': "absent/no (empty past history defaulted to none)" if hyp_is_default else fmt_bool(hyp),
                'insufficient_vegetable_intake': "no/none (defaulted to adequate when not detected)" if veg_is_default else fmt_bool(veg),
                'physical_inactivity': "no/none (defaulted to adequate when not detected)" if exe_is_default else fmt_bool(exe),
            })
        except:
            continue
    
    # 3. Export results
    if results:
        pd.DataFrame(results).to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"Assessment complete: {len(results)} records processed to {output_path}")

if __name__ == "__main__":
    process_data()
