import pandas as pd
import numpy as np
import re
import warnings

# Suppress unnecessary dtype warnings
warnings.filterwarnings('ignore', category=pd.errors.DtypeWarning)

# ==========================================
# 1. Field mapping
# ==========================================
FIELD_MAPPING = {
    'age': ['app_pat_age', 'age'],
    'gender': ['pat_sex', 'sex'],
    'height': ['height'],
    'weight': ['weight'], 
    'scr': ['creatinine (cr)', 'serum_creatinine', 'scr', 'creatinine'], 
    'FBG': ['glucose (glu)', 'fasting_glucose', 'glucose', 'fbg'], 
    'ALT': ['ALT', 'alt'],
    'ALP': ['ALP', 'alp', 'alkaline phosphatase'],
    'acute_pancreatitis': ['acute_pancreatitis', 'pancreatitis_history', 'acute pancreatitis'],
    'insulin': ['insulin', 'fasting_serum_insulin', 'serum_insulin'],
    'aspirin': ['aspirin'],
    'gastroprotective': ['stomach_medication', 'gastroprotectant', 'ppi', 'proton_pump_inhibitor', 'h2ra', 'omeprazole', 'ranitidine'],
    'delta_bmi': ['bmi_change', 'delta_bmi'],
    'bmi_old': ['bmi_one_year_ago', 'bmi_1yr_ago', 'bmi_last_year']
}

# ==========================================
# 2. Basic cleaning, matching logic and formulas
# ==========================================
def normalize_str(s):
    if pd.isna(s): return ""
    return re.sub(r'[\(\)\（\）\[\]\【\】\{\}\s]', '', str(s)).lower()

def get_category(name):
    norm_name = normalize_str(name)
    for key, patterns in FIELD_MAPPING.items():
        for p in patterns:
            # Exact match
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

def parse_gender_to_female_flag(val):
    s = str(val).strip()
    if pd.isna(val) or s == "": return np.nan
    if s in ['female', 'Female', 'F', '1', 'Woman']: return 1
    if s in ['male', 'Male', 'M', '0', 'Man']: return 0
    return np.nan

def check_keyword_exist(val_list, keyword_list):
    if not val_list: return 0
    combined_text = " ".join([str(v) for v in val_list]).lower()
    for kw in keyword_list:
        if kw.lower() in combined_text:
            return 1
    return 0

def calculate_bmi(weight, height):
    if pd.isna(weight) or pd.isna(height) or height <= 0:
        return np.nan
    h_m = height / 100.0 if height > 3 else height
    return weight / (h_m * h_m)

def calculate_egfr_2021(scr, age, sex_female):
    if any(pd.isna(x) for x in [scr, age, sex_female]) or scr <= 0:
        return np.nan
    scr_mg_dL = scr if scr < 20 else scr / 88.4
    kappa = 0.7 if sex_female == 1 else 0.9
    alpha = -0.241 if sex_female == 1 else -0.302
    gender_mult = 1.012 if sex_female == 1 else 1.0
    scr_k = scr_mg_dL / kappa
    egfr = 142 * (min(scr_k, 1)**alpha) * (max(scr_k, 1)**-1.200) * (0.9938**age) * gender_mult
    return egfr

# ==========================================
# 3. Pancreatic cancer risk scoring core logic (NODM)
# ==========================================
def calculate_pancreatic_cancer_risk(age, sex_female, acute_panc, insulin, aspirin, gastro_drug,
                                     glu, bmi, delta_bmi, egfr, alp, alt):
    debug_info = {
        'age': age, 'is_female': sex_female, 'acute_panc': acute_panc,
        'insulin': insulin, 'aspirin': aspirin, 'gastro_drug': gastro_drug,
        'glucose': glu, 'bmi': bmi, 'delta_bmi': delta_bmi,
        'egfr': egfr, 'alp': alp, 'alt': alt
    }

    # 1-year
    required_1yr = [age, insulin, aspirin, gastro_drug, glu, delta_bmi, egfr, alp, alt]
    if any(pd.isna(x) for x in required_1yr):
        risk_1yr = np.nan
    else:
        x_1yr = (
                0.079446 * age + 1.069337 * insulin - 1.098602 * aspirin + 1.222557 * gastro_drug +
                0.112956 * glu - 0.206535 * delta_bmi + 0.037461 * egfr +
                0.005154 * alp + 0.001996 * alt - 17.429252
        )
        risk_1yr = 1 / (1 + np.exp(-x_1yr))

    # 3-year
    required_3yr = [age, sex_female, acute_panc, glu, bmi, delta_bmi, egfr, alp, alt]
    if any(pd.isna(x) for x in required_3yr):
        risk_3yr = np.nan
    else:
        x_3yr = (
                0.069605 * age - 0.469378 * sex_female + 1.256417 * acute_panc + 0.084842 * glu -
                0.080945 * bmi - 0.148898 * delta_bmi + 0.023994 * egfr +
                0.004456 * alp + 0.001619 * alt - 11.735245
        )
        risk_3yr = 1 / (1 + np.exp(-x_3yr))

    return risk_1yr, risk_3yr, debug_info

def determine_risk_level_and_code(p1, p3):
    if pd.isna(p1) and pd.isna(p3):
        return "-", "-", "data missing; cannot determine"

    p1_val = p1 if not pd.isna(p1) else 0
    p3_val = p3 if not pd.isna(p3) else 0

    if p1_val > 0.0168 or p3_val > 0.0174:
        return "high risk", "C25.4.3", "high-risk criteria: 1-year probability >1.68% or 3-year probability >1.74%"
    if p1_val > 0.0045 or p3_val > 0.0074:
        return "medium risk", "C25.4.2", "medium-risk criteria: 1-year probability >0.45% or 3-year probability >0.74%"
    return "low risk", "C25.4.1", "low risk: below medium/high thresholds"

# ==========================================
# 4. Main data-processing routine
# ==========================================
def process_data(base_path, item_path, output_path):
    print("--- Starting early pancreatic cancer risk model (NODM) ---")

    def read_csv_safe(path):
        for enc in ['gbk', 'utf-8-sig', 'utf-8']:
            try: return pd.read_csv(path, encoding=enc, low_memory=False)
            except: continue
        return pd.DataFrame()

    base_df = read_csv_safe(base_path)
    item_df = read_csv_safe(item_path)
    if base_df.empty:
        print("Base data file could not be read or is empty")
        return

    # [Important fix]: strip redundant leading zeros that break table joins
    base_df['app_no'] = base_df['app_no'].astype(str).str.strip().str.lstrip('0')
    base_df['pat_id'] = base_df['pat_id'].astype(str).str.strip().str.lstrip('0')

    # 1. Pivot to wide table
    records = []
    if not item_df.empty:
        item_df['app_no'] = item_df['app_no'].astype(str).str.strip().str.lstrip('0')
        item_df['pat_id'] = item_df['pat_id'].astype(str).str.strip().str.lstrip('0')
        
        item_df['cat'] = item_df['res_itm_name'].apply(get_category)
        filtered = item_df.dropna(subset=['cat', 'res_value']).copy()

        for (app_no, pat_id), group in filtered.groupby(['app_no', 'pat_id']):
            row = {'app_no': app_no, 'pat_id': pat_id}
            data_pool = {k: [] for k in FIELD_MAPPING.keys()}

            for _, item in group.iterrows():
                data_pool[item['cat']].append(str(item['res_value']))

            # [Fix]: drop mean computation and take the first valid number
            for num_field in ['FBG', 'ALP', 'ALT', 'bmi_old', 'delta_bmi', 'height', 'weight', 'scr']:
                vals = [clean_numeric(v) for v in data_pool.get(num_field, [])]
                vals = [v for v in vals if pd.notna(v)]
                row[num_field] = vals[0] if vals else np.nan

            row['insulin'] = check_keyword_exist(data_pool.get('insulin', []), FIELD_MAPPING['insulin'])
            row['aspirin'] = check_keyword_exist(data_pool.get('aspirin', []), FIELD_MAPPING['aspirin'])
            row['gastroprotective'] = check_keyword_exist(data_pool.get('gastroprotective', []), FIELD_MAPPING['gastroprotective'])
            row['acute_pancreatitis'] = check_keyword_exist(data_pool.get('acute_pancreatitis', []), FIELD_MAPPING['acute_pancreatitis'])

            records.append(row)

    wide_items = pd.DataFrame(records) if records else pd.DataFrame(columns=['app_no', 'pat_id'])

    # 2. Merge basic information (left join so people are not dropped for missing items)
    final = pd.merge(base_df, wide_items, on=['app_no', 'pat_id'], how='left')

    # 3. Scoring and result generation
    results = []
    for _, r in final.iterrows():
        age_map = clean_numeric(r.get('age'))
        age_base = clean_numeric(r.get('app_pat_age'))
        age = age_map if pd.notna(age_map) else age_base

        sex_raw = r.get('gender')
        if pd.isna(sex_raw): sex_raw = r.get('pat_sex')
        sex_female = parse_gender_to_female_flag(sex_raw)

        bmi = calculate_bmi(r.get('weight'), r.get('height'))
        egfr = calculate_egfr_2021(r.get('scr'), age, sex_female)
        
        delta_bmi = r.get('delta_bmi')
        # When delta_bmi is absent, derive it as current bmi minus bmi_old from a year earlier
        if pd.isna(delta_bmi) and pd.notna(bmi) and pd.notna(r.get('bmi_old')):
            delta_bmi = bmi - r.get('bmi_old')

        # Perform score estimation
        risk_1yr, risk_3yr, debug = calculate_pancreatic_cancer_risk(
            age=age, sex_female=sex_female, acute_panc=r.get('acute_pancreatitis'),
            insulin=r.get('insulin'), aspirin=r.get('aspirin'), gastro_drug=r.get('gastroprotective'),
            glu=r.get('FBG'), bmi=bmi, delta_bmi=delta_bmi, egfr=egfr, alp=r.get('ALP'), alt=r.get('ALT')
        )

        risk_level_str, risk_code, standard_desc = determine_risk_level_and_code(risk_1yr, risk_3yr)

        # Format for presentation
        def fmt_in(v, is_num=False):
            if pd.isna(v) or v is None or v == "": return "missing"
            return f"{v:.2f}" if is_num else str(v)

        def fmt_bool(v):
            if pd.isna(v): return "missing"
            return "yes" if v == 1 else "no"

        results.append({
            'model_name': 'NODM Pancreatic Cancer Risk Prediction Model',
            'serial_number': r.get('app_no'),
            'ID': r.get('pat_id'),
            'name': r.get('pat_name', 'unknown'),
            'sex': "male" if sex_female == 0 else ("female" if sex_female == 1 else "missing"),
            'age': int(age) if pd.notna(age) else "missing",
            # --- Result display (use "-" when not computable) ---
            'risk_probability': f"{risk_1yr * 100:.2f}%" if pd.notna(risk_1yr) else "-",
            '3-year_cancer_probability (%)': f"{risk_3yr * 100:.2f}%" if pd.notna(risk_3yr) else "-",
            'risk_level': risk_level_str,
            'code': risk_code,
            'risk_criteria': standard_desc,
            # --- Input indicator check values (missing shown as "missing") ---
            'FBG': fmt_in(debug['glucose'], True),
            'eGFR': fmt_in(debug['egfr'], True),      
            'ALP': fmt_in(debug['alp'], True),
            'ALT': fmt_in(debug['alt'], True),
            'BMI': fmt_in(debug['bmi'], True),        
            'bmi_change': fmt_in(debug['delta_bmi'], True),
            'acute_pancreatitis_history': fmt_bool(debug['acute_panc']),
            'on_insulin': fmt_bool(debug['insulin']),
            'on_aspirin': fmt_bool(debug['aspirin']),
            'on_gastroprotectant': fmt_bool(debug['gastro_drug']),
        })

    if results:
        pd.DataFrame(results).to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"--- Assessment complete: {len(results)} results, exported to {output_path} ---")
    else:
        print("--- internal error: no valid results! ---")

if __name__ == "__main__":
    process_data(
        "base.csv",
        "item_result.csv",
        "nodm_pancreatic_cancer_risk_result.csv"
    )
