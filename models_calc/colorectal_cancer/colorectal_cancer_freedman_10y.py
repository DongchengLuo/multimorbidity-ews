import pandas as pd
import numpy as np
import re
import os

# ==========================================
# 1. Field mapping (exact-match logic)
# ==========================================
FIELD_MAPPING = {
    'age': ['app_pat_age', 'age'],
    'gender': ['pat_sex', 'sex'],
    'fdr_count': ['first_degree_relatives_colorectal_cancer', 'first_degree_cancer_count', 'colorectal_cancer_family_history', 'fdr_count', 'family_history'],
    'endoscopy_10yr': ['endoscopy', 'colonoscopy_history', 'sigmoidoscopy', 'colonoscopy', 'colonoscopy_within_10y'],
    'polyp_history': ['intestinal_polyp', 'polyp_history', 'colon_polyp'],
    'veg_servings': ['vegetable_intake', 'daily_vegetable_servings', 'vegetable_servings', 'veg_g'],
    'aspirin_nsaid': ['aspirin', 'nsaid', 'NSAID', 'salicylate_use'],
    'bmi': ['(body mass index) BMI', 'bmi', 'BMI']
}

# Risk-grading configuration
RISK_THRESHOLDS = {'LOW_CUT': 3.0, 'HIGH_CUT': 6.0}
RISK_CODES = {'low': 'C18.2.1', 'medium': 'C18.2.2', 'high': 'C18.2.3'}
GAUGE_REF = f"0-{int(RISK_THRESHOLDS['LOW_CUT'])}|{int(RISK_THRESHOLDS['LOW_CUT'])}-{int(RISK_THRESHOLDS['HIGH_CUT'])}|{int(RISK_THRESHOLDS['HIGH_CUT'])}-100"

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
            # Change: switch to exact match
            if normalize_str(p) == norm_name: return key
    return None

def clean_numeric(val):
    if pd.isna(val) or str(val).strip() == "": return np.nan
    try:
        match = re.search(r"[-+]?\d*\.\d+|\d+", str(val))
        return float(match.group()) if match else np.nan
    except:
        return np.nan

def parse_sex(val):
    s = normalize_str(val)
    if 'female' in s or 'f' in s or s == '0' or '2' in s:
        return 0 # female
    if 'male' in s or 'm' in s or s == '1':
        return 1 # male
    return np.nan

def parse_bool_indicator(val_list):
    """Generic boolean determination"""
    if not val_list: return np.nan
    combined_str = " ".join([str(v) for v in val_list]).lower()
    if any(x in combined_str for x in ['yes', 'present', 'true', 'positive', 'pos', '+']): return 1
    if any(x in combined_str for x in ['no', 'absent', 'false', 'negative', 'neg', '-']): return 0
    return np.nan

# ==========================================
# 3. Core computation logic (simplified Freedman model)
# ==========================================
def calculate_freedman_risk(age, sex_val, fdr, has_endo, has_polyp, has_aspirin, veg_servings):
    """
    Compute the relative risk factor (RR) and final risk percentage
    Core: age, sex and endoscopy history are required
    """
    if any(pd.isna(x) for x in [age, sex_val, has_endo]):
        return None, "-", "-", "-"

    is_male = (sex_val == 1)
    rr = {'p': 1.0, 'd': 1.0, 'r': 1.0}
    
    # 1. Family-history factor (defaults to 0)
    fdr_n = fdr if pd.notna(fdr) else 0
    if is_male:
        f_p = 1.81 if fdr_n == 1 else (3.28 if fdr_n >= 2 else 1.0)
        f_d = 1.68 if fdr_n == 1 else (2.81 if fdr_n >= 2 else 1.0)
        f_r = 1.49 if fdr_n >= 1 else 1.0
    else:
        f_p = 1.51 if fdr_n == 1 else (2.27 if fdr_n >= 2 else 1.0)
        f_d = 1.45 if fdr_n == 1 else (2.09 if fdr_n >= 2 else 1.0)
        f_r = 1.53 if fdr_n >= 1 else 1.0
    rr['p'] *= f_p; rr['d'] *= f_d; rr['r'] *= f_r

    # 2. Endoscopy and polyp factors
    if has_endo == 0: 
        rr['p'] *= 1.42 if is_male else 1.82
        rr['d'] *= 2.83 if is_male else 3.44
        rr['r'] *= 3.86 if is_male else 2.99
    elif has_polyp == 1: 
        rr['p'] *= 1.77 if is_male else 2.62
        rr['d'] *= 1.34 if is_male else 4.35
        rr['r'] *= 1.92 if is_male else 3.19

    # 3. Protective factors (aspirin, vegetables)
    if has_aspirin == 1:
        rr['p'] *= 0.65 if is_male else 0.63
        rr['d'] *= 0.71 if is_male else 0.70
        rr['r'] *= 0.66 if is_male else 0.70
    
    if pd.notna(veg_servings) and veg_servings >= 5:
        rr['p'] *= 0.58 if is_male else 0.72

    # Compute simulated absolute risk
    base_risk = 1.2 * (age / 50)**2 
    risk_pct = base_risk * rr['p'] 

    # Risk grading
    if risk_pct < RISK_THRESHOLDS['LOW_CUT']:
        risk_level, risk_code = 'low risk', RISK_CODES['low']
        suggestion = "maintain a healthy diet; recheck colonoscopy in 5-10 years"
    elif risk_pct < RISK_THRESHOLDS['HIGH_CUT']:
        risk_level, risk_code = 'medium risk', RISK_CODES['medium']
        suggestion = "colonoscopy every 3-5 years; increase vegetable intake"
    else:
        risk_level, risk_code = 'high risk', RISK_CODES['high']
        suggestion = "higher risk; consult a physician soon and repeat colonoscopy within 1-2 years"

    return risk_pct, risk_level, risk_code, suggestion

# ==========================================
# 4. Main data-processing routine
# ==========================================
def process_data(base_path, item_path, output_path):
    print(f"--- Starting Colorectal Cancer (Freedman) risk model ---")

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

    # 1. Pivot indicators to wide table
    records = []
    if not item_df.empty:
        item_name_col = 'res_itm_name' if 'res_itm_name' in item_df.columns else 'item_name'
        item_val_col = 'res_value' if 'res_value' in item_df.columns else 'result'

        item_df['cat'] = item_df[item_name_col].apply(get_category)
        filtered = item_df.dropna(subset=['cat', item_val_col]).copy()

        for pat_id, group in filtered.groupby('pat_id'):
            row = {'pat_id': str(pat_id).strip()}
            data_pool = {k: [] for k in FIELD_MAPPING.keys()}
            for _, item in group.iterrows():
                data_pool[item['cat']].append(str(item[item_val_col]))

            # Extract and convert
            fdr_vals = [clean_numeric(v) for v in data_pool.get('fdr_count', [])]
            row['fdr_count'] = np.nanmax(fdr_vals) if fdr_vals else np.nan
            veg_vals = [clean_numeric(v) for v in data_pool.get('veg_servings', [])]
            row['veg_servings'] = np.nanmean(veg_vals) if veg_vals else np.nan

            row['has_endo'] = parse_bool_indicator(data_pool.get('endoscopy_10yr', []))
            row['has_polyp'] = parse_bool_indicator(data_pool.get('polyp_history', []))
            row['has_aspirin'] = parse_bool_indicator(data_pool.get('aspirin_nsaid', []))
            
            # Store raw values for display
            row['raw_family_history'] = " | ".join(data_pool.get('fdr_count', []))
            row['raw_endoscopy_history'] = " | ".join(data_pool.get('endoscopy_10yr', []))
            row['raw_polyp_history'] = " | ".join(data_pool.get('polyp_history', []))
            row['raw_vegetable_intake'] = " | ".join(data_pool.get('veg_servings', []))
            records.append(row)

    wide_items = pd.DataFrame(records) if records else pd.DataFrame(columns=['pat_id'])

    # 2. Merge (left join)
    base_df['pat_id'] = base_df['pat_id'].astype(str).str.strip()
    final = pd.merge(base_df, wide_items, on='pat_id', how='left')

    results = []
    for _, r in final.iterrows():
        # Fetch input
        age = clean_numeric(r.get('app_pat_age') or r.get('age'))
        sex_val = parse_sex(r.get('pat_sex') or r.get('sex'))
        
        fdr = r.get('fdr_count')
        has_endo = r.get('has_endo')
        has_polyp = r.get('has_polyp')
        has_aspirin = r.get('has_aspirin')
        veg_servings = r.get('veg_servings')

        # Compute risk
        risk_pct, risk_level, risk_code, suggestion = calculate_freedman_risk(
            age, sex_val, fdr, has_endo, has_polyp, has_aspirin, veg_servings
        )

        # Formatting function
        def fmt_in(v): return v if (pd.notna(v) and v != "") else "missing"
        def fmt_bool(v): return "yes" if v == 1 else ("no" if v == 0 else "missing")

        results.append({
            'app_no': r.get('app_no', r.get('app_no', '-')),
            'pat_id': r['pat_id'],
            'model_name': 'Colorectal Cancer (Freedman)',
            'name': r.get('pat_name', 'unknown'),
            'sex': 'male' if sex_val == 1 else ('female' if sex_val == 0 else "missing"),
            'age': int(age) if pd.notna(age) else "missing",
            # ----- Raw indicator section (missing shown as "missing") -----
            'family_history': fmt_in(r.get('raw_family_history')),
            'endoscopy_history': fmt_in(r.get('raw_endoscopy_history')),
            'polyp_history': fmt_in(r.get('raw_polyp_history')),
            'vegetable_intake': fmt_in(r.get('raw_vegetable_intake')),
            'has_polyp': fmt_bool(has_polyp),
            'on_aspirin': fmt_bool(has_aspirin),
            # ----- Evaluation result section (shows "-" when not computable) -----
            'risk_probability': f"{risk_pct:.2f}%" if risk_pct is not None else "-",
            'prediction_horizon': 'next 10 years' if risk_pct is not None else "-",
            'risk_level': risk_level,
            'code': risk_code,
            'gauge_reference': GAUGE_REF if risk_pct is not None else "-",
            'recommendation': suggestion
        })

    # 3. Export
    if results:
        res_df = pd.DataFrame(results)
        res_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"--- Assessment complete! Results saved to: {output_path}, {len(res_df)} records. ---")
    else:
        print("--- Assessment complete, no valid results ---")

if __name__ == "__main__":
    process_data("base.csv", "item_result.csv", "crc_freedman_risk_result.csv")
