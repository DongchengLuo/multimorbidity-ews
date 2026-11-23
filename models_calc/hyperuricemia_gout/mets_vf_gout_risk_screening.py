import pandas as pd
import numpy as np
import re
import math

# ==========================================
# 1. Field mapping (exact match)
# ==========================================
FIELD_MAPPING = {
    'age': ['app_pat_age', 'age'],
    'gender': ['pat_sex', 'sex', 'gender'],
    'height': ['height', 'body_length'],
    'weight': ['weight'],
    'wc': ['waist', 'abdominal_circumference', 'wc'],
    'FBG': ['GLU', 'fasting_glucose', 'glucose', 'glu', 'fpg', 'fbg'],
    'TG': ['triglyceride', 'TG', 'tg', 'triglycerides'],
    'HDL-C': ['HDL_cholesterol', 'HDL', 'HDL-C', 'hdl', 'hdl-c',
              'high_density_lipoprotein']
}


# ==========================================
# 2. Basic utility functions
# ==========================================
def normalize_str(s):
    if pd.isna(s): return ""
    return re.sub(r'[\(\)\（\）\[\]\【\】\{\}\s\-_]', '', str(s)).lower()


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


def parse_gender(val):
    s = str(val).strip()
    if pd.isna(val) or s == "": return np.nan
    if any(k in s for k in ['male', 'M', '1', 'Male']): return 1
    if any(k in s for k in ['female', 'F', '0', '2', 'Female']): return 0
    return np.nan


# ==========================================
# 3. METS-VF core scoring logic (based on the 2025 literature formula)
# ==========================================
def calculate_mets_vf_score(age, sex, height, weight, wc, fbg, tg, hdl_c, units_are_mmol=True):
    required_vars = {'age': age, 'gender': sex, 'height': height, 'weight': weight, 'wc': wc, 'FBG': fbg, 'TG': tg,
                     'HDL-C': hdl_c}

    # 1. Completeness check: return the placeholder "-" without computing when items are missing
    for k, v in required_vars.items():
        if pd.isna(v) or (isinstance(v, (int, float)) and v <= 0):
            # Sex 0 (female) is valid
            if k == 'gender' and v == 0:
                pass
            else:
                return None, "-", "-", "-", {'BMI': '-', 'WHtR': '-', 'METS_IR': '-'}

    # Unit conversion (mmol/L -> mg/dL)
    fbg_mg = fbg * 18.018 if units_are_mmol else fbg
    tg_mg = tg * 88.57 if units_are_mmol else tg
    hdl_mg = hdl_c * 38.67 if units_are_mmol else hdl_c

    height_cm = height if height > 3 else height * 100.0
    bmi = weight / ((height_cm / 100.0) ** 2)

    try:
        # METS-IR computation
        mets_ir = (math.log(2 * fbg_mg + tg_mg) * bmi) / math.log(hdl_mg)

        # METS-VF computation
        whtr = wc / height_cm
        term_ir = 0.011 * (math.log(mets_ir) ** 3)
        term_whtr = 3.239 * (math.log(whtr) ** 3)
        term_sex = 0.319 * sex
        term_age = 0.594 * math.log(age)
        score = 4.466 + term_ir + term_whtr + term_sex + term_age

        # Risk-tier determination
        if score < 6.71:
            level, code = "low risk", "M10.1.1"
        elif score <= 7.67:
            level, code = "medium risk", "M10.1.2"
        else:
            level, code = "high risk", "M10.1.3"

        debug_info = {'BMI': round(bmi, 2), 'WHtR': round(whtr, 3), 'METS_IR': round(mets_ir, 2)}
        return score, mets_ir, level, code, debug_info

    except (ValueError, ZeroDivisionError):
        return None, "-", "-", "-", {'BMI': '-', 'WHtR': '-', 'METS_IR': '-'}


# ==========================================
# 4. Main data-processing routine
# ==========================================
def process_data(base_path, item_path, output_path):
    print("--- Starting METS-VF (2025) risk assessment ---")

    def read_csv_safe(path):
        for enc in ['gbk', 'utf-8-sig', 'utf-8']:
            try:
                return pd.read_csv(path, encoding=enc, low_memory=False, dtype=str)
            except:
                continue
        return pd.DataFrame()

    base_df = read_csv_safe(base_path)
    item_df = read_csv_safe(item_path)

    if base_df.empty:
        print("Error: base data file could not be read or is empty")
        return

    records = []
    if not item_df.empty:
        item_df['cat'] = item_df['res_itm_name'].apply(get_category)
        filtered = item_df.dropna(subset=['cat', 'res_value']).copy()

        grouped = filtered.groupby(['app_no', 'pat_id'])

        for (app_no, pat_id), group in grouped:
            # Compatibility: strip redundant leading '0' from the primary key
            safe_app_no = str(app_no).strip().lstrip('0') or '0'
            safe_pat_id = str(pat_id).strip().lstrip('0') or '0'

            row = {'app_no': safe_app_no, 'pat_id': safe_pat_id}
            data_pool = {k: np.nan for k in FIELD_MAPPING.keys()}

            # Keep only the first valid value matched that day (no averaging)
            for _, item in group.iterrows():
                if item['cat'] and pd.isna(data_pool[item['cat']]):
                    val = clean_numeric(item['res_value'])
                    if not pd.isna(val): data_pool[item['cat']] = val

            for key in FIELD_MAPPING.keys():
                row[key] = data_pool[key]
            records.append(row)

    wide_items = pd.DataFrame(records) if records else pd.DataFrame(
        columns=['app_no', 'pat_id'] + list(FIELD_MAPPING.keys()))

    base_df.rename(columns={'app_no': 'app_no'}, inplace=True, errors='ignore')

    # Format the primary key so both sides match
    base_df['app_no'] = base_df['app_no'].astype(str).str.strip().str.lstrip('0').replace('', '0')
    base_df['pat_id'] = base_df['pat_id'].astype(str).str.strip().str.lstrip('0').replace('', '0')

    # Use a left join so nobody in the base table is lost
    final = pd.merge(base_df, wide_items, on=['app_no', 'pat_id'], how='left')
    results = []

    GAUGE_REF = "Q1:<6.71|Q2:6.71-7.08|Q3:7.09-7.67|Q4:>7.67"

    for _, r in final.iterrows():
        # Prefer indicator extraction; fall back to basic information
        age = r['age'] if 'age' in r and pd.notna(r['age']) else clean_numeric(r.get('app_pat_age'))
        gender_raw = r['gender'] if 'gender' in r and pd.notna(r['gender']) else r.get('pat_sex')
        gender_val = parse_gender(gender_raw)

        height = r.get('height')
        weight = r.get('weight')
        wc = r.get('wc')
        fbg = r.get('FBG')
        tg = r.get('TG')
        hdl_c = r.get('HDL-C')

        score, mets_ir, level, code, details = calculate_mets_vf_score(
            age, gender_val, height, weight, wc, fbg, tg, hdl_c
        )

        # Assemble output results
        results.append({
            'app_no': r.get('app_no'),
            'pat_id': r.get('pat_id'),
            'pat_name': r.get('pat_name', 'unknown'),

            # --- Raw feature inputs: missing shown as "missing" ---
            'age': int(age) if pd.notna(age) else "missing",
            'sex': ('male' if gender_val == 1 else 'female') if pd.notna(gender_val) else "missing",
            'height': round(height, 1) if pd.notna(height) else "missing",
            'weight': round(weight, 1) if pd.notna(weight) else "missing",
            'waist': round(wc, 1) if pd.notna(wc) else "missing",
            'fasting_glucose': round(fbg, 2) if pd.notna(fbg) else "missing",
            'triglyceride': round(tg, 2) if pd.notna(tg) else "missing",
            'HDL': round(hdl_c, 2) if pd.notna(hdl_c) else "missing",

            # --- Intermediate derived values and model output: missing shown as "-" ---
            'BMI': details.get('BMI', "-"),
            'waist_height_ratio': details.get('WHtR', "-"),
            'METS-IR_index': details.get('METS_IR', "-"),
            'code': code if score is not None else "-",
            'gauge_reference': GAUGE_REF,
            'predicted_score': round(score, 4) if score is not None else "-",
            'risk_level': level if score is not None else "-",
            'model_name': 'gout_risk_METS-VF'
        })

    if results:
        res_df = pd.DataFrame(results)
        res_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"--- Done! {len(res_df)} results saved to {output_path} ---")


if __name__ == "__main__":
    process_data("base.csv", "item_result.csv", "mets_vf_gout_risk_result.csv")
