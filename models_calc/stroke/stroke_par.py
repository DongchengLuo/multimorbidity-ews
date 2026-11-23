import pandas as pd
import numpy as np
import re
import math

# ==========================================
# 1. Field mapping (for the China-PAR stroke risk model)
# ==========================================
FIELD_MAPPING = {
    'sbp': ['systolic_bp', 'blood_pressure', 'sbp', 'systolic_pressure'],
    'TC': ['total_cholesterol', 'CHOL', 'TC', 'chol'],
    'HDL-C': ['HDL', 'HDL_cholesterol', 'HDL-C'],
    'wc': ['waist', 'wc'],
    'smoke': ['smoking_history', 'smoking_status', 'smoker', 'smoke'],
    'has_diabetes': ['diabetes_history', 'prior_diabetes', 'type2_diabetes', 'diabetes'],
    'ASCVD_family_history': ['ascvd_family_history', 'stroke_family_history', 'family_history'],
    'residential_area': ['residence', 'living_environment', 'residential_area'],
    'region': ['geographic_region', 'region']
}

# Model parameter constants (China-PAR)
PARAMS = {
    'male': {'s0_10': 0.963, 'mean_ln_age': 3.86, 'mean_ln_sbp': 4.81, 'mean_ln_tc': 5.21, 'mean_ln_hdl': 3.89},
    'female': {'s0_10': 0.982, 'mean_ln_age': 3.85, 'mean_ln_sbp': 4.80, 'mean_ln_tc': 5.22, 'mean_ln_hdl': 3.95,
               'mean_ln_wc': 4.33}
}


# ==========================================
# 2. Basic helpers and unit-alignment functions
# ==========================================
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


def clean_numeric(val):
    if pd.isna(val) or str(val).strip() == "":
        return np.nan
    try:
        if '/' in str(val):
            return max([float(x.strip()) for x in str(val).split('/') if x.strip()])
        match = re.search(r"[-+]?\d*\.\d+|\d+", str(val))
        return float(match.group()) if match else np.nan
    except:
        return np.nan


def check_positive_keyword(text_list, keywords):
    combined_text = " ".join([str(t) for t in text_list if not pd.isna(t)])
    combined_text = normalize_str(combined_text)
    positive_flags = ['yes', 'present', '1', 'true', 'positive'] + keywords
    for k in positive_flags:
        if k in combined_text:
            return 1
    return 0


def align_lipid_unit(val, unit_str):
    """
    Dual lipid unit alignment (force mg/dL):
    1. Prefer reading and judging the res_unit column.
    2. Without a unit or with an unrecognized unit, fall back to magnitude (<20 is treated as mmol/L).
    Factor: 1 mmol/L = 38.66 mg/dL
    """
    if pd.isna(val):
        return val

    # Strategy 1: explicit unit recognition
    unit_str = str(unit_str).strip().lower()
    if 'mmol' in unit_str:
        return round(val * 38.66, 2)
    elif 'mg/dl' in unit_str or 'mg' in unit_str:
        return round(val, 2)

    # Strategy 2: numeric-threshold fallback (mmol/L is usually < 10 and never above 20)
    if val < 20:
        return round(val * 38.66, 2)

    return round(val, 2)


# ==========================================
# 3. China-PAR core logic
# ==========================================
def calculate_china_par(row):
    """Returns (risk probability, risk tier, risk code, computation-details string)"""
    age = clean_numeric(row.get('app_pat_age'))
    gender_str = str(row.get('pat_sex', ''))
    is_male = any(x in gender_str for x in ['male', '1'])

    sbp = row.get('sbp')
    tc = row.get('TC')
    hdl = row.get('HDL-C')
    smoke = row.get('flag_smoke', 0)
    diabetes = row.get('flag_diabetes', 0)
    family_history = row.get('flag_family', 0)

    urban = 1 if not row.get('residential_area', '').strip() or check_positive_keyword(
        [row.get('residential_area', '')], ['urban']) else 0
    north = 1 if check_positive_keyword([row.get('region', '')], ['north']) else 0

    required_missing = []
    if pd.isna(age): required_missing.append('age')
    if not gender_str: required_missing.append('sex')
    if pd.isna(sbp): required_missing.append('systolic_bp')
    if pd.isna(tc): required_missing.append('total_cholesterol')
    if pd.isna(hdl): required_missing.append('HDL')
    if not is_male:
        wc = row.get('wc')
        if pd.isna(wc):
            required_missing.append('waist (required for women)')

    if required_missing:
        return None, "-", "", f"missing required indicators: {','.join(required_missing)}"

    try:
        if is_male:
            p = PARAMS['male']
            score = 0.587 * (math.log(age) - p['mean_ln_age']) + \
                    1.156 * (math.log(sbp) - p['mean_ln_sbp']) + \
                    0.354 * (math.log(tc) - p['mean_ln_tc']) - \
                    0.428 * (math.log(hdl) - p['mean_ln_hdl']) + \
                    0.449 * smoke + 0.542 * diabetes + \
                    0.18 * north + 0.12 * urban + 0.25 * family_history
            s0 = p['s0_10']
        else:
            p = PARAMS['female']
            wc = row.get('wc')
            score = 0.612 * (math.log(age) - p['mean_ln_age']) + \
                    1.230 * (math.log(sbp) - p['mean_ln_sbp']) + \
                    0.410 * (math.log(tc) - p['mean_ln_tc']) - \
                    0.480 * (math.log(hdl) - p['mean_ln_hdl']) + \
                    0.460 * smoke + 0.580 * diabetes + \
                    0.15 * north + 0.10 * urban + \
                    0.35 * (math.log(wc) - p['mean_ln_wc'])
            s0 = p['s0_10']

        prob = (1 - (s0 ** math.exp(score))) * 100

        if prob < 5.0:
            level = "low risk"
            code = "I64.3.1"
        elif 5.0 <= prob < 10.0:
            level = "medium risk"
            code = "I64.3.2"
        else:
            level = "high risk"
            code = "I64.3.3"

        details = f"score computed; smoking:{smoke}, diabetes:{diabetes}, family_history:{family_history}, north:{north}, urban:{urban}"
        return round(prob, 2), level, code, details

    except Exception as e:
        return None, "computation_error", "", str(e)


# ==========================================
# 4. Main data-processing routine
# ==========================================
def process_data(base_path, item_path, output_path):
    print("--- Starting [China-PAR 10-year stroke risk] model ---")

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

    item_df['cat'] = item_df['res_itm_name'].apply(get_category)
    filtered = item_df.dropna(subset=['cat', 'res_value']).copy()

    records = []
    print("Analyzing indicator data...")

    for (app_no, pat_id), group in filtered.groupby(['app_no', 'pat_id']):
        row = {'app_no': str(app_no).strip(), 'pat_id': str(pat_id).strip()}

        # Handle the unit column; use an empty string when res_unit is absent
        has_unit_col = 'res_unit' in group.columns

        data_pool = {k: [] for k in FIELD_MAPPING.keys()}

        for _, item in group.iterrows():
            u = item['res_unit'] if has_unit_col and not pd.isna(item['res_unit']) else ""
            data_pool[item['cat']].append({'val': item['res_value'], 'unit': u})

        # Extract SBP
        sbp_items = data_pool.get('sbp', [])
        row['sbp'] = clean_numeric(sbp_items[0]['val']) if sbp_items else np.nan

        # Extract and auto-convert TC
        tc_items = data_pool.get('TC', [])
        if tc_items:
            raw_tc = clean_numeric(tc_items[0]['val'])
            row['TC'] = align_lipid_unit(raw_tc, tc_items[0]['unit'])
        else:
            row['TC'] = np.nan

        # Extract and auto-convert HDL-C
        hdl_items = data_pool.get('HDL-C', [])
        if hdl_items:
            raw_hdl = clean_numeric(hdl_items[0]['val'])
            row['HDL-C'] = align_lipid_unit(raw_hdl, hdl_items[0]['unit'])
        else:
            row['HDL-C'] = np.nan

        # Other indicator extraction
        wc_items = data_pool.get('wc', [])
        row['wc'] = clean_numeric(wc_items[0]['val']) if wc_items else np.nan

        row['flag_smoke'] = check_positive_keyword([x['val'] for x in data_pool.get('smoke', [])], [])
        row['flag_diabetes'] = check_positive_keyword([x['val'] for x in data_pool.get('has_diabetes', [])], [])
        row['flag_family'] = check_positive_keyword([x['val'] for x in data_pool.get('ASCVD_family_history', [])], [])

        row['residential_area'] = " ".join([str(x['val']) for x in data_pool.get('residential_area', [])])
        row['region'] = " ".join([str(x['val']) for x in data_pool.get('region', [])])

        records.append(row)

    wide_items = pd.DataFrame(records)
    base_df['app_no'] = base_df['app_no'].astype(str).str.strip()
    base_df['pat_id'] = base_df['pat_id'].astype(str).str.strip()
    final = pd.merge(wide_items, base_df, on=['app_no', 'pat_id'], how='inner')

    results = []
    print("Generating final assessment results...")

    for _, r in final.iterrows():
        prob, risk_level, code, details = calculate_china_par(r)

        action = "maintain a healthy lifestyle; regular check-ups"
        if risk_level == "high risk":
            action = "intensify prevention under medical guidance; control BP and lipids"
        elif risk_level == "medium risk":
            action = "intensify intervention on underlying disease; monitor BP and metabolic markers"
        elif risk_level == "-":
            action = "please provide complete exam indicators for an accurate assessment"

        results.append({
            'model_name': 'China-PAR 10-Year Stroke Risk Model',
            'pat_id': r['pat_id'],
            'app_no': r['app_no'],
            'pat_name': r.get('pat_name', 'unknown'),
            'sex': r.get('pat_sex', ''),
            'age': r.get('app_pat_age', '-'),
            'risk_probability': f"{prob:.2f}%" if prob is not None else "-",
            'risk_level': risk_level,
            'code': code,
            'gauge_reference': '0%-5%|5%-10%|10%-100%',
            'recommendation': action,
            'prediction_horizon': 'next 10 years',
            'systolic_bp (mmHg)': r['sbp'],
            'total_cholesterol (mg/dL)': r['TC'],  # already converted to mg/dL here
            'HDL-C(mg/dL)': r['HDL-C'],  # already converted to mg/dL here
            'waist (cm)': r['wc'],
            'smoking_history': "yes" if r['flag_smoke'] else "no",
            'diabetes_history': "yes" if r['flag_diabetes'] else "no",
            'family_history': "yes" if r['flag_family'] else "no",
            'residence': r['residential_area'] if r['residential_area'] else "not provided (default urban)",
            'geographic_region': r['region'] if r['region'] else "not provided (default non-northern)",
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
        "china_par_stroke_result.csv"
    )