import pandas as pd
import numpy as np
import math
import re

# ==========================================
# 1. Field mapping (strict-match dict)
# ==========================================
FIELD_MAPPING = {
    'sbp': ['systolic_bp', 'blood_pressure'],
    'history_column': ['past_history', 'past_medical_history'],
    'TC': ['total_cholesterol', 'total_cholesterol_chol', 'chol'],
    'HDL_C': ['HDL_cholesterol', 'HDL-C', 'HDL'],
    'wc': ['waist'],
    'smoke': ['smoking_history', 'smoking'],
    'ASCVD_family_history': ['ascvd_family_history', 'family_history'],
    'residential_area': ['residence']
}


# ==========================================
# 2. Data cleaning and parsing logic
# ==========================================
def normalize_str(s):
    if pd.isna(s): return ""
    return re.sub(r'[\(\)\（\）\[\]\【\】\{\}\s]', '', str(s)).lower()


def get_category(name):
    norm_name = normalize_str(name)
    for key, patterns in FIELD_MAPPING.items():
        for p in patterns:
            if norm_name == normalize_str(p):
                return key
    return None


def clean_numeric(val):
    if pd.isna(val) or str(val).strip() == "": return np.nan
    try:
        match = re.search(r"[-+]?\d*\.\d+|\d+", str(val))
        return float(match.group()) if match else np.nan
    except:
        return np.nan


def parse_sbp_from_bp(val):
    if pd.isna(val) or str(val).strip() == "": return np.nan
    s = str(val).strip().replace('／', '/')
    if '/' in s:
        target = s.split('/')[0]
    else:
        target = s
    return clean_numeric(target)


def parse_medical_history(text, keyword, check_untreated=False):
    if pd.isna(text) or str(text).strip() == "": return False
    text = str(text)
    negation_words = ['denies', 'absent', 'not seen', 'no disease', 'without', 'exclude', 'none']

    match = re.search(keyword, text)
    if not match: return False

    start_idx = match.start()
    prefix = text[max(0, start_idx - 6):start_idx]
    if any(nw in prefix for nw in negation_words): return False

    if check_untreated:
        suffix = text[match.end():match.end() + 15]
        # Block descriptions explicitly stating untreated / refused medication
        if re.search(r'untreated|not_medicated|refuses_medication|irregular_medication', suffix):
            return False
        # Capture substantive treatment behavior (e.g. on antihypertensives, intermittent dosing, controlled)
        if re.search(r'taking|medicated|on_treatment|using|controlled', suffix):
            return True
    return True


def parse_smoking(val):
    s = str(val).strip()
    if any(k in s for k in ['yes', 'present', '1', 'smoking', 'daily', 'current']): return True
    return False


# ==========================================
# 3. Core risk computation (China-PAR)
# ==========================================
def calculate_china_par_risk(**kwargs):
    try:
        sex_raw = str(kwargs.get('gender', '')).strip()
        is_male = sex_raw in ['1', '1.0', 'male', 'M', 'm']
        is_female = sex_raw in ['2', '2.0', 'female', 'F', 'f']

        if not (is_male or is_female): return np.nan

        age = float(kwargs.get('age', 0))
        sbp = float(kwargs.get('sbp', 0))
        tc = float(kwargs.get('TC', 0))
        hdl = float(kwargs.get('HDL_C', 0))
        wc = float(kwargs.get('wc', 0))

        if any(pd.isna(v) or v <= 0 for v in [age, sbp, tc, hdl, wc]):
            return None

        # The original China-PAR formula uses mg/dL for TC and HDL-C.
        if tc <= 20: tc = tc * 38.61
        if hdl <= 10: hdl = hdl * 38.61

        ln_age, ln_sbp, ln_tc, ln_hdl, ln_wc = map(math.log, [age, sbp, tc, hdl, wc])

        is_treated = 1 if kwargs.get('is_treated') else 0
        smoker = 1 if kwargs.get('is_smoker') else 0
        diabetes = 1 if kwargs.get('has_diabetes') else 0
        family = 1 if kwargs.get('family_history') else 0

        region_str = str(kwargs.get('region', ''))
        north = 1 if 'north' in region_str else 0
        urban = 0 if any(x in region_str for x in ['farming', 'township', 'village']) else 1

        if is_male:
            s10, mean_xb = 0.9707, 140.68
            sum_bx = (31.97 * ln_age) + (0.62 * ln_tc) + (-0.69 * ln_hdl) + \
                     (-0.71 * ln_wc) + (3.96 * smoker) + (0.36 * diabetes) + \
                     (0.48 * north) + (-0.16 * urban) + (6.22 * family)
            if is_treated:
                sum_bx += (27.39 * ln_sbp) + (-6.02 * ln_age * ln_sbp)
            else:
                sum_bx += (26.15 * ln_sbp) + (-5.73 * ln_age * ln_sbp)
            sum_bx += (-0.94 * ln_age * smoker) + (-1.53 * ln_age * family)
        else:
            s10, mean_xb = 0.9851, 117.26
            sum_bx = (24.87 * ln_age) + (0.06 * ln_tc) + (-0.22 * ln_hdl) + \
                     (1.48 * ln_wc) + (0.49 * smoker) + (0.57 * diabetes) + (0.54 * north)
            if is_treated:
                sum_bx += (20.71 * ln_sbp) + (-4.53 * ln_age * ln_sbp)
            else:
                sum_bx += (19.98 * ln_sbp) + (-4.36 * ln_age * ln_sbp)

        risk = (1 - math.pow(s10, math.exp(sum_bx - mean_xb))) * 100
        return risk
    except:
        return None


# ==========================================
# 4. Risk grading and code mapping
# ==========================================
def get_risk_info(risk):
    if risk is None or np.isnan(risk): return "-", "N/A"
    if risk < 5.0:
        return "low risk", "I25.3.1"
    elif risk < 10.0:
        return "medium risk", "I25.3.2"
    else:
        return "high risk", "I25.3.3"


# ==========================================
# 5. Batch evaluation logic
# ==========================================
def process_data(base_path, item_path, output_path):
    print("--- Starting ASCVD-PAR model assessment ---")

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
        print("Data file read failed")
        return

    # Aggressive ID cleaning
    for df in [base_df, item_df]:
        for col in ['app_no', 'pat_id']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.replace(r'\.0$', '', regex=True)

    item_df['cat'] = item_df['res_itm_name'].apply(get_category)
    filtered = item_df.dropna(subset=['cat', 'res_value']).copy()

    records = []
    for (app_no, pat_id), group in filtered.groupby(['app_no', 'pat_id']):
        row = {'app_no': app_no, 'pat_id': pat_id}
        data_pool = {k: [] for k in FIELD_MAPPING.keys()}
        for _, item in group.iterrows():
            data_pool[item['cat']].append(str(item['res_value']))

        row['sbp'] = next((parse_sbp_from_bp(v) for v in data_pool['sbp'] if not pd.isna(parse_sbp_from_bp(v))), np.nan)
        row['TC'] = next((clean_numeric(v) for v in data_pool['TC'] if not pd.isna(clean_numeric(v))), np.nan)
        row['HDL_C'] = next((clean_numeric(v) for v in data_pool['HDL_C'] if not pd.isna(clean_numeric(v))), np.nan)
        row['wc'] = next((clean_numeric(v) for v in data_pool['wc'] if not pd.isna(clean_numeric(v))), np.nan)
        row['history_column'] = " ".join(data_pool['history_column'])
        row['smoke'] = " ".join(data_pool['smoke'])
        row['ASCVD_family_history'] = " ".join(data_pool['ASCVD_family_history'])
        row['residential_area'] = " ".join(data_pool['residential_area'])
        records.append(row)

    wide_items = pd.DataFrame(records)
    final = pd.merge(wide_items, base_df, on=['app_no', 'pat_id'],
                     how='inner') if not wide_items.empty else base_df.copy()

    results = []
    for _, r in final.iterrows():
        hist_text = f"{r.get('history_column', '')} {r.get('past_history', '')} {r.get('past_medical_history', '')}"
        is_diagnosed = parse_medical_history(hist_text, "coronary_heart_disease")
        age = pd.to_numeric(r.get('app_pat_age', r.get('age')), errors='coerce')
        gender = str(r.get('pat_sex', r.get('gender', '')))

        # Handle missing residence
        region_raw = f"{r.get('residential_area', '')} {r.get('residence', '')}".strip()
        display_region = "missing (defaulted to southern urban)" if not region_raw or region_raw.lower() in ['nan',
                                                                                            'none'] else region_raw

        # Core indicator determination
        has_diabetes = parse_medical_history(hist_text, "diabetes")
        is_treated_status = parse_medical_history(hist_text, r"hypertension|high_bp|high_normal_bp", check_untreated=True)
        smoke_context = f"{r.get('smoke', '')} {r.get('smoking_history', '')}"
        is_smoker_status = parse_smoking(smoke_context) or parse_medical_history(smoke_context, "smoking")
        family_context = f"{r.get('ASCVD_family_history', '')} {r.get('ascvd_family_history', '')}"
        has_family_status = parse_medical_history(family_context, "present") or ('yes' in family_context)

        if is_diagnosed:
            risk, risk_level, risk_code = None, "already diagnosed; not computed", "N/A"
        elif pd.isna(age) or not (35 <= age <= 74):
            risk, risk_level, risk_code = None, "-", "N/A"
        else:
            risk = calculate_china_par_risk(
                gender=gender, age=age, sbp=r.get('sbp'), TC=r.get('TC'),
                HDL_C=r.get('HDL_C'), wc=r.get('wc'), is_treated=is_treated_status,
                has_diabetes=has_diabetes, is_smoker=is_smoker_status,
                region=region_raw, family_history=has_family_status
            )
            risk_level, risk_code = get_risk_info(risk)

        results.append({
            'pat_id': r.get('pat_id', ''),
            'app_no': r.get('app_no', ''),
            'pat_name': r.get('pat_name', r.get('name', 'anonymous')),
            'age': int(age) if pd.notna(age) else 'missing',
            'sex': gender,
            'systolic_bp (mmHg)': r.get('sbp') if pd.notna(r.get('sbp')) else "missing",
            'total_cholesterol (mmol/L)': r.get('TC') if pd.notna(r.get('TC')) else "missing",
            'HDL (mg/dL)': r.get('HDL_C') if pd.notna(r.get('HDL_C')) else "missing",
            'waist (cm)': r.get('wc') if pd.notna(r.get('wc')) else "missing",
            'smoking_history (determined)': "yes" if is_smoker_status else "no",
            'diabetes_history (determined)': "yes" if has_diabetes else "no",
            'hypertension_treatment_status (determined)': "yes" if is_treated_status else "no",
            'ascvd_family_history (determined)': "yes" if has_family_status else "no",
            'residence': display_region,
            'risk_probability': f"{risk:.2f}%" if risk is not None else "-",
            'prediction_horizon': 'next 10 years',
            'risk_level': risk_level,
            'code': risk_code,
            'gauge_reference': "0-5%|5%-10%|10%-100%",
            'model_name': 'ASCVD-PAR model'
        })

    if results:
        pd.DataFrame(results).to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"Assessment complete: {len(results)} records.")


if __name__ == "__main__":
    process_data("base.csv", "item_result.csv", "china_PAR_result.csv")
