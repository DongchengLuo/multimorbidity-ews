import pandas as pd
import numpy as np
import re
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model_unit_utils import convert_measurement

# ==========================================
# 1. Metadata and field mapping
# ==========================================
MODEL_NAME = "PREVENT ASCVD"
GAUGE_REF = "0%-7.5%|7.5%-20%|20%-100%"

FIELD_MAPPING = {
    'age': ['app_pat_age', 'age'],
    'gender': ['pat_sex', 'sex'],
    'sbp': ['systolic_bp', 'blood_pressure', 'SBP'],
    'TC': ['total_cholesterol', 'CHOL', 'TC'],
    'HDL-C': ['HDL_cholesterol', 'HDL', 'HDL-C'],
    'egfr': ['eGFR', 'eGFR (CKD-EPI)'],
    'scr': ['creatinine (Cr)', 'creatinine', 'serum_creatinine'],

    # Split out past history separately
    'history_shared': ['past_history'],

    'diabetes': ['diabetes', 'prior_diabetes'],
    'smoke': ['smoking_history', 'smoking', 'smoke', 'smoker'],
    'bp_treatment': ['antihypertensive', 'on_antihypertensive', 'antihypertensive_treatment'],
    'statin_use': ['statin', 'lipid_lowering', 'on_statin'],
    'has_ascvd': ['diagnosed_ASCVD', 'prior_ascvd', 'CHD_or_stroke_history', 'has_ascvd'],
    'has_hf': ['diagnosed_heart_failure', 'heart_failure_history', 'has_hf'],

    # Lab indicators for diabetes determination
    'glucose': [
        'GLU_1h_postprandial',
        'GLU_2h_postprandial',
        'GLU_3h_postprandial',
        'GLU_30min_postprandial',
        'blood_glucose_micro',
        'GLU_2h_postprandial_micro'
    ],
    'FBG': ['GLU'],
    'Hba1c': [
        'HbA1c',
        'GHb'
    ]
}

# ==========================================
# 2. Basic helpers and context-cleaning logic
# ==========================================
def normalize_str(s):
    if pd.isna(s):
        return ""
    return re.sub(r'[\(\)\（\）\[\]\【\】\{\}\s\|\、]', '', str(s)).lower()

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
        match = re.search(r"[-+]?\d*\.\d+|\d+", str(val))
        return float(match.group()) if match else np.nan
    except:
        return np.nan

def first_valid_numeric(*vals):
    for v in vals:
        num = clean_numeric(v)
        if not pd.isna(num):
            return num
    return np.nan

def check_negation(text, targets):
    """Precise context analysis: avoid negations falsely killing medication/history"""
    text = re.sub(r'\s+', '', str(text)).lower()
    if not any(t in text for t in targets):
        return False
    clauses = re.split(r'[。；;\n]', text)
    for clause in clauses:
        for target in targets:
            if target in clause:
                if re.search(r'(denies|no|none|not found|not seen|not taking|not using|not eating)[^present]*?' + re.escape(target), clause):
                    continue
                if re.search(re.escape(target) + r'.{0,6}?(no|denies|none|not taking|not using|not eating|negative)', clause):
                    continue
                return True
    return False

def parse_history(key, val_list):
    """
    Integrated history and medication-name detection
    """
    clean_vals = [str(v).strip() for v in val_list if not pd.isna(v) and str(v).strip() != ""]
    if not clean_vals:
        return np.nan

    positive_words = {'1', 'yes', 'present', 'true', 'positive'}
    negative_words = {'0', 'no', 'absent', 'false', 'negative', 'not seen', 'denies', 'none'}

    lowered = [v.lower() for v in clean_vals]
    if any(v in positive_words for v in lowered):
        return 1
    if all(v in negative_words for v in lowered):
        return 0

    # Joining with spaces is safe here because val_list now belongs to a single field
    combined_text = " ".join(clean_vals)

    mapping = {
        'diabetes': ['diabetes', 'glucose_lowering', 'insulin', 'metformin', 'acarbose'],
        'bp_treatment': ['antihypertensive', 'dihydropyridine', 'sartan', 'pril', 'lol', 'thiazide'],
        'statin_use': ['statin', 'lipid_lowering', 'lipids', 'atorvastatin', 'rosuvastatin'],
        'smoke': ['smoking', 'smoker']
        , 'has_ascvd': ['ascvd', 'atherosclerotic_cardiovascular_disease', 'coronary_heart_disease', 'myocardial_infarction', 'stroke']
        , 'has_hf': ['heart_failure', 'heartfailure']
    }

    if check_negation(combined_text, mapping.get(key, [])):
        return 1

    if re.search(r'(denies|no|none|not seen|not taking|not using|not eating|negative)', combined_text):
        return 0

    return np.nan

def parse_shared_history_status(key, val_list):
    """
    Dedicated parsing of the shared "past history" field
    """
    clean_vals = [str(v).strip() for v in val_list if not pd.isna(v) and str(v).strip() != ""]
    if not clean_vals:
        return np.nan

    alias_map = {
        'diabetes': ['diabetes', 'prior_diabetes'],
        'bp_treatment': ['antihypertensive_treatment', 'on_antihypertensive', 'antihypertensive', 'antihypertensive_therapy'],
        'statin_use': ['statin_therapy', 'on_statin', 'statin', 'lipid_lowering_therapy'],
        'smoke': ['smoking_history', 'smoker', 'smoking']
        , 'has_ascvd': ['prior_ascvd', 'atherosclerotic_cardiovascular_disease', 'coronary_heart_disease', 'myocardial_infarction', 'stroke']
        , 'has_hf': ['heart_failure_history', 'heart_failure']
    }

    positive_tokens = ['yes', 'present', 'positive', 'true', 'taking', 'use', 'oral', 'on_treatment']
    negative_tokens = ['no', 'absent', 'negative', 'false', 'not_taking', 'not_using', 'not_eating', 'discontinued', 'denies', 'none']

    clauses = re.split(r'[。；;\n]', "；".join(clean_vals))
    aliases = alias_map.get(key, [])

    for clause in clauses:
        clause_clean = re.sub(r'\s+', '', str(clause)).lower()
        for alias in aliases:
            alias_clean = str(alias).lower()
            if alias_clean in clause_clean:
                pos = clause_clean.find(alias_clean)
                before = clause_clean[max(0, pos - 8):pos]
                after = clause_clean[pos:pos + 20]

                if any(tok in before or tok in after for tok in negative_tokens):
                    return 0
                if any(tok in before or tok in after for tok in positive_tokens):
                    return 1

                if key == 'diabetes':
                    return 1

    return np.nan

# ==========================================
# [Key change 1]: separate extraction of dedicated history and past history
# ==========================================
def resolve_history_value(key, specific_vals=None, shared_history_vals=None):
    """
    Parse dedicated fields and past history separately: judge dedicated fields first, then past history if inconclusive
    """
    specific_vals = specific_vals or []
    shared_history_vals = shared_history_vals or []

    # 1. Parse dedicated fields first (e.g. only the smoking-history box)
    specific_status = parse_history(key, specific_vals)
    if not pd.isna(specific_status):
        return specific_status

    # 2. If dedicated fields are inconclusive, parse the shared "past history" field independently
    shared_status = parse_shared_history_status(key, shared_history_vals)
    if not pd.isna(shared_status):
        return shared_status

    # 3. Past-history fallback extraction (kept separate; never blindly concatenated with dedicated fields)
    return parse_history(key, shared_history_vals)

# ==========================================
# 2.1 Diabetes criteria
# ==========================================
def value_meets_cutoff(val, mmol_threshold, mgdl_threshold):
    num = clean_numeric(val)
    if pd.isna(num):
        return False

    text = str(val).lower().replace(" ", "")
    if 'mg/dl' in text or 'mgdl' in text:
        return num >= mgdl_threshold
    if 'mmol/l' in text or 'mmol' in text:
        return num >= mmol_threshold

    if num < 30:
        return num >= mmol_threshold
    return num >= mgdl_threshold

# ==========================================
# [Key change 2]: separate text-based diabetes detection
# ==========================================
def extract_diabetes_status(diabetes_vals=None, hba1c_vals=None, fbg_vals=None, glucose_vals=None, shared_history_vals=None):
    diabetes_vals = diabetes_vals or []
    hba1c_vals = hba1c_vals or []
    fbg_vals = fbg_vals or []
    glucose_vals = glucose_vals or []
    shared_history_vals = shared_history_vals or []

    # --- Laboratory indicator evaluation ---
    for v in hba1c_vals:
        num = clean_numeric(v)
        if not pd.isna(num) and num >= 6.5:
            return 1

    for v in fbg_vals:
        if value_meets_cutoff(v, mmol_threshold=7.0, mgdl_threshold=126):
            return 1

    for v in glucose_vals:
        if value_meets_cutoff(v, mmol_threshold=11.1, mgdl_threshold=200):
            return 1

    # --- Text-history evaluation (separate independent checks) ---
    # 1. Prefer parsing the dedicated diabetes text field first
    specific_status = parse_history('diabetes', diabetes_vals)
    if not pd.isna(specific_status):
        return specific_status

    # 2. If dedicated fields are inconclusive, parse the "past history" field independently
    shared_status = parse_shared_history_status('diabetes', shared_history_vals)
    if not pd.isna(shared_status):
        return shared_status

    # 3. Past-history fallback extraction
    return parse_history('diabetes', shared_history_vals)

def calculate_egfr_ckd_epi(scr, age, gender):
    """scr must already be normalized to mg/dL by the unit-conversion layer."""
    if pd.isna(scr) or pd.isna(age) or gender not in ['male', 'female']:
        return None
    s_cr = scr
    kappa, alpha = (0.7, -0.241) if gender == 'female' else (0.9, -0.302)
    try:
        egfr = (
            142
            * (min(s_cr / kappa, 1) ** alpha)
            * (max(s_cr / kappa, 1) ** -1.200)
            * (0.9938 ** age)
            * (1.012 if gender == 'female' else 1.0)
        )
        return round(egfr, 2)
    except:
        return None

def normalize_sex(sex_raw):
    if pd.isna(sex_raw) or str(sex_raw).strip() == "":
        return None
    s = str(sex_raw).strip()
    s_upper = s.upper()
    if ('male' in s) or (s_upper == 'M') or ('MALE' in s_upper):
        return 'male'
    if ('female' in s) or (s_upper == 'F') or ('FEMALE' in s_upper):
        return 'female'
    return None

def format_item_value(val, as_int=False):
    if pd.isna(val) or val is None or str(val).strip() == "":
        return "missing"
    if as_int:
        try:
            return int(float(val))
        except:
            return val
    return val

def format_yes_no_missing(val):
    if pd.isna(val):
        return "missing"
    return "yes" if int(val) == 1 else "no"

# ==========================================
# 3. PREVENT-ASCVD 2023 core algorithm
# ==========================================
def calculate_prevent_ascvd_2023(sex, age, smoke, sbp, tc, hdl, diabetes, bp_treat, statin, egfr):
    required_values = [age, smoke, sbp, tc, hdl, diabetes, bp_treat, statin, egfr]
    if not sex or any(pd.isna(v) for v in required_values):
        return None

    is_female = (sex == 'female')

    age_std = (age - 55) / 10
    non_hdl_val = ((tc - hdl) * 0.02586 - 3.5)
    hdl_std = (hdl * 0.02586 - 1.3) / 0.3
    sbp_lt110 = (min(sbp, 110) - 110) / 20
    sbp_gt110 = (max(sbp, 110) - 130) / 20
    egfr_lt60 = (min(egfr, 60) - 60) / -15
    egfr_gt60 = (max(egfr, 60) - 90) / -15

    try:
        if is_female:
            log_odds = (
                -3.819975 + 0.719883 * age_std + 0.1176967 * non_hdl_val - 0.151185 * hdl_std
                - 0.0835358 * sbp_lt110 + 0.3592852 * sbp_gt110 + 0.8348585 * diabetes
                + 0.4831078 * smoke + 0.4864619 * egfr_lt60 + 0.0397779 * egfr_gt60
                + 0.2265309 * bp_treat - 0.0592374 * statin - 0.0395762 * bp_treat * sbp_gt110
                + 0.0844423 * statin * non_hdl_val - 0.0567839 * age_std * non_hdl_val
                + 0.0325692 * age_std * hdl_std - 0.1035985 * age_std * sbp_gt110
                - 0.2417542 * age_std * diabetes - 0.0791142 * age_std * smoke
                - 0.1671492 * age_std * egfr_lt60
            )
        else:
            log_odds = (
                -3.500655 + 0.7099847 * age_std + 0.1658663 * non_hdl_val - 0.1144285 * hdl_std
                - 0.2837212 * sbp_lt110 + 0.3239977 * sbp_gt110 + 0.7189597 * diabetes
                + 0.3956973 * smoke + 0.3690075 * egfr_lt60 + 0.0203619 * egfr_gt60
                + 0.2036522 * bp_treat - 0.0865581 * statin - 0.0322916 * bp_treat * sbp_gt110
                + 0.114563 * statin * non_hdl_val - 0.0300005 * age_std * non_hdl_val
                + 0.0232747 * age_std * hdl_std - 0.0927024 * age_std * sbp_gt110
                - 0.2018525 * age_std * diabetes - 0.0970527 * age_std * smoke
                - 0.1217081 * age_std * egfr_lt60
            )

        return round((math.exp(log_odds) / (1 + math.exp(log_odds))) * 100, 2)
    except:
        return None

# ==========================================
# 4. Main data-processing routine
# ==========================================
def process_data(base_path, item_path, output_path):
    print(f"--- Starting {MODEL_NAME} assessment ---")

    def read_csv_safe(path):
        for enc in ['utf-8-sig', 'gb18030', 'gbk', 'utf-8']:
            try:
                return pd.read_csv(path, encoding=enc, low_memory=False)
            except:
                continue
        return pd.DataFrame()

    base_df = read_csv_safe(base_path)
    item_df = read_csv_safe(item_path)

    if base_df.empty or item_df.empty:
        print("data file unreadable; check path and encoding")
        return

    item_df['cat'] = item_df['res_itm_name'].apply(get_category)
    filtered = item_df.dropna(subset=['cat']).copy()

    records = []
    for (app_no, pat_id), group in filtered.groupby(['app_no', 'pat_id']):
        row = {'app_no': str(app_no).strip(), 'pat_id': str(pat_id).strip()}
        data_pool = {k: group[group['cat'] == k]['res_value'].tolist() for k in FIELD_MAPPING.keys()}

        # Non-unit-sensitive numeric items
        for nf in ['sbp', 'egfr']:
            nums = [clean_numeric(v) for v in data_pool.get(nf, []) if not pd.isna(clean_numeric(v))]
            row[nf] = nums[0] if nums else np.nan

        # Unit-sensitive indicators: read res_value and res_unit from the same row.
        conversion_specs = {
            'TC': ('cholesterol', 'mg/dL'),
            'HDL-C': ('cholesterol', 'mg/dL'),
            'scr': ('creatinine', 'mg/dL'),
        }
        unit_notes, unit_errors = [], []
        for cat, (indicator, target_unit) in conversion_specs.items():
            cat_rows = group[group['cat'] == cat]
            converted_value = np.nan
            cat_errors = []
            for _, measurement in cat_rows.iterrows():
                result = convert_measurement(
                    measurement.get('res_value'),
                    measurement.get('res_unit', ''),
                    indicator=indicator,
                    target_unit=target_unit,
                )
                if result.ok:
                    converted_value = result.value
                    unit_notes.append(f"{cat}: {result.rule}")
                    break
                cat_errors.append(f"{cat}: {result.error}")
            if pd.isna(converted_value):
                unit_errors.extend(cat_errors)
            row[cat] = converted_value
        row['unit_conversion'] = '；'.join(unit_notes)
        row['unit_error'] = '；'.join(unit_errors)

        shared_history_vals = data_pool.get('history_shared', [])

        # PREVENT uses a binary "diagnosed diabetes" variable; elevated lab glucose cannot be used here
        # Never substitute for clinical diagnosis; avoid mistaking a single abnormal test for diabetes.
        row['diabetes'] = resolve_history_value(
            'diabetes',
            specific_vals=data_pool.get('diabetes', []),
            shared_history_vals=shared_history_vals,
        )

        row['smoke'] = resolve_history_value(
            'smoke',
            specific_vals=data_pool.get('smoke', []),
            shared_history_vals=shared_history_vals
        )
        row['bp_treatment'] = resolve_history_value(
            'bp_treatment',
            specific_vals=data_pool.get('bp_treatment', []),
            shared_history_vals=shared_history_vals
        )
        row['statin_use'] = resolve_history_value(
            'statin_use',
            specific_vals=data_pool.get('statin_use', []),
            shared_history_vals=shared_history_vals
        )
        row['has_ascvd'] = resolve_history_value(
            'has_ascvd',
            specific_vals=data_pool.get('has_ascvd', []),
            shared_history_vals=shared_history_vals,
        )
        row['has_hf'] = resolve_history_value(
            'has_hf',
            specific_vals=data_pool.get('has_hf', []),
            shared_history_vals=shared_history_vals,
        )

        # Project compatibility mode: missing binary history/medication fields keep the original 0-point handling.
        for binary_key in ('diabetes', 'smoke', 'bp_treatment', 'statin_use', 'has_ascvd', 'has_hf'):
            if pd.isna(row[binary_key]):
                row[binary_key] = 0

        records.append(row)

    if records:
        wide_items = pd.DataFrame(records)
    else:
        wide_items = pd.DataFrame(columns=[
            'app_no', 'pat_id', 'sbp', 'TC', 'HDL-C', 'egfr', 'scr',
            'diabetes', 'smoke', 'bp_treatment', 'statin_use', 'has_ascvd', 'has_hf'
        ])

    base_df['app_no'] = base_df['app_no'].astype(str).str.strip()
    base_df['pat_id'] = base_df['pat_id'].astype(str).str.strip()

    final = pd.merge(base_df, wide_items, on=['app_no', 'pat_id'], how='left')

    results = []
    for _, r in final.iterrows():
        age = first_valid_numeric(r.get('app_pat_age'), r.get('age'))

        sex_raw = r.get('pat_sex')
        if pd.isna(sex_raw) or str(sex_raw).strip() == "":
            sex_raw = r.get('gender')
        sex = normalize_sex(sex_raw)

        sbp_val = r.get('sbp', np.nan)
        tc_val = r.get('TC', np.nan)
        hdl_val = r.get('HDL-C', np.nan)
        scr_val = r.get('scr', np.nan)
        egfr_val = r.get('egfr', np.nan)

        if pd.isna(egfr_val) and not pd.isna(scr_val) and not pd.isna(age) and sex in ['male', 'female']:
            calc_egfr = calculate_egfr_ckd_epi(scr_val, age, sex)
            egfr_val = calc_egfr if calc_egfr is not None else np.nan

        diabetes_val = r.get('diabetes', np.nan)
        smoke_val = r.get('smoke', np.nan)
        bp_treat_val = r.get('bp_treatment', np.nan)
        statin_val = r.get('statin_use', np.nan)
        has_ascvd = r.get('has_ascvd', np.nan)
        has_hf = r.get('has_hf', np.nan)

        required = {
            'sex': sex, 'age': age, 'smoking_status': smoke_val, 'systolic_bp': sbp_val,
            'total_cholesterol': tc_val, 'HDL': hdl_val, 'diabetes_status': diabetes_val,
            'antihypertensive_treatment_status': bp_treat_val, 'statin_treatment_status': statin_val, 'eGFR': egfr_val,
        }
        missing = [name for name, value in required.items() if value is None or pd.isna(value)]

        if missing:
            score = None
            calc_status = 'input_incomplete'
            reason_parts = []
            reason_parts.append('missing:' + '、'.join(missing))
            if r.get('unit_error', ''):
                reason_parts.append(str(r.get('unit_error')))
            no_calc_reason = '；'.join(reason_parts)
        else:
            score = calculate_prevent_ascvd_2023(
                sex, age, smoke_val, sbp_val, tc_val, hdl_val,
                diabetes_val, bp_treat_val, statin_val, egfr_val
            )
            calc_status = 'computed' if score is not None else 'computation_failed'
            no_calc_reason = '' if score is not None else 'numeric_anomaly_in_formula'

        if score is None:
            risk_score = "-"
            predict_time = "-"
            risk_layer = "-"
            code = "-"
        else:
            risk_score = f"{score}%"
            predict_time = "next 10 years"
            risk_layer = "low risk" if score < 7.5 else ("medium risk" if score < 20 else "high risk")
            code = "I25.2.1" if score < 7.5 else ("I25.2.2" if score < 20 else "I25.2.3")

        results.append({
            'pat_id': r.get('pat_id'),
            'app_no': r.get('app_no'),
            'pat_name': r.get('pat_name', 'unknown'),

            'sex': format_item_value(sex_raw),
            'age': format_item_value(age, as_int=True),

            'risk_probability': risk_score,
            'prediction_horizon': predict_time,
            'risk_level': risk_layer,
            'code': code,
            'calculation_status': calc_status,
            'not_computed_reason': no_calc_reason,

            'gauge_reference': GAUGE_REF,
            'model_name': MODEL_NAME,

            'systolic_bp': format_item_value(sbp_val),
            'total_cholesterol (mg/dL)': format_item_value(tc_val),
            'HDL (mg/dL)': format_item_value(hdl_val),
            'eGFR': format_item_value(egfr_val),
            'unit_conversion': r.get('unit_conversion', ''),
            'unit_error': r.get('unit_error', ''),
            'smoking': format_yes_no_missing(smoke_val),
            'diabetes': format_yes_no_missing(diabetes_val),
            'antihypertensive_treatment': format_yes_no_missing(bp_treat_val),
            'statin_therapy': format_yes_no_missing(statin_val)
            , 'prior_ascvd': format_yes_no_missing(has_ascvd)
            , 'heart_failure': format_yes_no_missing(has_hf)
        })

    res_df = pd.DataFrame(results)
    res_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"--- Processing complete. Results saved to {output_path} ---")

if __name__ == "__main__":
    process_data(
        "base.csv",
        "item_result.csv",
        "prevent_ASCVD_output.csv"
    )
