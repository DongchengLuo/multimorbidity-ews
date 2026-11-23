import pandas as pd
import numpy as np
import math
import re
from model_unit_utils import convert_measurement

# ==========================================
# 1. Exact-match field mapping
# Hard-code strictly per the given reference table; drop fuzzy matching
# ==========================================
FIELD_MAPPING = {
    'age': ['app_pat_age'],
    'gender': ['pat_sex'],
    'TC': ['total_cholesterol', 'CHOL'],
    'HDL-C': ['HDL_cholesterol', 'HDL', 'HDL-C'],
    'sbp_combined': ['blood_pressure'],
    'history_labels': ['past_history'],
    'smoke': ['smoking_history'],
    'has_diabetes_direct': ['diabetes_history']
}

# ==========================================
# 2. Model constants
# ==========================================
MODEL_NAME = "Framingham 2008 General CVD-Derived CHF Risk Model"
GAUGE_REF_STRING = "0%-100%"


# ==========================================
# 3. Data extraction and parsing logic
# ==========================================
def get_category_exact(name):
    """Exact matching of header field text"""
    if pd.isna(name):
        return None
    name_str = str(name).strip()
    for key, exact_names in FIELD_MAPPING.items():
        if name_str in exact_names:
            return key
    return None


def clean_numeric_strict(val):
    if pd.isna(val) or str(val).strip() == "":
        return np.nan
    try:
        return float(val)
    except ValueError:
        return np.nan


def extract_sbp_from_combined(bp_str):
    if pd.isna(bp_str) or str(bp_str).strip() == "":
        return np.nan
    nums = re.findall(r"\d+", str(bp_str))
    if len(nums) >= 1:
        return float(nums[0])
    return np.nan


def parse_gender_exact(val):
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    if s in ['female', 'Female', '2', 'F']:
        return 1
    if s in ['male', 'Male', '1', 'M']:
        return 0
    return np.nan


def parse_bool_flexible(val, pos_keywords, neg_keywords):
    """
    For non-normalized text such as smoking history or standalone diabetes history.
    Match explicit negative phrases first, then positive phrases.
    """
    if pd.isna(val) or str(val).strip() == "":
        return np.nan

    s = str(val).strip()

    for neg in neg_keywords:
        if neg in s:
            return 0

    for pos in pos_keywords:
        if pos in s:
            return 1

    return np.nan


# ==========================================
# 4. Clinical NLP: high-precision past-history extraction engine
# ==========================================
def parse_clinical_text(history_text, positive_keywords, negative_keywords=None, negation_words=None):
    if pd.isna(history_text) or str(history_text).strip() == "":
        return np.nan

    text = str(history_text).strip()

    if negation_words is None:
        negation_words = ["absent", "denies", "not found", "no disease", "none", "no"]

    if text in ["absent", "no", "-", "no_special_findings", "healthy", "no_disease_found"]:
        return 0

    if negative_keywords:
        for kw in negative_keywords:
            if kw in text:
                return 0

    clauses = re.split(r'[，。；,;！!、\n\s]+', text)
    for clause in clauses:
        for kw in positive_keywords:
            if kw in clause:
                is_negated = False
                for neg in negation_words:
                    if neg in clause and clause.find(neg) < clause.find(kw):
                        is_negated = True
                        break
                if not is_negated:
                    return 1

    return 0


def extract_diabetes_exact(text):
    dm_keywords = ['diabetes', 'glucose_lowering', 'insulin', 'T2DM', 'T1DM']
    return parse_clinical_text(text, positive_keywords=dm_keywords)


def extract_htn_treatment_exact(text):
    """
    Returns:
    1 = explicitly on antihypertensive treatment
    0 = hypertension present but untreated / unmedicated, or history only with no medication detected
    NaN = undetermined
    """
    if pd.isna(text) or str(text).strip() == "":
        return np.nan

    negative_treatment_kws = ['untreated', 'not_medicated', 'discontinued', 'refuses_treatment', 'irregular_medication', 'no_intervention']
    positive_action_kws = ['on_treatment', 'regular_medication', 'controlled_on_medication', 'on_medication', 'antihypertensive_drug', 'antihypertensive_treatment']
    drug_kws = [
        'dihydropyridine', 'nifedipine', 'amlodipine', 'felodipine', 'ramipril',
        'sartan', 'valsartan', 'irbesartan', 'losartan', 'candesartan',
        'pril', 'enalapril', 'captopril', 'benazepril',
        'lol', 'metoprolol', 'bisoprolol', 'atenolol',
        'zosin', 'terazosin',
        'diuretic', 'hydrochlorothiazide', 'furosemide', 'spironolactone'
    ]
    all_positive_kws = positive_action_kws + drug_kws

    if parse_clinical_text(text, positive_keywords=[], negative_keywords=negative_treatment_kws) == 0:
        return 0

    treatment_status = parse_clinical_text(text, positive_keywords=all_positive_kws)
    if treatment_status == 1:
        return 1

    htn_disease_kws = ['hypertension', 'high_bp']
    if parse_clinical_text(text, positive_keywords=htn_disease_kws) == 1:
        return 0

    return np.nan


# ==========================================
# 6. Core computation logic (Framingham 2008 Cox regression)
# ==========================================
def calculate_framingham_chf_risk(age, is_female, total_c_mgdl, hdl_c_mgdl, sbp, is_treated, is_smoker, is_diabetic):
    indicators = [age, is_female, total_c_mgdl, hdl_c_mgdl, sbp, is_treated, is_smoker, is_diabetic]
    if any(pd.isna(x) for x in indicators):
        return np.nan, None

    if age <= 0 or total_c_mgdl <= 0 or hdl_c_mgdl <= 0 or sbp <= 0:
        return np.nan, None

    ln_age = math.log(age)
    ln_tc = math.log(total_c_mgdl)
    ln_hdl = math.log(hdl_c_mgdl)
    ln_sbp = math.log(sbp)

    if is_female == 1:
        beta = {
            "age": 2.32888, "tc": 1.20904, "hdl": -0.70833,
            "sbp_untrt": 2.76157, "sbp_trt": 2.82263,
            "smoke": 0.52873, "diab": 0.69154
        }
        s10 = 0.95012
        xm_beta = 26.1931
        cf_chf = 0.1250
    else:
        beta = {
            "age": 3.06117, "tc": 1.12370, "hdl": -0.93263,
            "sbp_untrt": 1.93303, "sbp_trt": 1.99881,
            "smoke": 0.65451, "diab": 0.57367
        }
        s10 = 0.88936
        xm_beta = 23.9802
        cf_chf = 0.1148

    sbp_component = beta["sbp_trt"] * ln_sbp if is_treated == 1 else beta["sbp_untrt"] * ln_sbp

    xi_beta = (
        ln_age * beta["age"] +
        ln_tc * beta["tc"] +
        ln_hdl * beta["hdl"] +
        sbp_component +
        is_smoker * beta["smoke"] +
        is_diabetic * beta["diab"]
    )

    p_cvd = 1 - (s10 ** math.exp(xi_beta - xm_beta))
    p_cvd = min(max(p_cvd, 0), 1)

    p_chf = p_cvd * cf_chf
    p_chf = min(max(p_chf, 0), 1)

    return p_chf * 100, {
        "P_General_CVD": p_cvd * 100,
        "Calibration_Factor": cf_chf
    }


# ==========================================
# 7. Main data-processing routine
# ==========================================
def process_data(base_path, item_path, output_path):
    print(f"--- Starting {MODEL_NAME} assessment ---")

    def read_csv_safe(path):
        for enc in ['gbk', 'utf-8-sig', 'utf-8']:
            try:
                return pd.read_csv(path, encoding=enc, low_memory=False)
            except Exception:
                continue
        return pd.DataFrame()

    base_df = read_csv_safe(base_path)
    item_df = read_csv_safe(item_path)

    if base_df.empty or item_df.empty:
        print("error: cannot read the input file.")
        return

    item_df['cat'] = item_df['res_itm_name'].apply(get_category_exact)
    filtered = item_df.dropna(subset=['cat', 'res_value']).copy()

    records = []
    for (app_no, pat_id), group in filtered.groupby(['app_no', 'pat_id']):
        row = {'app_no': str(app_no).strip(), 'pat_id': str(pat_id).strip()}

        for std_key in FIELD_MAPPING.keys():
            cat_rows = group[group['cat'] == std_key]
            row[std_key] = cat_rows.iloc[0]['res_value'] if not cat_rows.empty else np.nan
            row[f'{std_key}_unit'] = cat_rows.iloc[0].get('res_unit', '') if not cat_rows.empty else ''

        # Raw value
        row['TC_raw_val'] = clean_numeric_strict(row.get('TC'))
        row['HDL_raw_val'] = clean_numeric_strict(row.get('HDL-C'))
        row['sbp_raw_val'] = extract_sbp_from_combined(row.get('sbp_combined'))

        # Convert units only per res_unit as the source requires; never guess from magnitude.
        tc_result = convert_measurement(row['TC_raw_val'], row.get('TC_unit'), indicator='cholesterol', target_unit='mg/dL')
        hdl_result = convert_measurement(row['HDL_raw_val'], row.get('HDL-C_unit'), indicator='cholesterol', target_unit='mg/dL')
        sbp_result = convert_measurement(row['sbp_raw_val'], row.get('sbp_combined_unit'), indicator='blood_pressure', target_unit='mmHg')
        row['TC_mgdl'] = tc_result.value if tc_result.ok else np.nan
        row['HDL_mgdl'] = hdl_result.value if hdl_result.ok else np.nan
        row['sbp_val'] = sbp_result.value if sbp_result.ok else np.nan
        row['TC_unit_note'] = tc_result.rule or tc_result.error
        row['HDL_unit_note'] = hdl_result.rule or hdl_result.error
        row['SBP_unit_note'] = sbp_result.rule or sbp_result.error

        # Hypertension treatment status
        row['bp_treated_val'] = extract_htn_treatment_exact(row.get('history_labels'))

        # Diabetes history: prefer the dedicated field, fall back to past history when missing
        diab_direct = parse_bool_flexible(
            row.get('has_diabetes_direct'),
            pos_keywords=['yes', 'present', '1', 'positive', 'diabetes'],
            neg_keywords=['no', 'absent', '0', 'negative', 'no_diabetes']
        )
        diab_hist = extract_diabetes_exact(row.get('history_labels'))
        row['diabetes_val'] = diab_direct if pd.notna(diab_direct) else diab_hist

        # Smoking history: avoid overly broad single-character matches such as "no/not/not yet"
        row['smoke_val'] = parse_bool_flexible(
            row.get('smoke'),
            pos_keywords=['smoking', 'smoking_history_present', 'long_term_smoking', 'current_smoker', 'occasional_smoker'],
            neg_keywords=['no_smoking_history', 'denies_smoking', 'non-smoker', 'never_smoker', 'former_smoker', 'quit_smoking', 'never_smoked']
        )

        records.append(row)

    wide_items = pd.DataFrame(records)
    if wide_items.empty:
        print("No valid indicator data extracted.")
        return

    base_df['app_no'] = base_df['app_no'].astype(str).str.strip()
    base_df['pat_id'] = base_df['pat_id'].astype(str).str.strip()
    final = pd.merge(wide_items, base_df, on=['app_no', 'pat_id'], how='inner')

    results = []
    for _, r in final.iterrows():
        age_val = clean_numeric_strict(r.get('age') if pd.notna(r.get('age')) else r.get('app_pat_age'))
        gender_val = parse_gender_exact(r.get('gender') if pd.notna(r.get('gender')) else r.get('pat_sex'))

        risk, detail = calculate_framingham_chf_risk(
            age=age_val,
            is_female=gender_val,
            total_c_mgdl=r['TC_mgdl'],
            hdl_c_mgdl=r['HDL_mgdl'],
            sbp=r['sbp_val'],
            is_treated=r['bp_treated_val'],
            is_smoker=r['smoke_val'],
            is_diabetic=r['diabetes_val']
        )

        gender_str = "female" if gender_val == 1 else ("male" if gender_val == 0 else "-")
        treated_str = "yes" if r['bp_treated_val'] == 1 else ("no" if r['bp_treated_val'] == 0 else "-")
        smoke_str = "yes" if r['smoke_val'] == 1 else ("no" if r['smoke_val'] == 0 else "-")
        diab_str = "yes" if r['diabetes_val'] == 1 else ("no" if r['diabetes_val'] == 0 else "-")

        if pd.isna(risk):
            display_score = "-"
            risk_level = "-"
            risk_code = "-"
            p_cvd_out = "-"
        else:
            display_score = f"{risk:.2f}%"
            p_cvd_out = f"{detail['P_General_CVD']:.2f}%"
            risk_level = "continuous_heart_failure_risk"
            risk_code = "-"

        results.append({
            'app_no': r['app_no'],
            'pat_id': r['pat_id'],
            'pat_name': r.get('pat_name', 'unknown'),
            'age': age_val if pd.notna(age_val) else "-",
            'sex': gender_str,

            # Keep raw values for verification
            'total_cholesterol (TC)': r['TC_raw_val'] if pd.notna(r['TC_raw_val']) else "-",
            'HDL-C': r['HDL_raw_val'] if pd.notna(r['HDL_raw_val']) else "-",

            # Values actually used by the formula
            'total_cholesterol (TC, mg/dL)': round(r['TC_mgdl'], 2) if pd.notna(r['TC_mgdl']) else "-",
            'HDL (HDL-C, mg/dL)': round(r['HDL_mgdl'], 2) if pd.notna(r['HDL_mgdl']) else "-",
            'TC_raw_unit': r.get('TC_unit', '-'),
            'HDL_raw_unit': r.get('HDL-C_unit', '-'),
            'BP_raw_unit': r.get('sbp_combined_unit', '-'),
            'unit handling': f"TC[{r['TC_unit_note']}]；HDL[{r['HDL_unit_note']}]；SBP[{r['SBP_unit_note']}]",

            'extracted_systolic_bp (SBP)': r['sbp_val'] if pd.notna(r['sbp_val']) else "-",
            'hypertension/medication history (NLP-extracted)': treated_str,
            'smoking_history': smoke_str,
            'diabetes_history (multi-source)': diab_str,
            'overall_CVD_probability (P_CVD)': p_cvd_out,
            'risk_probability': display_score,
            'risk_level': risk_level,
            'code': risk_code,
            'gauge_reference': GAUGE_REF_STRING,
            'model_name': MODEL_NAME
        })

    if results:
        res_df = pd.DataFrame(results)
        res_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f" assessment complete! File saved to: {output_path}")
    else:
        print(" No assessment records generated.")


if __name__ == "__main__":
    process_data("base.csv", "item_result.csv", "CHF_Framingham_result.csv")
