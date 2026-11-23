import pandas as pd
import numpy as np
import re
import math

# ==========================================
# 1. Field mapping
# ==========================================
FIELD_MAPPING = {
    'age': ['app_pat_age', 'age'],
    'gender': ['pat_sex', 'sex', 'gender'],
    'BMI': ['(body mass index) BMI', 'bmi', 'BMI', 'body_mass_index'],
    'height': ['height'],
    'weight': ['weight'],
    'bp_combined': ['blood_pressure'],
    'sbp': ['systolic_bp', 'systolic_bp (sbp)', 'sbp'],
    'dbp': ['diastolic_bp', 'diastolic_bp (dbp)'],
    'LDL_C': ['LDL', 'LDL-C', 'LDL_C', 'ldl'],
    'HDL_C': ['HDL', 'HDL-C', 'HDL_C', 'hdl'],
    'smoke': ['smoking_history', 'smoking_status', 'smoke', 'smoker'],
    'has_diabetes': ['prior_diabetes', 'type2_diabetes', 'diabetes'],
    'has_dyslipidemia': ['dyslipidemia_history', 'has_dyslipidemia', 'dyslipidemia'],
    'has_pregnancy_htn': ['gestational_hypertension_history', 'pregnancy_htn', 'PIH', 'gestational_hypertension']
}

MODEL_NAME = "JAHA 2025 Young Adult (18-39) Hypertension Risk Model"
GAUGE_REF_STRING = "0-5%|5-15%|15-100%"


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
            if normalize_str(p) in norm_name: return key
    return None


def clean_numeric(val):
    if pd.isna(val) or str(val).strip() == "": return np.nan
    try:
        match = re.search(r"[-+]?\d*\.\d+|\d+", str(val))
        return float(match.group()) if match else np.nan
    except:
        return np.nan


def parse_combined_bp(val):
    if pd.isna(val) or str(val).strip() == "": return np.nan, np.nan
    nums = re.findall(r"\d+\.?\d*", str(val))
    if len(nums) >= 2: return float(nums[0]), float(nums[1])
    return np.nan, np.nan


def parse_gender(val):
    s = str(val).strip()
    if any(k in s for k in ['female', 'Female', 'F', '2']): return 1
    if any(k in s for k in ['male', 'Male', 'M', '1']): return 0
    return np.nan


def parse_smoke_status(val):
    s = normalize_str(val)
    if not s: return 0, 0
    is_curr = 1 if any(k in s for k in ["current", "yes"]) else 0
    is_form = 1 if any(k in s for k in ["quit", "former", "past"]) else 0
    return is_curr, is_form


def parse_bool_history(val):
    # [Change 1]: return np.nan instead of 0 when raw data is missing, preserving the evidence of absence
    if pd.isna(val) or str(val).strip() == "": return np.nan
    s = normalize_str(val)
    if any(k in s for k in ['yes', 'present', '1', 'positive']): return 1
    return 0


# ==========================================
# 3. Core computation logic
# ==========================================
def calculate_jaha_risk(age, is_female, sbp, dbp, bmi, ldl, hdl, smoke_val, diabetes, dyslipidemia, preg_htn):
    essentials = [age, is_female, sbp, dbp, bmi, ldl, hdl]
    if any(pd.isna(x) for x in essentials) or not (18 <= age <= 39):
        return np.nan, None

    if is_female == 1:
        s10, xm_beta = 0.8960605, 42.10233
        beta = {
            "ln_age": 0.51310766, "ln_age_sq": 0.08080281, "bmi_25_30": 0.12920225, "bmi_30_plus": 0.4464456,
            "ln_sbp": 5.11938101, "ln_dbp": 3.60224434, "current_smoker": 0.19089945, "former_smoker": 0.08637985,
            "dyslipidemia": 0.18845726, "diabetes": 0.33920666, "ln_ldl": 0.01259775, "ln_hdl": -0.0010893,
            "pregnancy_htn": 0.42246845
        }
    else:
        s10, xm_beta = 0.825469, 37.37255
        beta = {
            "ln_age": 0.1618221, "ln_age_sq": 0.0267456, "bmi_25_30": 0.0766266, "bmi_30_plus": 0.3873278,
            "ln_sbp": 5.2360618, "ln_dbp": 2.6688015, "current_smoker": 0.1555035, "former_smoker": 0.114527,
            "dyslipidemia": 0.1425061, "diabetes": 0.3089758, "ln_ldl": 0.0062288, "ln_hdl": -0.0055955,
            "pregnancy_htn": 0
        }

    ln_age, ln_sbp, ln_dbp = math.log(age), math.log(sbp), math.log(dbp)
    is_curr, is_form = parse_smoke_status(smoke_val)

    xi_beta = (
            ln_age * beta["ln_age"] + (ln_age ** 2) * beta["ln_age_sq"] + ln_sbp * beta["ln_sbp"] + ln_dbp * beta[
        "ln_dbp"] +
            (1 if 25 <= bmi < 30 else 0) * beta["bmi_25_30"] + (1 if bmi >= 30 else 0) * beta["bmi_30_plus"] +
            is_curr * beta["current_smoker"] + is_form * beta["former_smoker"] +
            diabetes * beta["diabetes"] + dyslipidemia * beta["dyslipidemia"] +
            math.log(ldl) * beta["ln_ldl"] + math.log(hdl) * beta["ln_hdl"] +
            (preg_htn if is_female else 0) * beta["pregnancy_htn"]
    )

    risk = (1 - math.pow(s10, math.exp(xi_beta - xm_beta))) * 100
    return risk, xi_beta


# ==========================================
# 4. Main data-processing routine
# ==========================================
def process_data(base_path, item_path, output_path):
    print(f"--- Starting {MODEL_NAME} (precise missing-value output) ---")

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
        print("error: base file empty or unreadable.")
        return

    # 1. Pivot Item data to a wide table
    records = []
    if not item_df.empty:
        item_df['cat'] = item_df['res_itm_name'].apply(get_category)
        filtered = item_df.dropna(subset=['cat', 'res_value']).copy()

        for (app_no, pat_id), group in filtered.groupby(['app_no', 'pat_id']):
            row = {'app_no': str(app_no).strip(), 'pat_id': str(pat_id).strip()}
            data_pool = {k: None for k in FIELD_MAPPING.keys()}
            for _, item in group.iterrows():
                data_pool[item['cat']] = item['res_value']

            bp_sbp, bp_dbp = parse_combined_bp(data_pool.get('bp_combined'))
            sbp_single = clean_numeric(data_pool.get('sbp'))
            dbp_single = clean_numeric(data_pool.get('dbp'))

            row['sbp'] = sbp_single if pd.notna(sbp_single) else bp_sbp
            row['dbp'] = dbp_single if pd.notna(dbp_single) else bp_dbp

            bmi_val = clean_numeric(data_pool.get('BMI'))
            if pd.isna(bmi_val):
                weight, height = clean_numeric(data_pool.get('weight')), clean_numeric(data_pool.get('height'))
                if pd.notna(weight) and pd.notna(height) and height > 0:
                    bmi_val = weight / ((height / 100 if height > 3 else height) ** 2)

            row['bmi'] = bmi_val
            row['ldl'] = clean_numeric(data_pool.get('LDL_C'))
            row['hdl'] = clean_numeric(data_pool.get('HDL_C'))

            # Keep the extracted raw string for later checks
            row['smoke_raw'] = data_pool.get('smoke')
            row['diabetes'] = parse_bool_history(data_pool.get('has_diabetes'))
            row['dyslipidemia'] = parse_bool_history(data_pool.get('has_dyslipidemia'))
            row['preg_htn'] = parse_bool_history(data_pool.get('has_pregnancy_htn'))
            records.append(row)

    wide_items = pd.DataFrame(records)
    if wide_items.empty:
        wide_items = pd.DataFrame(
            columns=['app_no', 'pat_id', 'sbp', 'dbp', 'bmi', 'ldl', 'hdl', 'smoke_raw', 'diabetes', 'dyslipidemia',
                     'preg_htn'])

    # 2. Merge basic information
    base_df['app_no'] = base_df['app_no'].astype(str).str.strip()
    base_df['pat_id'] = base_df['pat_id'].astype(str).str.strip()
    final = pd.merge(base_df, wide_items, on=['app_no', 'pat_id'], how='left')

    # 3. Computation and formatting
    results = []
    for _, r in final.iterrows():
        age = clean_numeric(r.get('app_pat_age'))
        gender_val = parse_gender(r.get('pat_sex'))

        # ==========================================
        # [Core change]: dual-track past-history handling
        # ==========================================
        # 1. Capture the raw status (number, text or NaN)
        diab_raw = r.get('diabetes')
        dysl_raw = r.get('dyslipidemia')
        preg_raw = r.get('preg_htn')
        smoke_raw = r.get('smoke_raw')

        # 2. Compute safe values for the model (coerce missing to 0 or empty string)
        diab_calc = 0 if pd.isna(diab_raw) else diab_raw
        dysl_calc = 0 if pd.isna(dysl_raw) else dysl_raw
        preg_calc = 0 if pd.isna(preg_raw) else preg_raw
        smoke_calc = "" if pd.isna(smoke_raw) else smoke_raw

        # 3. Display text (generate a notice based on missing status)
        diab_out = "missing (defaulted to none)" if pd.isna(diab_raw) else ("yes" if diab_raw == 1 else "no")
        dysl_out = "missing (defaulted to none)" if pd.isna(dysl_raw) else ("yes" if dysl_raw == 1 else "no")
        smoke_out = "missing (defaulted to none)" if pd.isna(smoke_raw) or str(smoke_raw).strip() == "" else str(smoke_raw)

        if gender_val == 1:
            preg_out = "missing (defaulted to none)" if pd.isna(preg_raw) else ("yes" if preg_raw == 1 else "no")
        else:
            preg_out = "-"  # mark a dash for males

        # Call the computation function with safe calc values
        risk, _ = calculate_jaha_risk(
            age, gender_val, r.get('sbp'), r.get('dbp'), r.get('bmi'),
            r.get('ldl'), r.get('hdl'), smoke_calc,
            diab_calc, dysl_calc, preg_calc
        )

        # Risk logic evaluation
        if pd.isna(risk):
            display_score = "-"
            risk_level, risk_code = "-", "-"
            suggestion = "indicators missing or age out of range (18-39 with BP/BMI/lipids required)."
        else:
            display_score = f"{risk:.2f}%"
            if risk < 5:
                risk_level, risk_code = "low risk", "I10.2.1"
                suggestion = "10-year hypertension risk low; maintain a healthy lifestyle."
            elif risk < 15:
                risk_level, risk_code = "medium risk", "I10.2.2"
                suggestion = "medium risk; control weight and reduce sodium intake."
            else:
                risk_level, risk_code = "high risk", "I10.2.3"
                suggestion = "significant risk; monitor blood pressure regularly and consult a physician."

        results.append({
            'app_no': r.get('app_no'),
            'pat_id': r.get('pat_id'),
            'pat_name': r.get('pat_name', 'unknown'),
            'age': age if pd.notna(age) else "-",
            'sex': r.get('pat_sex', "-"),
            'BMI': f"{r.get('bmi'):.2f}" if pd.notna(r.get('bmi')) else "-",
            'systolic_bp': r.get('sbp') if pd.notna(r.get('sbp')) else "-",
            'diastolic_bp': r.get('dbp') if pd.notna(r.get('dbp')) else "-",
            'LDL-C': r.get('ldl') if pd.notna(r.get('ldl')) else "-",
            'HDL-C': r.get('hdl') if pd.notna(r.get('hdl')) else "-",

            # [New display column]: history display
            'smoking_history': smoke_out,
            'diabetes_history': diab_out,
            'dyslipidemia_history': dysl_out,
            'gestational_hypertension_history': preg_out,

            'predicted_score': display_score,
            'risk_level': risk_level,
            'code': risk_code,
            'gauge_reference': GAUGE_REF_STRING,
            'suggestion': suggestion,
            'model_name': MODEL_NAME
        })

    # 4. Export
    if results:
        res_df = pd.DataFrame(results)
        res_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f" assessment complete! Results saved to: {output_path}")
    else:
        print(" No assessment records generated.")


if __name__ == "__main__":
    process_data(
        "base.csv",
        "item_result.csv",
        "HTN predict for young adults_output.csv"
    )
