import pandas as pd
import numpy as np
import re
import math

# ==========================================
# 1. Field mapping
# ==========================================
FIELD_MAPPING = {
    'age': ['app_pat_age', 'age', 'nl'],
    'gender': ['pat_sex', 'sex', 'gender'],
    'BMI': ['(body mass index) BMI', 'bmi', 'BMI', 'body_mass_index'],
    'height': ['height'],
    'weight': ['weight'],
    'sbp': ['systolic_bp', 'systolic_bp (sbp)', 'sbp', 'sys_bp', 'blood_pressure', 'bp'],
    'dbp': ['diastolic_bp', 'diastolic_bp (dbp)', 'dbp', 'dia_bp', 'blood_pressure'],
    'smoke': ['smoking_history', 'smoking_status', 'smoke', 'smoker', 'is_smoker'],
    'parental_htn': ['parental_hypertension', 'hypertension_family_history', 'family_hist_htn'],
    'has_hypertension': ['past_history', 'disease_history', 'past_disease_history', 'past_medical_history']
}

# ==========================================
# 2. Model parameters
# ==========================================
INTERCEPT, SCALE = 22.94954, 0.87692
BETA_AGE, BETA_GENDER_FEMALE = -0.15641, -0.20293
BETA_SBP, BETA_DBP = -0.05933, -0.12847
BETA_SMOKE, BETA_PARENTAL_HTN = -0.19073, -0.16612
BETA_BMI, BETA_AGE_DBP_INTERACTION = -0.03388, 0.00162

MODEL_NAME = "Framingham Heart Study Weibull Hypertension Model"
GAUGE_REF_STRING = "0-5|5-10|10-100"


# ==========================================
# 3. Basic cleaning functions
# ==========================================
def normalize_str(s):
    if pd.isna(s): return ""
    return re.sub(r'[\(\)\（\）\[\]\【\】\{\}\s]', '', str(s)).lower()


def get_category(name):
    norm_name = normalize_str(name)
    for key, patterns in FIELD_MAPPING.items():
        for p in patterns:
            if normalize_str(p) == norm_name: return key
    return None


def clean_numeric(val):
    if pd.isna(val) or str(val).strip() == "": return np.nan
    try:
        match = re.search(r"[-+]?\d*\.\d+|\d+", str(val))
        return float(match.group()) if match else np.nan
    except:
        return np.nan


def parse_bp_value(val):
    s = str(val).strip()
    if not s or s.lower() == 'nan': return None, None
    if '/' in s:
        parts = s.split('/')
        if len(parts) >= 2:
            try:
                v1 = float(re.findall(r"[-+]?\d*\.\d+|\d+", parts[0])[0])
                v2 = float(re.findall(r"[-+]?\d*\.\d+|\d+", parts[1])[0])
                return max(v1, v2), min(v1, v2)
            except:
                pass
    try:
        match = re.search(r"[-+]?\d*\.\d+|\d+", s)
        if match: return float(match.group()), None
    except:
        pass
    return None, None


def parse_gender(val):
    s = str(val).strip()
    if any(k in s for k in ['female', 'Female', 'F', '0', '2', 'Women']): return 1
    if any(k in s for k in ['male', 'Male', 'M', '1', 'Men']): return 0
    return np.nan


def parse_smoke(val):
    s = str(val).strip()
    if any(k in s for k in ['yes', 'Yes', 'Y', '1', 'present', 'True']): return 1
    if any(k in s for k in ['no', 'No', 'N', '0', 'absent', 'False']): return 0
    return np.nan


def parse_parental_htn(val):
    num = clean_numeric(val)
    if not pd.isna(num) and 0 <= num <= 2: return int(num)
    s = str(val).strip()
    if any(k in s for k in ['bilateral', 'two', 'both']): return 2
    if any(k in s for k in ['one', 'single', 'father', 'mother']): return 1
    if any(k in s for k in ['absent', 'no', '0']): return 0
    return np.nan


# ==========================================
# 4. Core computation logic
# ==========================================
def calculate_hypertension_risk(age, gender_val, sbp, dbp, bmi, smoke, parental_htn):
    # Required-indicator check
    if any(pd.isna(x) for x in [age, gender_val, sbp, dbp, bmi]): return np.nan

    lp = (INTERCEPT + BETA_AGE * age + BETA_GENDER_FEMALE * gender_val +
          BETA_SBP * sbp + BETA_DBP * dbp + BETA_SMOKE * smoke +
          BETA_PARENTAL_HTN * parental_htn + BETA_BMI * bmi +
          BETA_AGE_DBP_INTERACTION * (age * dbp))

    try:
        exponent = (math.log(4) - lp) / SCALE
        survival = math.exp(-math.exp(exponent))
        return (1 - survival) * 100
    except:
        return np.nan


# ==========================================
# 5. Main data-processing routine
# ==========================================
def process_data(base_path, item_path, output_path):
    print(f"--- Starting {MODEL_NAME} assessment ---")

    def read_csv_safe(path):
        for enc in ['gbk', 'utf-8-sig', 'utf-8']:
            try:
                return pd.read_csv(path, encoding=enc, low_memory=False)
            except:
                continue
        return pd.DataFrame()

    base_df = read_csv_safe(base_path)
    item_df = read_csv_safe(item_path)

    # 1. Wide-table processing (first valid value only)
    records = []
    if not item_df.empty:
        item_df['cat'] = item_df['res_itm_name'].apply(get_category)
        filtered = item_df.dropna(subset=['cat', 'res_value']).copy()

        for (app_no, pat_id), group in filtered.groupby(['app_no', 'pat_id']):
            row = {'app_no': str(app_no).strip(), 'pat_id': str(pat_id).strip(),
                   'sbp': np.nan, 'dbp': np.nan, 'bmi': np.nan, 'height': np.nan,
                   'weight': np.nan, 'smoke': np.nan, 'parental_htn': np.nan, 'has_hypertension': np.nan}

            for _, item in group.iterrows():
                cat, val = item['cat'], str(item['res_value'])
                if cat in ['sbp', 'dbp'] and pd.isna(row[cat]):
                    v1, v2 = parse_bp_value(val)
                    if v1 and v2:
                        row['sbp'], row['dbp'] = v1, v2
                    elif v1:
                        row['sbp' if cat == 'sbp' else 'dbp'] = v1
                elif cat == 'BMI' and pd.isna(row['bmi']):
                    row['bmi'] = clean_numeric(val)
                elif cat == 'height' and pd.isna(row['height']):
                    row['height'] = clean_numeric(val)
                elif cat == 'weight' and pd.isna(row['weight']):
                    row['weight'] = clean_numeric(val)
                elif cat == 'smoke' and pd.isna(row['smoke']):
                    row['smoke'] = parse_smoke(val)
                elif cat == 'parental_htn' and pd.isna(row['parental_htn']):
                    row['parental_htn'] = parse_parental_htn(val)
                elif cat == 'has_hypertension':
                    row['has_hypertension'] = (row['has_hypertension'] + "；" + val) if pd.notna(
                        row['has_hypertension']) else val

            if pd.isna(row['bmi']) and pd.notna(row['height']) and pd.notna(row['weight']):
                h, w = row['height'], row['weight']
                if h > 0 and w > 0:
                    h = h / 100.0 if h > 30 else h
                    row['bmi'] = w / (h * h)
            records.append(row)

    wide_items = pd.DataFrame(records)
    base_df['app_no'] = base_df['app_no'].astype(str).str.strip()
    base_df['pat_id'] = base_df['pat_id'].astype(str).str.strip()

    # 2. Outer join to avoid dropping anyone
    final = pd.merge(wide_items, base_df, on=['app_no', 'pat_id'], how='outer') if not wide_items.empty else base_df

    # 3. Computation results
    results = []
    for _, r in final.iterrows():
        age = clean_numeric(r.get('app_pat_age')) or clean_numeric(r.get('age'))
        gender_val = parse_gender(r.get('pat_sex'))
        if pd.isna(gender_val): gender_val = parse_gender(r.get('gender'))

        sbp, dbp, bmi = r.get('sbp'), r.get('dbp'), r.get('bmi')

        # --- Smoking history and parental hypertension: missing defaults to none ---
        raw_smoke = r.get('smoke')
        smoke_calc = 0 if pd.isna(raw_smoke) else int(raw_smoke)
        smoke_display = "missing (defaulted to none)" if pd.isna(raw_smoke) else int(raw_smoke)

        raw_p_htn = r.get('parental_htn')
        p_htn_calc = 0 if pd.isna(raw_p_htn) else int(raw_p_htn)
        p_htn_display = "missing (defaulted to none)" if pd.isna(raw_p_htn) else int(raw_p_htn)

        # Past-history detection
        raw_history = str(r.get('has_hypertension', ''))
        temp_hist = re.sub(r'(denies|no|not found|none)[^,.;\n]*hypertension', '', raw_history)
        is_diagnosed = "hypertension" in temp_hist and "denies" not in temp_hist

        risk = calculate_hypertension_risk(age, gender_val, sbp, dbp, bmi, smoke_calc, p_htn_calc)

        # Risk determination
        if is_diagnosed:
            score_out, level_out, code_out, suggest_out = "-", "previously_diagnosed", "-", "-"
        elif pd.isna(risk):
            score_out, level_out, code_out, suggest_out = "-", "-", "-", "-"
        else:
            score_out = round(risk, 2)
            if risk < 5:
                level_out, code_out, suggest_out = "low risk", "I10.1.1", "low risk; maintain a healthy lifestyle."
            elif risk <= 10:
                level_out, code_out, suggest_out = "medium risk", "I10.1.2", "medium risk; monitor blood pressure regularly."
            else:
                level_out, code_out, suggest_out = "high risk", "I10.1.3", "high risk; start lifestyle intervention and see a physician promptly."

        results.append({
            'app_no': r.get('app_no', 'missing'), 'pat_id': r.get('pat_id', 'missing'), 'name': r.get('pat_name', 'unknown'),
            'age': int(age) if pd.notna(age) else "missing",
            'sex': ('male' if gender_val == 0 else 'female') if pd.notna(gender_val) else "missing",
            'BMI': round(bmi, 2) if pd.notna(bmi) else "missing",
            'systolic_bp': round(sbp, 1) if pd.notna(sbp) else "missing",
            'diastolic_bp': round(dbp, 1) if pd.notna(dbp) else "missing",
            'smoking_history': smoke_display,
            'parental_hypertension': p_htn_display,
            'past_history': raw_history if r.get('has_hypertension') else "missing",
            'predicted_score': score_out, 'risk_level': level_out, 'code': code_out,
            'gauge_reference': GAUGE_REF_STRING, 'recommendation': suggest_out, 'model_name': MODEL_NAME
        })

    pd.DataFrame(results).to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"--- Processing complete: {len(results)} records exported ---")


if __name__ == "__main__":
    # Use a model-specific file name so it does not overwrite the CHD Framingham model when both run.
    process_data("base.csv", "item_result.csv", "hypertension_framingham_output.csv")
