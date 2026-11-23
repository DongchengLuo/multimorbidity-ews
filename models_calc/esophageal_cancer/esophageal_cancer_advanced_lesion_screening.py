import pandas as pd
import numpy as np
import re
import math
import os

# ==========================================
# 1. Field mapping (see the precise field-matching reference table)
# ==========================================
FIELD_MAPPING = {
    'age': ['app_pat_age', 'age'],
    'sex': ['pat_sex', 'sex'],
    '(body mass index) BMI': ['(body mass index) BMI', 'bmi', 'BMI'],
    'height': ['height'],
    'weight': ['weight'],
    'smoking_history': ['smoking_history', 'smoking_status', 'smoker'],
    'dysphagia': ['dysphagia', 'difficulty_swallowing', 'food_obstruction_sensation'],
    'retrosternal_pain': ['retrosternal_pain', 'esophageal_pain', 'chest_pain']
}


# ==========================================
# 2. Basic cleaning and matching logic
# ==========================================
def normalize_str(s):
    if pd.isna(s):
        return ""
    return re.sub(r'[\(\)\（\）\[\]\【\】\{\}\s]', '', str(s)).lower()


def get_category(name):
    norm_name = normalize_str(name)
    for key, patterns in FIELD_MAPPING.items():
        for p in patterns:
            # Switch to exact match (equal after stripping brackets and spaces)
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


def parse_binary_condition(val_list):
    """Parse a binary indicator: NaN when no record exists, otherwise determine positivity"""
    if not val_list:
        # Return NaN (missing) when no record is found for the item
        return np.nan
    combined_str = " ".join([str(v) for v in val_list])
    if any(kw in combined_str for kw in ['yes', 'present', '1', 'smoking', 'pain', 'difficulty', 'positive']):
        return 1
    return 0


# ==========================================
# 3. Esophageal cancer model (Liu et al. 2020) core logic
# ==========================================
def calculate_esophageal_risk(age, bmi, smoking, dysphagia, retro_pain):
    """
    Model: esophageal high-grade lesion risk prediction (Liu et al. 2020)
    Formula: P = 1 / (1 + exp(-z))
    """
    intercept = -15.80
    c_age = 0.18
    c_smoke = 0.83
    c_bmi_low = 0.52
    c_dysphagia = 1.61
    c_retro_pain = 0.48

    bmi_flag = 1 if (bmi <= 22) else 0

    z = (intercept +
         (c_age * age) +
         (c_smoke * smoking) +
         (c_bmi_low * bmi_flag) +
         (c_dysphagia * dysphagia) +
         (c_retro_pain * retro_pain))

    prob = 1 / (1 + math.exp(-z))

    # Risk tiering and code mapping
    if prob >= 0.0156786:
        level, sug, code = "high risk", "strongly recommend endoscopy, consider NBI or iodine staining", "C15.2.3"
    elif prob >= 0.0026916:
        level, sug, code = "medium risk", "recommend endoscopy; close observation advised", "C15.2.2"
    else:
        level, sug, code = "low risk", "regular follow-up; decide on endoscopy based on clinical presentation", "C15.2.1"

    return prob, level, sug, code


# ==========================================
# 4. Main data-processing routine
# ==========================================
def process_esophageal_data(base_path, item_path, output_path):
    print("--- Starting esophageal high-grade lesion risk assessment ---")

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
        print("failed to read file; check the path.")
        return

    item_df['cat'] = item_df['res_itm_name'].apply(get_category)
    filtered = item_df.dropna(subset=['cat', 'res_value']).copy()

    records = []
    for (app_no, pat_id), group in filtered.groupby(['app_no', 'pat_id']):
        row = {'app_no': str(app_no).strip(), 'pat_id': str(pat_id).strip()}
        pool = {k: [] for k in FIELD_MAPPING.keys()}

        for _, item in group.iterrows():
            pool[item['cat']].append(str(item['res_value']))

        # Numeric and identifier extraction
        bmi_vals = pool.get('(body mass index) BMI', [])
        row['bmi'] = clean_numeric(bmi_vals[0]) if bmi_vals else np.nan

        if pd.isna(row['bmi']):
            h_vals = pool.get('height', [])
            w_vals = pool.get('weight', [])
            h = clean_numeric(h_vals[0]) if h_vals else np.nan
            w = clean_numeric(w_vals[0]) if w_vals else np.nan
            if h > 0 and w > 0:
                h_m = h / 100.0 if h > 3 else h
                row['bmi'] = w / (h_m * h_m)

        row['smoking'] = parse_binary_condition(pool.get('smoking_history', []))
        row['dysphagia'] = parse_binary_condition(pool.get('dysphagia', []))
        row['retro_pain'] = parse_binary_condition(pool.get('retrosternal_pain', []))

        records.append(row)

    wide_df = pd.DataFrame(records)

    if wide_df.empty:
        print("no valid indicator data extracted; ending.")
        return

    base_df['app_no'] = base_df['app_no'].astype(str).str.strip()
    base_df['pat_id'] = base_df['pat_id'].astype(str).str.strip()

    final_df = pd.merge(wide_df, base_df, on=['app_no', 'pat_id'], how='inner')

    results = []
    for _, r in final_df.iterrows():
        age = clean_numeric(r.get('age'))
        bmi = r.get('bmi')
        smoking = r.get('smoking')
        dysphagia = r.get('dysphagia')
        retro_pain = r.get('retro_pain')

        # Core change 1: detect any missing indicator (no longer skip with continue)
        has_missing = pd.isna(age) or pd.isna(bmi) or pd.isna(smoking) or pd.isna(dysphagia) or pd.isna(retro_pain)

        # Core change 2: handle results separately by missing status
        if has_missing:
            # Force "-" output when anything is missing
            prob_str = "-"
            level = "-"
            code = "-"
        else:
            # Data complete; compute probability
            prob, level, sug, code = calculate_esophageal_risk(
                age, bmi, smoking, dysphagia, retro_pain
            )
            prob_str = f"{prob:.4%}"

        # Core change 3: check basic fields separately; assign "missing" to avoid int()/float() errors
        fmt_age = "missing" if pd.isna(age) else int(age)
        fmt_bmi = "missing" if pd.isna(bmi) else f"{bmi:.2f}"
        fmt_smoking = "missing" if pd.isna(smoking) else int(smoking)
        fmt_dysphagia = "missing" if pd.isna(dysphagia) else int(dysphagia)
        fmt_retro_pain = "missing" if pd.isna(retro_pain) else int(retro_pain)

        results.append({
            'pat_id': r.get('pat_id', ''),
            'app_no': r.get('app_no', ''),
            'pat_name': r.get('pat_name', 'unknown'),
            'model_name': 'Esophageal Cancer and High-Grade Lesion Clinical Prediction Model (Liu et al. 2020)',
            'sex': r.get('pat_sex', ''),
            'age': fmt_age,
            'BMI': fmt_bmi,
            'smoking_history': fmt_smoking,
            'dysphagia': fmt_dysphagia,
            'retrosternal_pain': fmt_retro_pain,
            'risk_probability': prob_str,
            'risk_level': level,
            'code': code,
            'gauge_reference': '0%-0.269%|0.269%-1.568%|1.568%-100%'
        })

    if results:
        # Output results
        pd.DataFrame(results).to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"Assessment complete; exported {len(results)} results (including patients with missing data) to {output_path}")
    else:
        print("no valid results; check the base data or extraction logic.")


if __name__ == "__main__":
    process_esophageal_data(
        "base.csv",
        "item_result.csv",
        "esophageal_cancer_screening_result.csv"
    )
