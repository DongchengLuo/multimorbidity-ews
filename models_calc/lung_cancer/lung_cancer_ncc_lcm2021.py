import pandas as pd
import numpy as np
import math
import re

# ==========================================
# 1. Field-mapping configuration
# ==========================================
FIELD_MAPPING = {
    'age': ['app_pat_age', 'age'],
    'gender': ['pat_sex', 'sex'],
    'bmi': ['(body mass index) BMI', 'bmi', 'BMI'],
    'height': ['height'],
    'weight': ['weight'],
    'smoke': ['smoking_history', 'smoking_status'],
    'history': ['past_history', 'past_medical_history'],
    'resp_history': ['respiratory_disease_history', 'respiratory_history'],
    'cpd': ['daily_cigarettes', 'cigarettes_per_day', 'cigarettes/day', 'daily_smoking'],
    'yrs': ['smoking_years', 'smoking_duration'],
    'quit': ['years_since_quit', 'time_since_quit']
}

# ==========================================
# 2. NCC-LCm2021 model parameters
# ==========================================
S0_3YR = 0.999999998
THRESHOLD_SMOKER = 0.51
THRESHOLD_NON_SMOKER = 0.47

CHRONIC_KEYWORDS = ['asthma', 'bronchiectasis', 'chronic_bronchitis', 'emphysema', 'copd', 'chronic_obstructive_pulmonary_disease', 'COPD', 'pulmonary_tuberculosis', 'tuberculosis', 'silicosis', 'pneumoconiosis']


# ==========================================
# Utility functions
# ==========================================
def is_blank(val):
    return pd.isna(val) or str(val).strip() == ""


def normalize_str(s):
    if is_blank(s): return ""
    return re.sub(r'[\(\)\（\）\[\]\【\】\{\}\s]', '', str(s)).lower()


def get_category(name):
    norm_name = normalize_str(name)
    for key, patterns in FIELD_MAPPING.items():
        for p in patterns:
            if normalize_str(p) in norm_name: return key
    return None


def clean_numeric(val):
    if is_blank(val): return np.nan
    try:
        match = re.search(r"[-+]?\d*\.\d+|\d+", str(val))
        return float(match.group()) if match else np.nan
    except:
        return np.nan


def join_unique_text(values):
    vals = [str(v).strip() for v in values if not is_blank(v)]
    if not vals: return np.nan
    return " ".join(sorted(set(vals)))


def format_text(val):
    return "missing" if is_blank(val) else str(val).strip()


def format_numeric(val, digits=None, as_int=False):
    if pd.isna(val): return "missing"
    if as_int:
        try:
            return int(float(val))
        except:
            return val
    if digits is not None:
        try:
            return round(float(val), digits)
        except:
            return val
    return val


def parse_smoke_status(smoke_status):
    if is_blank(smoke_status): return None
    s = str(smoke_status).strip().replace(" ", "")
    negative_patterns = ['never_smoker', 'non-smoker', 'no_smoking_history', 'denies_smoking', 'never_smoked']
    positive_patterns = ['current_smoker', 'smoking', 'quit_smoking', 'former_smoker', 'smoking_history']

    if any(p in s for p in negative_patterns): return False
    if any(p in s for p in positive_patterns): return True
    if s in ['no', 'absent', '0', 'false', 'False']: return False
    if s in ['yes', 'present', '1', 'true', 'True']: return True
    return None


def is_female(gender):
    if is_blank(gender): return None
    s = str(gender).strip().upper()
    if 'female' in s or s == 'F' or 'FEMALE' in s or s == '2': return True
    if 'male' in s or s == 'M' or 'MALE' in s or s == '1': return False
    return None


def has_lung_cancer_history(history_str):
    if is_blank(history_str): return False
    s = str(history_str)
    if 'lung_cancer' in s:
        negations = ['denies', 'absent', 'not found', 'exclude', 'normal', 'unknown']
        pos = s.find('lung_cancer')
        prefix = s[max(0, pos - 5):pos]
        if any(neg in prefix for neg in negations): return False
        return True
    return False


# === Core change: output "missing (defaulted to none)" when absent ===
def extract_resp_history_display(history_str):
    if is_blank(history_str): return "missing (defaulted to none)"
    s = str(history_str)
    return "present" if any(x in s for x in CHRONIC_KEYWORDS) else "absent"


def calculate_lung_cancer_risk(age, bmi, gender, smoke_status, history, cpd_raw, yrs_raw, quit_raw):
    try:
        smoke_flag = parse_smoke_status(smoke_status)
        if pd.isna(age) or pd.isna(bmi) or smoke_flag is None:
            return np.nan, None, smoke_flag, "core indicators (age/BMI/smoking status) missing"

        if age < 40 or age > 74:
            return np.nan, None, smoke_flag, f"skipped: age ({age} years) outside range (40-74)"

        if has_lung_cancer_history(history):
            return np.nan, None, smoke_flag, "excluded: past history indicates lung cancer"

        if smoke_flag is False:
            sex_flag = is_female(gender)
            # === Core change: removed the past-history blocking rule; default to no history and continue ===
            if sex_flag is None:
                return np.nan, THRESHOLD_NON_SMOKER, smoke_flag, "non-smoker model missing [sex]"

            h_str = "" if is_blank(history) else str(history)
            fam_hist = 1 if ('lung_cancer' in h_str and any(
                x in h_str for x in ['family', 'relative', 'father', 'mother', 'brother', 'sister'])) else 0
            chronic_dis = 1 if any(x in h_str for x in CHRONIC_KEYWORDS) else 0
            sex_val = 1 if sex_flag else 0

            lp = 4.37 * math.log(age) - 0.05 * bmi - 0.40 * sex_val + 0.35 * fam_hist + 0.29 * chronic_dis
            prob = (1 - math.pow(S0_3YR, math.exp(lp))) * 100
            return prob, THRESHOLD_NON_SMOKER, smoke_flag, "data complete, computable (non-smoker model)"

        else:
            cpd = clean_numeric(cpd_raw)
            yrs = clean_numeric(yrs_raw)
            if pd.isna(cpd) or pd.isna(yrs):
                return np.nan, THRESHOLD_SMOKER, smoke_flag, "smoker model missing [daily cigarettes] or [smoking years]"

            cpd_map = {10: 0.0, 20: 0.44, 30: 0.32, 40: 0.61, 999: 0.60}
            cpd_coeff = next(v for k, v in sorted(cpd_map.items()) if cpd <= k)

            yrs_map = {10: 0.0, 20: 0.12, 30: 0.05, 40: 0.37, 999: 0.60}
            yrs_coeff = next(v for k, v in sorted(yrs_map.items()) if yrs <= k)

            quit_yrs = clean_numeric(quit_raw)
            quit_yrs_val = 0 if pd.isna(quit_yrs) else quit_yrs

            lp = 4.95 * math.log(age) - 1.00 * math.log(bmi) + cpd_coeff + yrs_coeff - 0.02 * quit_yrs_val
            prob = (1 - math.pow(S0_3YR, math.exp(lp))) * 100
            return prob, THRESHOLD_SMOKER, smoke_flag, "data complete, computable (smoker model)"

    except:
        return np.nan, None, None, "computation anomaly"


# ==========================================
# Main data-processing logic
# ==========================================
def process_data(base_path, item_path, output_path):
    print("--- Starting NCC-LCm2021 lung cancer risk prediction ---")

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
        print("error: cannot read base file or it is empty.")
        return

    # 1) Wide-table extraction
    item_df['cat'] = item_df['res_itm_name'].apply(get_category)
    filtered = item_df.dropna(subset=['cat']).copy()

    records = []
    for (app_no, pat_id), group in filtered.groupby(['app_no', 'pat_id']):
        row = {'app_no': str(app_no).strip(), 'pat_id': str(pat_id).strip()}
        bmi_l, h_l, w_l, s_l, hist_l, resp_l, cpd_l, yrs_l, quit_l = [], [], [], [], [], [], [], [], []

        for _, item in group.iterrows():
            cat = item['cat']
            val = item['res_value']
            if cat == 'bmi':
                bmi_l.append(clean_numeric(val))
            elif cat == 'height':
                h_l.append(clean_numeric(val))
            elif cat == 'weight':
                w_l.append(clean_numeric(val))
            elif cat == 'smoke':
                s_l.append(val)
            elif cat == 'history':
                hist_l.append(val)
            elif cat == 'resp_history':
                resp_l.append(val)
            elif cat == 'cpd':
                cpd_l.append(val)
            elif cat == 'yrs':
                yrs_l.append(val)
            elif cat == 'quit':
                quit_l.append(val)

        row['bmi_orig'] = bmi_l[0] if bmi_l else np.nan
        row['height'] = h_l[0] if h_l else np.nan
        row['weight'] = w_l[0] if w_l else np.nan
        row['smoke'] = join_unique_text(s_l)
        row['history'] = join_unique_text(hist_l + resp_l)
        row['cpd_raw'] = join_unique_text(cpd_l)
        row['yrs_raw'] = join_unique_text(yrs_l)
        row['quit_raw'] = join_unique_text(quit_l)

        records.append(row)

    wide_items = pd.DataFrame(records) if records else pd.DataFrame(
        columns=['app_no', 'pat_id', 'bmi_orig', 'height', 'weight', 'smoke', 'history', 'cpd_raw', 'yrs_raw',
                 'quit_raw'])

    # 2) BMI completion
    def calc_bmi_fallback(r):
        if not pd.isna(r.get('bmi_orig', np.nan)): return r['bmi_orig']
        if not pd.isna(r.get('height', np.nan)) and not pd.isna(r.get('weight', np.nan)) and r['height'] > 0:
            h_m = r['height'] / 100 if r['height'] > 3 else r['height']
            return r['weight'] / (h_m ** 2)
        return np.nan

    if not wide_items.empty:
        wide_items['bmi'] = wide_items.apply(calc_bmi_fallback, axis=1)
    else:
        wide_items['bmi'] = np.nan

    # 3) Merge the base table (left join, keep leading zeros)
    base_df['app_no'] = base_df['app_no'].astype(str).str.strip()
    base_df['pat_id'] = base_df['pat_id'].astype(str).str.strip()
    final = pd.merge(base_df, wide_items, on=['app_no', 'pat_id'], how='left')

    # 4) Full output
    results = []
    for _, r in final.iterrows():
        age_val = clean_numeric(r.get('app_pat_age'))
        bmi_val = r.get('bmi', np.nan)
        sex_raw = r.get('pat_sex', np.nan)
        smoke_raw = r.get('smoke', np.nan)
        history_raw = r.get('history', np.nan)
        cpd_raw = r.get('cpd_raw', np.nan)
        yrs_raw = r.get('yrs_raw', np.nan)
        quit_raw = r.get('quit_raw', np.nan)

        prob, threshold, smoke_flag_used, remark = calculate_lung_cancer_risk(
            age_val, bmi_val, sex_raw, smoke_raw, history_raw, cpd_raw, yrs_raw, quit_raw
        )

        crowd_tag = "-" if smoke_flag_used is None else ("ever/current smoker" if smoke_flag_used else "non_smoker")
        risk_code, risk_prob, predict_time, risk_layer, gauge_ref_str = "-", "-", "-", "-", "-"

        if pd.notna(prob) and threshold is not None:
            mid_bound = round(threshold * 0.6, 3)
            high_bound = round(threshold, 3)
            gauge_ref_str = f"0-{mid_bound}|{mid_bound}-{high_bound}|{high_bound}-100"

            if prob >= threshold:
                risk_layer, risk_code = "high risk", "C34.1.3"
            elif prob >= (threshold * 0.6):
                risk_layer, risk_code = "medium risk", "C34.1.2"
            else:
                risk_layer, risk_code = "low risk", "C34.311"

            risk_prob = f"{prob:.4f}%"
            predict_time = "next 3 years"

        quit_val = clean_numeric(quit_raw)
        quit_display = "missing" if pd.isna(quit_val) else (int(quit_val) if float(quit_val).is_integer() else quit_val)

        results.append({
            'pat_id': format_text(r.get('pat_id')),
            'app_no': format_text(r.get('app_no')),
            'pat_name': format_text(r.get('pat_name')),
            'model_name': 'Lung Cancer NCC-LCm2021 models',
            'code': risk_code,
            'gauge_reference': gauge_ref_str,
            'risk_probability': risk_prob,
            'risk_level': risk_layer,
            'prediction_horizon': predict_time,
            'assessment_note': remark,
            'population_category': crowd_tag,
            'sex': format_text(sex_raw),
            'age': format_numeric(age_val, as_int=True),
            'BMI': format_numeric(bmi_val, digits=2),
            'smoking_record': format_text(smoke_raw),
            'daily_cigarettes': format_numeric(clean_numeric(cpd_raw), as_int=True),
            'smoking_years': format_numeric(clean_numeric(yrs_raw), as_int=True),
            'years_since_quit': quit_display,
            'respiratory_history': extract_resp_history_display(history_raw)  # use the new function returning "missing (defaulted to none)"
        })

    if results:
        res_df = pd.DataFrame(results)
        res_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"--- Task complete! {len(res_df)} results exported (with transparent notes) ---")


if __name__ == "__main__":
    process_data("base.csv", "item_result.csv", "lung_cancer_risk_perfect.csv")
