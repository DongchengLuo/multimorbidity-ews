import pandas as pd
import numpy as np
import re
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. Field mapping (separate smoking amount/years removed)
# ==========================================
FIELD_MAPPING = {
    # --- Required fields ---
    'age': ['app_pat_age', 'age'],
    'gender': ['pat_sex', 'sex'],
    'BMI': ['(body mass index) BMI', 'bmi', 'BMI'],
    'height': ['height'],
    'weight': ['weight'],
    'smoke': ['smoking_history', 'smoking_status'],  # keep only this combined field

    # --- Supplementary generic keywords ---
    'frequent_cough': ['frequent_cough', 'chronic_cough'],
    'chronic_respiratory_disease': ['chronic_respiratory_disease_history', 'copd_history'],

    # --- COPD-SQ specific fields ---
    'sob_level': ['shortness_of_breath_status', 'dyspnea_severity', 'exertional_dyspnea', 'shortness_of_breath'],
    'bio_fuel': ['biofuel', 'cooking_fuel', 'firewood_cooking'],
    'copd_family_history': ['copd_family_history', 'respiratory_disease_family_history']
}


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
            if normalize_str(p) in norm_name: return key
    return None


def clean_numeric(val):
    if pd.isna(val) or str(val).strip() == "": return np.nan
    try:
        match = re.search(r"[-+]?\d*\.\d+|\d+", str(val))
        return float(match.group()) if match else np.nan
    except:
        return np.nan


def parse_sob_level(val):
    s = str(val)
    if pd.isna(val) or s.strip() == "": return None
    if "flat_ground" in s and ("normal" in s or "slow" in s): return 2
    if "slope" in s or "brisk_walking" in s or "fast_walking" in s: return 1
    num = clean_numeric(val)
    if not pd.isna(num):
        if num >= 2: return 2
        if num == 1: return 1
    return 0


def parse_binary_feature(data_list):
    if not data_list:
        return None
    text = " ".join(data_list)
    if any(kw in text for kw in ['yes', 'present', '1']):
        return 1
    return 0


# [Revised function] parses pack-years and output text from a single smoking-history string
def parse_pack_years_from_text(text):
    if not text or str(text).strip() == "":
        return None, None

    # 1. Try to extract "X pack-years" directly (e.g. "20 pack-years")
    pack_match = re.search(r'(\d+(?:\.\d+)?)\s*pack-years', text)
    if pack_match:
        val = float(pack_match.group(1))
        return val, f"{val:.1f}"

    # 2. Try to extract both "X cigarettes" and "Y years" (e.g. "20/day for 10 years")
    cigs_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:cigarettes|sticks)', text)
    yrs_match = re.search(r'(\d+(?:\.\d+)?)\s*years', text)

    if cigs_match and yrs_match:
        cigs = float(cigs_match.group(1))
        yrs = float(yrs_match.group(1))
        val = (cigs / 20.0) * yrs
        return val, f"{val:.1f}"

    # 3. [Added logic] for people with only smoking years, assume one pack per day
    if yrs_match and not cigs_match:
        yrs = float(yrs_match.group(1))
        val = yrs * 1.0 # assume one pack per day
        return val, "missing; defaulted to 1 pack/day"

    # 4. If no numeric value is found but explicit non-smoking wording is present
    if any(kw in text for kw in ["absent", "no", "never", "never_smoked", "non_smoker", "quit"]):
        return 0.0, "0.0"

    # 5. [Added logic] smokers with unknown pack-years are scored 2 points directly
    # When relevant wording exists, assign a value in the 1-14 pack-year range (e.g. 1.0) so scoring triggers the +2 points
    if any(kw in text for kw in ["present", "smoke", "cigarette", "often", "occasionally", "small_amount", "pack"]):
        return 1.0, "missing value; defaulted to 1-14 pack-years"

    # 6. Unparseable content is marked as missing
    return None, None


# ==========================================
# 3. COPD-SQ core scoring logic
# ==========================================
def calculate_copd_sq_score(age, bmi, pack_years, cough, sob_level, bio_fuel, family_history):
    score = 0
    if 35 <= age <= 49:
        score += 0
    elif 50 <= age <= 59:
        score += 4
    elif 60 <= age <= 69:
        score += 8
    elif age >= 70:
        score += 11

    if 1 <= pack_years < 15:
        score += 2
    elif 15 <= pack_years < 30:
        score += 4
    elif pack_years >= 30:
        score += 5

    if not pd.isna(bmi):
        if bmi < 18.5:
            score += 7
        elif 18.5 <= bmi <= 23.9:
            score += 4
        elif 24 <= bmi <= 27.9:
            score += 1

    if cough: score += 5
    if sob_level == 1:
        score += 3
    elif sob_level == 2:
        score += 6
    if bio_fuel: score += 1
    if family_history: score += 3

    return score


# ==========================================
# 4. Main data-processing routine
# ==========================================
def process_data(base_path, item_path, output_path):
    print("--- Starting COPD-SQ screening model ---")

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
        print(f"data file unreadable or empty: Base={len(base_df)}, Item={len(item_df)}")
        return

    base_df['app_no'] = base_df['app_no'].astype(str).str.strip()
    base_df['pat_id'] = base_df['pat_id'].astype(str).str.strip()
    item_df['app_no'] = item_df['app_no'].astype(str).str.strip()
    item_df['pat_id'] = item_df['pat_id'].astype(str).str.strip()

    item_df['cat'] = item_df['res_itm_name'].apply(get_category)
    filtered = item_df.dropna(subset=['cat', 'res_value']).copy()

    records = []

    for _, base_row in base_df.iterrows():
        app_no = str(base_row['app_no']).strip()
        pat_id = str(base_row['pat_id']).strip()
        row = {
            'app_no': app_no,
            'pat_id': pat_id,
            'pat_name': base_row.get('pat_name', 'unknown'),
            'pat_sex': base_row.get('pat_sex', ''),
        }

        age_val = base_row['app_pat_age'] if 'app_pat_age' in base_df.columns else np.nan
        row['app_pat_age'] = clean_numeric(age_val)

        group = filtered[(filtered['app_no'] == app_no) & (filtered['pat_id'] == pat_id)]
        data_pool = {k: [] for k in FIELD_MAPPING.keys()}
        for _, item in group.iterrows():
            data_pool[item['cat']].append(str(item['res_value']))

        # 1. BMI extraction
        bmi_vals = [clean_numeric(v) for v in data_pool.get('BMI', []) if not pd.isna(clean_numeric(v))]
        row['bmi'] = bmi_vals[0] if bmi_vals else np.nan

        if pd.isna(row['bmi']):
            h_vals = [clean_numeric(v) for v in data_pool.get('height', []) if not pd.isna(clean_numeric(v))]
            h = h_vals[0] if h_vals else np.nan
            w_vals = [clean_numeric(v) for v in data_pool.get('weight', []) if not pd.isna(clean_numeric(v))]
            w = w_vals[0] if w_vals else np.nan
            if not pd.isna(h) and not pd.isna(w) and h > 0:
                if h >= 100: h = h / 100.0
                row['bmi'] = w / (h * h)

        # 2. Extract smoking data (parse the full smoking-history string) [changed]
        smoke_hist = " ".join(data_pool.get('smoke', []))
        pack_val, pack_msg = parse_pack_years_from_text(smoke_hist)
        row['pack_years'] = pack_val
        row['pack_msg'] = pack_msg

        # 3. Other indicator extraction
        row['cough'] = parse_binary_feature(data_pool.get('frequent_cough', []))
        row['sob_level'] = parse_sob_level(" ".join(data_pool.get('sob_level', [])))
        row['bio_fuel'] = parse_binary_feature(data_pool.get('bio_fuel', []))
        row['family_history'] = parse_binary_feature(data_pool.get('copd_family_history', []))

        records.append(row)

    final = pd.DataFrame(records)

    # --- 3. Scoring and result generation ---
    results = []
    GAUGE_REF = "0-15|16-40"

    def format_output(val, true_str, false_str, missing_str):
        if val is None: return missing_str
        return true_str if val == 1 else false_str

    for _, r in final.iterrows():
        age = r.get('app_pat_age')

        if pd.isna(age):
            score = "-"
            risk_level = "cannot assess (missing age)"
            suggestion = "supplementary_basic_info"
            risk_code = "-"
        else:
            bmi = r.get('bmi', np.nan)

            pack_years_calc = r.get('pack_years') if r.get('pack_years') is not None else 0.0
            cough_calc = bool(r['cough']) if r['cough'] is not None else False
            sob_level_calc = r['sob_level'] if r['sob_level'] is not None else 0
            bio_fuel_calc = bool(r['bio_fuel']) if r['bio_fuel'] is not None else False
            family_history_calc = bool(r['family_history']) if r['family_history'] is not None else False

            score = calculate_copd_sq_score(
                age, bmi, pack_years_calc,
                cough_calc, sob_level_calc, bio_fuel_calc, family_history_calc
            )

            if score >= 16:
                risk_level = "high risk"
                suggestion = "recommend pulmonary function test"
                risk_code = "J44.3.3"
            else:
                risk_level = "low risk"
                suggestion = "regular follow-up"
                risk_code = "J44.3.1"

        bmi_out = r.get('bmi', np.nan)

        results.append({
            'pat_name': r.get('pat_name', 'unknown'),
            'pat_id': r.get('pat_id', ''),
            'app_no': r.get('app_no', ''),
            'sex': r.get('pat_sex', ''),
            'age': int(age) if not pd.isna(age) else "missing",
            'BMI': f"{bmi_out:.1f}" if not pd.isna(bmi_out) else "-",
            'pack_years': r.get('pack_msg') if r.get('pack_msg') is not None else "missing; defaulted to no smoking history", # [changed] use pack_msg
            'frequent_cough': format_output(r['cough'], 'yes', 'no', 'missing (defaulted to no)'),
            'dyspnea_grade': r['sob_level'] if r['sob_level'] is not None else "missing; defaulted to 0",
            'biofuel': format_output(r['bio_fuel'], 'yes', 'no', 'missing (defaulted to no)'),
            'copd_family_history': format_output(r['family_history'], 'present', 'absent', 'missing (defaulted to none)'),
            'predicted_score': score,
            'risk_level': risk_level,
            'prediction_horizon': 'current',
            'model_name': 'COPD-SQ screening model',
            'code': risk_code,
            'gauge_reference': GAUGE_REF,
            'recommendation': suggestion
        })

    if results:
        res_df = pd.DataFrame(results)
        res_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        valid_count = len(res_df[res_df['predicted_score'] != '-'])
        print(f"--- Assessment complete: {len(res_df)} records processed ---")
        print(f"--- Valid scored records: {valid_count} ---")
        print(f"--- Results exported to {output_path} ---")
    else:
        print("--- Assessment complete, but no valid results were generated ---")


if __name__ == "__main__":
    process_data(
        "base.csv",
        "item_result.csv",
        "copd_sq_screening_result.csv"
    )