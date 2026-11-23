import pandas as pd
import numpy as np
import re
import os

# ==========================================
# 1. Script metadata and configuration
# ==========================================
MODEL_NAME = "Chinese Modified Asia-Pacific Colorectal Screening Score (A-APCS)"
GAUGE_REF_STR = "0-1|1.5-2.5|3-7"

FIELD_MAPPING = {
    'age': ['app_pat_age', 'age'],
    'gender': ['pat_sex', 'sex'],
    'BMI': ['(body mass index) BMI', 'bmi'],
    'smoke': ['smoking_history', 'smoking_status', 'smoking'],
    'fdr_count': ['first_degree_relatives_colorectal_cancer', 'rectal_cancer_family_history', 'colorectal_cancer_family_history', 'family_history'],
    'height': ['height'],
    'weight': ['weight']
}


# ==========================================
# 2. Basic cleaning and matching utilities
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


def parse_boolean_text(text_list):
    if not text_list: return np.nan  # return NaN when empty so later logic can detect the default
    full_text = " ".join([str(t) for t in text_list]).lower()
    if any(kw in full_text for kw in ['yes', 'present', '1', 'true', 'smoking']):
        if "absent" in full_text or "no" in full_text or "not" in full_text:
            if "absent" in full_text and "present" not in full_text: return False
        return True
    return False


def parse_fdr_count(val_list):
    if not val_list: return np.nan  # return NaN when empty so later logic can detect the default
    text = " ".join([str(v) for v in val_list])
    nums = re.findall(r'\d+', text)
    if nums: return int(nums[0])
    if any(kw in text for kw in ['absent', 'no', 'none']): return 0
    if any(kw in text for kw in ['present', 'yes']): return 1
    return 0


# ==========================================
# 3. A-APCS core scoring logic
# ==========================================
def calculate_aapcs_score(age, gender_str, bmi, is_smoker, fdr_val):
    is_male = str(gender_str).strip() in ["male", "Male", "1"]
    score = 0.0

    # 1. Age scoring
    if age < 40:
        return None, "not_applicable", "age_lt_40", ""
    elif 40 <= age <= 49:
        score += 0
    elif 50 <= age <= 69:
        score += 1
    else:
        score += 2  # >= 70

    # 2. Sex
    score += 1 if is_male else 0
    # 3. Smoking
    score += 0.5 if is_smoker else 0
    # 4. BMI
    score += 1 if bmi >= 23.5 else 0
    # 5. Family history
    score += 2 if fdr_val >= 2 else (1 if fdr_val == 1 else 0)

    # Risk tiering
    if 0 <= score <= 1:
        return score, "low risk", "recommend_routine_screening", "C18.1.1"
    elif 1.5 <= score <= 2.5:
        return score, "medium risk", "recommend FIT or colonoscopy", "C18.1.2"
    else:
        return score, "high risk", "strongly recommend colonoscopy screening", "C18.1.3"


# ==========================================
# 4. Main data-processing routine
# ==========================================
def process_data(base_path, item_path, output_path):
    print(f"--- Starting {MODEL_NAME} screening task ---")

    def read_csv_safe(path):
        for enc in ['gbk', 'utf-8-sig', 'utf-8']:
            try:
                return pd.read_csv(path, encoding=enc, low_memory=False)
            except:
                continue
        return pd.DataFrame()

    base_df = read_csv_safe(base_path)
    item_df = read_csv_safe(item_path)
    if base_df.empty or item_df.empty: return

    # Wide-table preprocessing
    item_df['cat'] = item_df['res_itm_name'].apply(get_category)
    filtered = item_df.dropna(subset=['cat', 'res_value']).copy()

    records = []
    for (app_no, pat_id), group in filtered.groupby(['app_no', 'pat_id']):
        row = {'app_no': str(app_no).strip(), 'pat_id': str(pat_id).strip()}
        data_pool = {k: [] for k in FIELD_MAPPING.keys()}
        for _, item in group.iterrows():
            data_pool[item['cat']].append(str(item['res_value']))

        # ---- Save raw indicator string (first value only) ----
        row['raw_bmi'] = data_pool.get('BMI', [''])[0] if data_pool.get('BMI') else ""
        row['raw_height'] = data_pool.get('height', [''])[0] if data_pool.get('height') else ""
        row['raw_weight'] = data_pool.get('weight', [''])[0] if data_pool.get('weight') else ""
        row['raw_smoking_record'] = data_pool.get('smoke', [''])[0] if data_pool.get('smoke') else ""
        row['raw_family_history_record'] = data_pool.get('fdr_count', [''])[0] if data_pool.get('fdr_count') else ""

        # ---- Extract values for computation ----
        bmi_vals = [clean_numeric(v) for v in data_pool.get('BMI', []) if pd.notna(clean_numeric(v))]
        row['bmi'] = bmi_vals[0] if bmi_vals else np.nan

        if pd.isna(row['bmi']):
            h_vals = [clean_numeric(v) for v in data_pool.get('height', []) if pd.notna(clean_numeric(v))]
            w_vals = [clean_numeric(v) for v in data_pool.get('weight', []) if pd.notna(clean_numeric(v))]

            h = h_vals[0] if h_vals else np.nan
            w = w_vals[0] if w_vals else np.nan

            if h > 3: h /= 100.0
            if h > 0 and w > 0: row['bmi'] = w / (h * h)

        # Fetch the parsed raw status (may be NaN)
        row['smoke_raw_val'] = parse_boolean_text(data_pool.get('smoke', []))
        row['fdr_raw_val'] = parse_fdr_count(data_pool.get('fdr_count', []))
        records.append(row)

    wide_items = pd.DataFrame(records)
    base_df['app_no'] = base_df['app_no'].astype(str).str.strip()
    base_df['pat_id'] = base_df['pat_id'].astype(str).str.strip()

    if wide_items.empty: return
    final = pd.merge(wide_items, base_df, on=['app_no', 'pat_id'], how='inner')

    results = []
    skipped_count = 0

    for _, r in final.iterrows():
        age = clean_numeric(r.get('app_pat_age'))
        gender = r.get('pat_sex', '')
        bmi = r.get('bmi')

        # Core-indicator missing check (block only on required age and BMI)
        if pd.isna(age) or pd.isna(bmi):
            skipped_count += 1
            continue

        # Decide whether defaults apply
        is_smoke_default = pd.isna(r['smoke_raw_val'])
        is_fdr_default = pd.isna(r['fdr_raw_val'])

        # Values used for computation (defaults applied when missing)
        calc_smoke = False if is_smoke_default else r['smoke_raw_val']
        calc_fdr = 0 if is_fdr_default else r['fdr_raw_val']

        score, risk, rec, code = calculate_aapcs_score(
            age, gender, bmi, calc_smoke, calc_fdr
        )

        if score is None:
            skipped_count += 1
            continue

        # Format output string
        out_smoke = "no (default)" if is_smoke_default else ("yes" if calc_smoke else "no")
        out_fdr = "0 (default)" if is_fdr_default else str(int(calc_fdr))

        # Build the final output
        results.append({
            'app_no': r['app_no'],
            'pat_id': r['pat_id'],
            'model_name': MODEL_NAME,
            'name': r.get('pat_name', 'unknown'),
            'sex': gender,
            'age': int(age),
            # ----- Raw indicator section -----
            'BMI': r['raw_bmi'],
            'height': r['raw_height'],
            'weight': r['raw_weight'],
            'smoking': r['raw_smoking_record'],
            'family_history': r['raw_family_history_record'],
            # ----- Decision result section -----
            'bmi_for_calculation': round(bmi, 2),
            'smoker': out_smoke,
            'family_history_count': out_fdr,
            # ----- Risk assessment results -----
            'predicted_score': score,
            'prediction_horizon': 'current',
            'risk_level': risk,
            'code': code,
            'gauge_reference': GAUGE_REF_STR,
        })

    if results:
        res_df = pd.DataFrame(results)
        res_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\nAssessment complete: {len(res_df)} results (skipped {skipped_count} records with missing core indicators)")
        print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    process_data(
        "base.csv",
        "item_result.csv",
        "aapcs_screening_result.csv"
    )