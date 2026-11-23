import pandas as pd
import numpy as np
import re
import os

# ==========================================
# 1. Field mapping (per NoSAS scoring criteria)
# ==========================================
FIELD_MAPPING = {
    'age': ['app_pat_age', 'age'],
    'gender': ['pat_sex', 'sex'],
    'BMI': ['(body mass index) BMI', 'bmi', 'BMI', 'body mass index'],
    'height': ['height'],
    'weight': ['weight'],
    'neck_circumference': ['neck_circumference', 'neck circumference', 'neck'],
    'snoring': ['snoring', 'snoring_during_sleep']
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


def parse_gender(val):
    s = str(val).lower()
    if pd.isna(val) or s.strip() == "": return np.nan
    if 'male' in s or 'male' in s or s == 'm' or s == '1': return 'male'
    if 'female' in s or 'female' in s or s == 'f' or s == '2': return 'female'
    return np.nan


def parse_boolean(val_list):
    """Strict snoring detection, first-hit principle"""
    if not val_list: return np.nan
    s = " ".join(val_list).lower()
    if s.strip() == "": return np.nan

    positive_keywords = ['yes', 'present', '1', 'true', 'y', 'positive', '+']
    negative_keywords = ['no', 'absent', '0', 'false', 'n', 'negative', '-']

    if any(kw in s for kw in positive_keywords): return True
    if any(kw in s for kw in negative_keywords): return False
    return np.nan


# ==========================================
# 3. NoSAS core scoring logic
# ==========================================
def calculate_nosas_score(neck_circumference, bmi, snoring, age, gender):
    # Strict validation: cannot evaluate when any item yields no numeric/boolean value
    if pd.isna(neck_circumference) or pd.isna(bmi) or pd.isna(snoring) or pd.isna(age) or pd.isna(gender):
        return "-"

    score = 0
    if neck_circumference > 40: score += 4
    if bmi >= 30:
        score += 5
    elif 25 <= bmi < 30:
        score += 3
    if snoring is True: score += 2
    if age > 55: score += 4
    if gender == 'male': score += 2

    return score


# ==========================================
# 4. Main data-processing routine
# ==========================================
def process_data(base_path, item_path, output_path):
    print("--- Starting NoSAS sleep-disordered breathing model (transparent full version) ---")

    def read_csv_safe(path):
        if not os.path.exists(path): return pd.DataFrame()
        for enc in ['gbk', 'utf-8-sig', 'utf-8']:
            try:
                return pd.read_csv(path, encoding=enc, low_memory=False)
            except:
                continue
        return pd.DataFrame()

    base_df = read_csv_safe(base_path)
    item_df = read_csv_safe(item_path)

    if base_df.empty:
        print("Error: base data file is empty or unreadable")
        return

    # Strip leading zeros that break joins
    base_df['app_no'] = base_df['app_no'].astype(str).str.strip().str.lstrip('0')
    base_df['pat_id'] = base_df['pat_id'].astype(str).str.strip().str.lstrip('0')

    # 1. Pivot to wide table
    records = []
    if not item_df.empty:
        item_df.columns = [c.strip() for c in item_df.columns]
        item_df['app_no'] = item_df['app_no'].astype(str).str.strip().str.lstrip('0')
        item_df['pat_id'] = item_df['pat_id'].astype(str).str.strip().str.lstrip('0')

        item_df['cat'] = item_df['res_itm_name'].apply(get_category)
        filtered = item_df.dropna(subset=['cat', 'res_value']).copy()

        # Helper: extract the first matching valid number (replaces averaging)
        def get_first_valid(val_list):
            for v in val_list:
                num = clean_numeric(v)
                if not pd.isna(num): return num
            return np.nan

        for (app_no, pat_id), group in filtered.groupby(['app_no', 'pat_id']):
            row = {'app_no': app_no, 'pat_id': pat_id}
            data_pool = {k: [] for k in FIELD_MAPPING.keys()}

            for _, item in group.iterrows():
                data_pool[item['cat']].append(str(item['res_value']))

            # Extract features
            row['bmi'] = get_first_valid(data_pool.get('BMI', []))

            # BMI fallback computation
            if pd.isna(row['bmi']):
                h = get_first_valid(data_pool.get('height', []))
                w = get_first_valid(data_pool.get('weight', []))
                if pd.notna(h) and pd.notna(w) and h > 0 and w > 0:
                    if h > 3: h = h / 100.0  # cm to m
                    row['bmi'] = w / (h * h)

            row['neck'] = get_first_valid(data_pool.get('neck_circumference', []))
            row['snoring'] = parse_boolean(data_pool.get('snoring', []))

            records.append(row)

    wide_items = pd.DataFrame(records) if records else pd.DataFrame(columns=['app_no', 'pat_id'])

    # 2. Merge basic information (left join, keep everyone)
    final = pd.merge(base_df, wide_items, on=['app_no', 'pat_id'], how='left')

    # 3. Scoring and result generation
    results = []
    GAUGE_REF = "0-7|8-17"

    # Format-output function
    def fmt_in(v, is_num=False, is_bool=False):
        if pd.isna(v) or v == "": return "missing"
        if is_bool: return "yes" if v else "no"
        return f"{v:.1f}" if is_num else str(v)

    for _, r in final.iterrows():
        age = clean_numeric(r.get('app_pat_age'))
        raw_gender = r.get('pat_sex', '')
        gender = parse_gender(raw_gender)

        bmi = r.get('bmi', np.nan)
        neck = r.get('neck', np.nan)
        snoring = r.get('snoring', np.nan)

        score = calculate_nosas_score(neck, bmi, snoring, age, gender)

        if score == "-":
            risk_level, suggestion, code, gauge = "-", "-", "-", "-"
            remark = "core indicators [age/sex/neck circumference/snoring/BMI] missing; cannot assess"
        else:
            gauge = GAUGE_REF
            remark = "data complete, computable"
            if score >= 8:
                risk_level, suggestion, code = "high risk", "recommend polysomnography (PSG)", "G47.1.3"
            else:
                risk_level, suggestion, code = "low risk", "current risk is low; keep observing", "G47.1.1"

        results.append({
            'pat_id': r.get('pat_id'),
            'app_no': r.get('app_no'),
            'pat_name': r.get('pat_name', 'unknown'),
            'model_name': 'NoSAS',

            # --- Raw indicator section ---
            'sex': raw_gender if raw_gender else "missing",
            'age': int(age) if not pd.isna(age) else "missing",
            'BMI': fmt_in(bmi, True),
            'neck_circumference (cm)': fmt_in(neck, True),
            'snoring': fmt_in(snoring, False, True),

            # --- Output result section ---
            'code': code,
            'predicted_score': int(score) if score != "-" else "-",
            'prediction_horizon': 'current' if score != "-" else "-",
            'risk_level': risk_level,
            'gauge_reference': gauge,
            'recommendation': suggestion,
            'assessment_note': remark
        })

    # 4. Export
    if results:
        res_df = pd.DataFrame(results)
        ordered_cols = [
            'pat_id', 'app_no', 'pat_name', 'model_name', 'code', 'gauge_reference',
            'predicted_score', 'risk_level', 'prediction_horizon', 'recommendation', 'assessment_note',
            'sex', 'age', 'BMI', 'neck_circumference (cm)', 'snoring'
        ]
        final_cols = [c for c in ordered_cols if c in res_df.columns]

        res_df[final_cols].to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"--- Assessment complete: {len(res_df)} records processed and exported ---")
    else:
        print("--- no valid data extracted ---")


if __name__ == "__main__":
    process_data("base.csv", "item_result.csv", "nosas_screening_result_perfect.csv")
