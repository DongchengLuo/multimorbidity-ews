import pandas as pd
import numpy as np
import re
import os

# ==========================================
# 1. Field mapping (for extracting res_itm_name from the long table)
# ==========================================
FIELD_MAPPING = {
    'BMI': ['(body mass index) BMI', 'bmi', 'BMI'],
    'height': ['height'],
    'weight': ['weight'],
    'neck_circumference': ['neck_circumference'],
    'snoring': ['snoring', 'snoring_sound'],
    'has_hypertension': ['past_history', 'past_medical_history', 'has_hypertension', 'hypertension', 'high_bp'],
    'tiredness': ['tiredness', 'fatigue', 'weakness', 'sleepiness', 'tired', 'daytime_sleepiness'],
    'observed': ['observed', 'apnea', 'breathing_pause', 'breath_holding', 'stop_breathing']
}


# ==========================================
# 2. Basic cleaning and matching logic
# ==========================================
def normalize_str(s):
    if pd.isna(s): return ""
    return re.sub(r'[\(\)\（\）\[\]\【\】\{\}\s\_\-]', '', str(s)).lower()


def get_category(name):
    norm_name = normalize_str(name)
    for key, patterns in FIELD_MAPPING.items():
        for p in patterns:
            if normalize_str(p) in norm_name: return key
    return None


def clean_numeric(val):
    if pd.isna(val) or val is None or str(val).strip() == "": return np.nan
    try:
        match = re.search(r"[-+]?\d*\.\d+|\d+", str(val))
        return float(match.group()) if match else np.nan
    except:
        return np.nan


def clean_boolean_strict(val_list, keyword_check=None):
    """Strict boolean extraction, first-hit principle"""
    if not val_list: return np.nan

    for val in val_list:
        if pd.isna(val) or str(val).strip() == "": continue
        s = str(val).strip().lower()

        # If long text such as past history, check for the disease without negation
        if keyword_check:
            if keyword_check not in s: continue
            negations = ['absent', 'denies', 'not seen', 'exclude', 'no', 'not found']
            pattern = rf"({'|'.join(negations)})[^\u4e00-\u9fa5]{{0,3}}{keyword_check}"
            if re.search(pattern, s): return 0
            return 1

        # Routine boolean checks
        true_keywords = ['yes', 'y', 'true', '1', 'present', 'meets', 'high']
        false_keywords = ['no', 'n', 'false', '0', 'absent', 'not_met', 'low']

        if any(kw == s or (kw in s and len(s) < 10) for kw in true_keywords): return 1
        if any(kw == s or (kw in s and len(s) < 10) for kw in false_keywords): return 0

    return np.nan


def clean_gender(val):
    if pd.isna(val) or str(val).strip() == "": return np.nan
    s = str(val).strip().lower()
    if any(k in s for k in ['male', 'man', '1']): return 1
    if any(k in s for k in ['female', 'woman', '2']): return 0
    return np.nan


# ==========================================
# 3. STOP-Bang core scoring logic
# ==========================================
def calculate_stop_bang_score(snoring, tiredness, observed, has_hypertension, bmi, age, neck, gender_score):
    if any(pd.isna(x) for x in [snoring, tiredness, observed, has_hypertension, bmi, age, neck, gender_score]):
        return "-", "-"

    stop_score = 0
    bang_score = 0
    debug_info = []

    # S, T, O, P
    if snoring == 1: debug_info.append("snoring"); stop_score += 1
    if tiredness == 1: debug_info.append("tiredness"); stop_score += 1
    if observed == 1: debug_info.append("apnea"); stop_score += 1
    if has_hypertension == 1: debug_info.append("hypertension"); stop_score += 1

    # B, A, N, G
    if bmi > 35: debug_info.append("BMI>35"); bang_score += 1
    if age > 50: debug_info.append("age_gt_50"); bang_score += 1
    if neck > 40: debug_info.append("neck_circumference_gt_40cm"); bang_score += 1
    if gender_score == 1: debug_info.append("male"); bang_score += 1

    total_stop_bang = stop_score + bang_score
    detail_str = "|".join(debug_info) if debug_info else "no_high_risk_items"

    return total_stop_bang, detail_str


# ==========================================
# 4. Main data-processing routine
# ==========================================
def process_data(base_path, item_path, output_path):
    print("--- Starting STOP-Bang model (transparent full version) ---")

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

    # Strip leading zeros from IDs to avoid join failures
    base_df['app_no'] = base_df['app_no'].astype(str).str.strip().str.lstrip('0')
    base_df['pat_id'] = base_df['pat_id'].astype(str).str.strip().str.lstrip('0')

    # 1. Pivot to wide table
    records = []
    if not item_df.empty:
        item_df['app_no'] = item_df['app_no'].astype(str).str.strip().str.lstrip('0')
        item_df['pat_id'] = item_df['pat_id'].astype(str).str.strip().str.lstrip('0')

        item_df['cat'] = item_df['res_itm_name'].apply(get_category)
        filtered = item_df.dropna(subset=['cat', 'res_value']).copy()

        def get_first_num(val_list):
            for v in val_list:
                num = clean_numeric(v)
                if not pd.isna(num): return num
            return np.nan

        for (app_no, pat_id), group in filtered.groupby(['app_no', 'pat_id']):
            row = {'app_no': app_no, 'pat_id': pat_id}
            data_pool = {k: [] for k in FIELD_MAPPING.keys()}

            for _, item in group.iterrows():
                data_pool[item['cat']].append(str(item['res_value']))

            # Extract numeric values
            row['bmi'] = get_first_num(data_pool.get('BMI', []))
            if pd.isna(row['bmi']):
                h = get_first_num(data_pool.get('height', []))
                w = get_first_num(data_pool.get('weight', []))
                if pd.notna(h) and pd.notna(w) and h > 0 and w > 0:
                    if h > 3: h = h / 100.0
                    row['bmi'] = w / (h * h)

            row['neck'] = get_first_num(data_pool.get('neck_circumference', []))

            # Extract booleans
            row['snoring'] = clean_boolean_strict(data_pool.get('snoring', []))
            row['tiredness'] = clean_boolean_strict(data_pool.get('tiredness', []))
            row['observed'] = clean_boolean_strict(data_pool.get('observed', []))

            # Extract hypertension (also from past history)
            row['has_hypertension'] = clean_boolean_strict(data_pool.get('has_hypertension', []),
                                                           keyword_check='hypertension')

            records.append(row)

    wide_items = pd.DataFrame(records) if records else pd.DataFrame(columns=['app_no', 'pat_id'])

    # 2. Merge basic information (left join, keep all)
    final = pd.merge(base_df, wide_items, on=['app_no', 'pat_id'], how='left')

    # 3. Scoring and result generation
    results = []
    GAUGE_REF_STR = "0-2|3-4|5-8"

    # Helper to format output consistently
    def fmt_in(v, is_num=False, is_bool=False):
        if pd.isna(v) or v == "": return "missing"
        if is_bool: return "yes" if v == 1 else "no"
        return f"{v:.1f}" if is_num else str(v)

    for _, r in final.iterrows():
        age = clean_numeric(r.get('app_pat_age'))
        gender_score = clean_gender(r.get('pat_sex'))
        bmi = r.get('bmi', np.nan)
        neck = r.get('neck', np.nan)

        total_score, details = calculate_stop_bang_score(
            snoring=r.get('snoring'),
            tiredness=r.get('tiredness'),
            observed=r.get('observed'),
            has_hypertension=r.get('has_hypertension'),
            bmi=bmi,
            age=age,
            neck=neck,
            gender_score=gender_score
        )

        # 4. Risk grading and assessment notes
        if total_score == "-":
            risk_level, risk_code, suggestion = "-", "-", "-"
            remark = "core indicators missing; cannot assess"
        else:
            remark = "data complete, computable"
            if total_score <= 2:
                risk_level, risk_code, suggestion = "low risk", "G47.2.1", "regular follow-up"
            elif total_score <= 4:
                risk_level, risk_code, suggestion = "medium risk", "G47.2.2", "specialist consultation advised"
            else:
                risk_level, risk_code, suggestion = "high risk", "G47.2.3", "strongly recommend PSG"

        # Assemble results
        results.append({
            'pat_id': r.get('pat_id'),
            'app_no': r.get('app_no'),
            'pat_name': r.get('pat_name', 'unknown'),
            'model_name': 'STOP-Bang OSA Risk Prediction Model',

            # --- Full transparency for the 8 inputs ---
            'sex': r.get('pat_sex', 'missing'),
            'age': int(age) if pd.notna(age) else "missing",
            'BMI': fmt_in(bmi, True),
            'neck_circumference (cm)': fmt_in(neck, True),
            'snoring (S)': fmt_in(r.get('snoring'), False, True),
            'tired (T)': fmt_in(r.get('tiredness'), False, True),
            'observed_apnea (O)': fmt_in(r.get('observed'), False, True),
            'hypertension (P)': fmt_in(r.get('has_hypertension'), False, True),

            # --- Result display ---
            'high_risk_hits': details if total_score != "-" else "-",
            'predicted_score': int(total_score) if total_score != "-" else "-",
            'code': risk_code,
            'gauge_reference': GAUGE_REF_STR if total_score != "-" else "-",
            'risk_level': risk_level,
            'recommendation': suggestion,
            'assessment_note': remark
        })

    if results:
        res_df = pd.DataFrame(results)

        # Layout: order basic info, results and the 8 raw indicators sensibly
        ordered_cols = [
            'pat_id', 'app_no', 'pat_name', 'model_name', 'code', 'gauge_reference',
            'total_score', 'risk_level', 'high_risk_hits', 'recommendation', 'assessment_note',
            'sex', 'age', 'BMI', 'neck_circumference (cm)', 'snoring (S)', 'tired (T)', 'observed_apnea (O)', 'hypertension (P)'
        ]
        final_cols = [c for c in ordered_cols if c in res_df.columns]

        res_df[final_cols].to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"--- Assessment complete: {len(res_df)} results generated, exported to {output_path} ---")
    else:
        print("--- Assessment complete, but no valid results were generated ---")


if __name__ == "__main__":
    process_data("base.csv", "item_result.csv", "stop_bang_screening_result_perfect.csv")
