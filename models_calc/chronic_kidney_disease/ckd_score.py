import pandas as pd
import numpy as np
import re

# ==========================================
# 1. Field mapping (original version retained)
# ==========================================
FIELD_MAPPING = {
    'age': ['app_pat_age', 'age'],
    'gender': ['pat_sex', 'sex'],
    'hb': ['hemoglobin (HGB)', 'hemoglobin', 'HGB', 'Hb', 'anemia_history', 'anemia'],
    'has_hypertension': ['has_hypertension', 'past_history', 'past_medical_history', 'hypertension_history', 'hypertension'],
    'sbp': ['sbp', 'systolic_bp', 'blood_pressure'],
    'dbp': ['dbp', 'diastolic_bp', 'blood_pressure'],
    'has_diabetes': ['has_diabetes', 'past_history', 'past_medical_history', 'diabetes_history', 'diabetes'],
    'FBG': ['GLU', 'fasting_glucose'],
    'Hba1c': ['Hba1c', 'HbA1c', 'GHb'],
    'has_cvd': ['has_cvd', 'past_history', 'past_medical_history', 'cardiovascular_disease', 'coronary_heart_disease', 'myocardial_infarction', 'heart_attack'],
    'has_stroke': ['has_stroke', 'past_history', 'past_medical_history', 'stroke'],
    'has_heart_failure': ['heart_failure', 'congestive_heart_failure', 'CHF', 'past_history'],
    'has_pvd': ['peripheral_vascular_disease', 'leg_circulation_disease', 'intermittent_claudication', 'lower_limb_atherosclerosis', 'PVD', 'past_history'],
    'urine_protein': ['urine_albumin (ALBU)']
}


# ==========================================
# 2. Basic cleaning logic
# ==========================================
def normalize_str(s):
    if pd.isna(s): return ""
    return re.sub(r'[\(\)\（\）\[\]\【\】\{\}\s\-\_]', '', str(s)).lower()


def get_categories_list(name):
    norm_name = normalize_str(name)
    if not norm_name: return []
    matched_keys = []
    for key, patterns in FIELD_MAPPING.items():
        if any(normalize_str(p) == norm_name for p in patterns):
            matched_keys.append(key)
    return matched_keys


def clean_numeric(val):
    if pd.isna(val) or str(val).strip() == "": return np.nan
    try:
        match = re.search(r"[-+]?\d*\.\d+|\d+", str(val))
        return float(match.group()) if match else np.nan
    except:
        return np.nan


def check_condition(values, field_key, context_keywords=None, sex_str=None):
    if not values: return None
    joined_text = " ".join([str(v) for v in values]).lower()
    nums = [clean_numeric(v) for v in values if pd.notna(clean_numeric(v))]

    # Clinical indicator determination
    if field_key == 'hb' and nums:
        hb = np.mean(nums)
        is_female = 'female' in str(sex_str) or 'female' in str(sex_str).lower()
        threshold = 120 if is_female else 130
        if 30 < hb < threshold: return True
        if hb >= threshold: return False

    if field_key == 'sbp' and any(n >= 140 for n in nums): return True
    if field_key == 'dbp' and any(n >= 90 for n in nums): return True
    if field_key == 'FBG' and any(n >= 7.0 for n in nums): return True
    if field_key == 'Hba1c' and any(n >= 6.5 for n in nums): return True

    if field_key == 'urine_protein':
        if re.search(r'(\d+)?\+', joined_text) or 'positive' in joined_text: return True
        if 'negative' in joined_text or '-' in joined_text: return False

    # Text keyword detection
    if any(kw in joined_text for kw in ['yes', 'present', 'true', 'positive']): return True
    if context_keywords:
        for kw in context_keywords:
            if kw in joined_text: return True
    if any(kw in joined_text for kw in ['no', 'absent', 'false', 'negative']): return False
    return None


# ==========================================
# 3. SCORED model scoring logic
# ==========================================
def calculate_scored_ckd(age, gender, is_anemia, is_htn, is_dm, is_cvd, is_chf, is_pvd, is_proteinuria):
    # Core-field missing check (age and sex are required)
    if pd.isna(age) or not gender:
        return None, None, None

    score = 0
    # 1. Age score
    age_val = float(age)
    if 50 <= age_val <= 59:
        score += 2
    elif 60 <= age_val <= 69:
        score += 3
    elif age_val >= 70:
        score += 4

    # 2. Sex score (female)
    if str(gender).strip() in ['female', 'Female', 'F', '1']:
        score += 1

    # 3. Disease-history score (add 1 when True; no points for False or None)
    # Note: the main routine already guarantees the History field is not None
    conditions = [is_anemia, is_htn, is_dm, is_cvd, is_chf, is_pvd, is_proteinuria]
    score += sum([1 for cond in conditions if cond is True])

    risk_label = "high risk" if score >= 4 else "low risk"
    risk_code = "N18.2.3" if score >= 4 else "N18.2.1"
    return score, risk_label, risk_code


# ==========================================
# 4. Main data-processing routine
# ==========================================
def process_data(base_path, item_path, output_path):
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

    # Preprocess item data
    item_df['cat'] = item_df['res_itm_name'].apply(get_categories_list)
    filtered = item_df.explode('cat').dropna(subset=['cat', 'res_value']).copy()

    item_dict_map = {}
    for (app_no, pat_id), group in filtered.groupby(['app_no', 'pat_id']):
        key = (str(app_no).strip(), str(pat_id).strip())
        data_pool = {k: [] for k in FIELD_MAPPING.keys()}
        for _, item in group.iterrows():
            data_pool[item['cat']].append(item['res_value'])
        item_dict_map[key] = data_pool

    base_df['app_no'] = base_df['app_no'].astype(str).str.strip()
    base_df['pat_id'] = base_df['pat_id'].astype(str).str.strip()

    results = []
    for _, row in base_df.iterrows():
        app_no_val, pat_id_val = row['app_no'], row['pat_id']
        key = (app_no_val, pat_id_val)
        data_pool = item_dict_map.get(key, {k: [] for k in FIELD_MAPPING.keys()})

        age = clean_numeric(row.get('app_pat_age'))
        gender = str(row.get('pat_sex', '')).strip()

        # --- Decision logic and labeling ---
        notes = []

        # 1. Anemia / Hb
        cond_anemia = check_condition(data_pool['hb'], 'hb', context_keywords=['anemia'], sex_str=gender)

        # 2. Hypertension (history OR blood pressure)
        c_htn_hist = check_condition(data_pool['has_hypertension'], 'has_hypertension', context_keywords=['hypertension'])
        c_sbp, c_dbp = check_condition(data_pool['sbp'], 'sbp'), check_condition(data_pool['dbp'], 'dbp')
        raw_htn = (c_htn_hist is True or c_sbp is True or c_dbp is True)
        if c_htn_hist is None and c_sbp is None and c_dbp is None:
            cond_htn = False
            notes.append("hypertension_history (default none)")
        else:
            cond_htn = raw_htn

        # 3. Diabetes (history OR glucose value)
        c_dm_hist = check_condition(data_pool['has_diabetes'], 'has_diabetes', context_keywords=['diabetes'])
        c_fbg, c_ghb = check_condition(data_pool['FBG'], 'FBG'), check_condition(data_pool['Hba1c'], 'Hba1c')
        raw_dm = (c_dm_hist is True or c_fbg is True or c_ghb is True)
        if c_dm_hist is None and c_fbg is None and c_ghb is None:
            cond_dm = False
            notes.append("diabetes_history (default none)")
        else:
            cond_dm = raw_dm

        # 4. Cardiovascular disease / stroke
        c_cvd = check_condition(data_pool['has_cvd'], 'has_cvd', context_keywords=['cardiovascular', 'coronary_heart_disease', 'infarction'])
        c_stroke = check_condition(data_pool['has_stroke'], 'has_stroke', context_keywords=['stroke'])
        if c_cvd is None and c_stroke is None:
            cond_cvd = False
            notes.append("cardiovascular_stroke_history (default none)")
        else:
            cond_cvd = (c_cvd is True or c_stroke is True)

        # 5. Other disease history (default determination)
        cond_chf = check_condition(data_pool['has_heart_failure'], 'has_heart_failure',
                                   context_keywords=['heart_failure'])
        if cond_chf is None:
            cond_chf = False
            notes.append("heart_failure_history (default none)")

        cond_pvd = check_condition(data_pool['has_pvd'], 'has_pvd', context_keywords=['peripheral_vascular', 'atherosclerosis', 'claudication'])
        if cond_pvd is None:
            cond_pvd = False
            notes.append("peripheral_vascular_history (default none)")

        # 6. Proteinuria (not normally defaulted clinically; for computation, treat no record as absent or keep missing)
        cond_prot = check_condition(data_pool['urine_protein'], 'urine_protein', context_keywords=['proteinuria', 'positive'])
        if cond_prot is None:
            cond_prot = False
            notes.append("proteinuria (missing treated as none)")

        # Compute score
        score, risk_label, risk_code = calculate_scored_ckd(
            age, gender, cond_anemia, cond_htn, cond_dm, cond_cvd, cond_chf, cond_pvd, cond_prot
        )

        all_hist = data_pool['has_hypertension'] + data_pool['has_diabetes'] + data_pool['has_cvd'] + \
                   data_pool['has_stroke'] + data_pool['has_heart_failure'] + data_pool['has_pvd']

        results.append({
            'pat_id': pat_id_val,
            'app_no': app_no_val,
            'pat_name': row.get('pat_name', 'unknown'),
            'sex': gender,
            'age': age if not pd.isna(age) else "-",
            'code': risk_code if risk_code else "",
            'predicted_score': score if score is not None else "-",
            'risk_level': risk_label if risk_label else "-",
            'data_annotation': "；".join(notes) if notes else "raw_data_complete",
            'hemoglobin': "/".join(map(str, data_pool['hb'])) if data_pool['hb'] else "-",
            'systolic_bp': "/".join(map(str, data_pool['sbp'])) if data_pool['sbp'] else "-",
            'diastolic_bp': "/".join(map(str, data_pool['dbp'])) if data_pool['dbp'] else "-",
            'fasting_glucose': "/".join(map(str, data_pool['FBG'])) if data_pool['FBG'] else "-",
            'urine_protein': "/".join(map(str, data_pool['urine_protein'])) if data_pool['urine_protein'] else "-",
            'conclusion_status': 'score_success' if score is not None else 'age_sex_missing'
        })

    if results:
        pd.DataFrame(results).to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"Processing complete, results saved to: {output_path}")


if __name__ == "__main__":
    # Ensure base.csv and item_result.csv are in the current directory
    process_data("base.csv", "item_result.csv", "ckd_scored_screening_N18_result.csv")
