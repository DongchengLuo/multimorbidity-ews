import pandas as pd
import numpy as np
import re

# ==========================================
# 1. Field mapping (for the "8+2" stroke screening scale and lab items)
# ==========================================
FIELD_MAPPING = {
    'sbp': ['systolic_bp', 'blood_pressure'],
    'dbp': ['diastolic_bp', 'blood_pressure'],
    'height': ['height'],
    'bmi': ['bmi', 'BMI'],
    'weight': ['weight'],
    'tc': ['total_cholesterol', 'CHOL', 'cholesterol'],
    'tg': ['triglyceride', 'TG'],
    'ldl': ['LDL', 'LDL-C'],
    'glu': ['glucose', 'GLU', 'glu'],  # glucose item
    'hba1c': ['HbA1c', 'GHb', 'hba1c', 'ghb'],  # HbA1c item
    'smoke_status': ['smoking_history', 'smoking_status', 'smoker'],
    'exercise_status': ['physical_exercise', 'exercise_status', 'physical_inactivity', 'exercise'],
    # Disease-specific fields
    'history_htn': ['hypertension_history', 'prior_hypertension', 'high_bp', 'high_normal_bp'],
    'history_dyslipidemia': ['dyslipidemia', 'hyperlipidemia', 'history_lipids', 'lipids_high'],
    'history_dm': ['diabetes_history', 'prior_diabetes'],
    'history_af_valvular': ['atrial_fibrillation', 'valvular_heart_disease', 'heart_disease_history', 'ecg'],  # include ECG to scan conclusions
    'family_history_stroke': ['stroke_family_history', 'family_stroke'],
    'history_stroke': ['stroke_history', 'prior_stroke'],
    'history_tia': ['transient_ischemic_attack', 'TIA', 'tia'],
    'general_history': ['past_history', 'medical_history', 'other_history']
}


# ==========================================
# 2. Basic helper functions
# ==========================================
def normalize_str(s):
    if pd.isna(s): return ""
    return re.sub(r'[\(\)\（\）\[\]\【\】\{\}\s\-\_]', '', str(s)).lower()


def get_category(name):
    norm_name = normalize_str(name)
    for key, patterns in FIELD_MAPPING.items():
        for p in patterns:
            if normalize_str(p) == norm_name:
                return key
    return None


def clean_numeric(val):
    if pd.isna(val) or str(val).strip() == "": return np.nan
    try:
        # Handle values with units, e.g. "7.2mmol/L"
        match = re.search(r"[-+]?\d*\.\d+|\d+", str(val))
        return float(match.group()) if match else np.nan
    except:
        return np.nan


def check_positive_keyword(text_list, keywords):
    combined_text = normalize_str(" ".join([str(t) for t in text_list if not pd.isna(t)]))
    positive_flags = ['yes', 'present', '1', 'true', 'positive'] + keywords
    return any(k in combined_text for k in positive_flags)


def check_disease_with_negation(text_list, targets):
    if isinstance(targets, str): targets = [targets]
    text = normalize_str(" ".join([str(t) for t in text_list if not pd.isna(t)]))
    if not any(t in text for t in targets): return False

    clauses = re.split(r'[。；;,\n]', text)
    for clause in clauses:
        for target in targets:
            if target in clause:
                if re.search(r'(denies|no|none|not found|not seen|excluded)[^present]*?' + target, clause): continue
                if re.search(target + r'[^present]*?(no|denies|none)', clause): continue
                return True
    return False


# ==========================================
# 3. "8+2" core risk-tiering logic
# ==========================================
def get_risk_info(risk_flags, warning_flags):
    risk_count = sum(1 for v in risk_flags.values() if v)
    has_stroke = warning_flags.get('prior_stroke', False)
    has_tia = warning_flags.get('prior_tia', False)

    if risk_count >= 3 or has_stroke or has_tia:
        return "high risk", "see a neurologist promptly; comprehensive intervention", risk_count, "I64.4.3"
    elif risk_count < 3 and (risk_flags['hypertension'] or risk_flags['diabetes'] or risk_flags['af_valvular']):
        return "medium risk", "intensify intervention; control underlying disease; regular follow-up", risk_count, "I64.4.2"
    else:
        return "low risk", "continue a healthy lifestyle; regular check-ups", risk_count, "I64.4.1"


# ==========================================
# 4. Main data-processing routine
# ==========================================
def process_data(base_path, item_path, output_path):
    print("--- Starting [8+2 stroke screening] enhanced linkage version ---")

    def read_csv_safe(path):
        for enc in ['utf-8-sig', 'gbk', 'utf-8', 'latin1']:
            try:
                return pd.read_csv(path, encoding=enc, low_memory=False)
            except:
                continue
        return pd.DataFrame()

    base_df = read_csv_safe(base_path)
    item_df = read_csv_safe(item_path)
    if base_df.empty or item_df.empty: return

    for df in [base_df, item_df]:
        for col in ['app_no', 'pat_id']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int).astype(str).str.replace(r'\.0$',
                                                                                                                '',
                                                                                                                regex=True)

    item_df['cat'] = item_df['res_itm_name'].apply(get_category)
    filtered = item_df.dropna(subset=['cat', 'res_value']).copy()

    records = []
    for (app_no, pat_id), group in filtered.groupby(['app_no', 'pat_id']):
        row = {'app_no': app_no, 'pat_id': pat_id}
        data_pool = {k: [] for k in FIELD_MAPPING.keys()}
        for _, item in group.iterrows():
            data_pool[item['cat']].append(str(item['res_value']))

        # Past-history pool
        gen_hist = data_pool.get('general_history', [])

        # 1. Hypertension determination (indicators + history)
        sbp_vals = [clean_numeric(v) for v in data_pool.get('sbp', []) if not pd.isna(clean_numeric(v))]
        dbp_vals = [clean_numeric(v) for v in data_pool.get('dbp', []) if not pd.isna(clean_numeric(v))]
        sbp = sbp_vals[0] if sbp_vals else np.nan
        dbp = dbp_vals[0] if dbp_vals else np.nan
        hist_htn = check_disease_with_negation(data_pool['history_htn'] + gen_hist, ["hypertension", "high_bp", "high_normal_bp"])
        row['flag_htn'] = (not pd.isna(sbp) and sbp >= 140) or (not pd.isna(dbp) and dbp >= 90) or hist_htn

        # 2. Diabetes determination (indicator linkage: glucose >= 7.0 or HbA1c >= 6.5%)
        glu_vals = [clean_numeric(v) for v in data_pool.get('glu', []) if not pd.isna(clean_numeric(v))]
        hba1c_vals = [clean_numeric(v) for v in data_pool.get('hba1c', []) if not pd.isna(clean_numeric(v))]
        glu = glu_vals[0] if glu_vals else np.nan
        hba1c = hba1c_vals[0] if hba1c_vals else np.nan
        hist_dm = check_disease_with_negation(data_pool['history_dm'] + gen_hist, "diabetes")
        # Linkage logic
        row['flag_dm'] = hist_dm or (not pd.isna(glu) and glu >= 7.0) or (not pd.isna(hba1c) and hba1c >= 6.5)

        # 3. Atrial fibrillation determination (scan ECG conclusions)
        row['flag_af'] = check_disease_with_negation(data_pool['history_af_valvular'] + gen_hist,
                                                     ["atrial_fibrillation", "valve", "heart_disease"])

        # 4. Dyslipidemia
        tc = clean_numeric(data_pool['tc'][0]) if data_pool['tc'] else np.nan
        tg = clean_numeric(data_pool['tg'][0]) if data_pool['tg'] else np.nan
        ldl = clean_numeric(data_pool['ldl'][0]) if data_pool['ldl'] else np.nan
        hist_lipid = check_disease_with_negation(data_pool['history_dyslipidemia'] + gen_hist,
                                                 ["lipids", "high_cholesterol", "lipids_high"])
        row['flag_dyslipidemia'] = hist_lipid or (not pd.isna(tc) and tc >= 5.2) or (not pd.isna(tg) and tg >= 1.7) or (
                    not pd.isna(ldl) and ldl >= 3.4)

        # 5. Other risk factors
        bmi_v = clean_numeric(data_pool['bmi'][0]) if data_pool['bmi'] else np.nan
        if pd.isna(bmi_v):
            h = clean_numeric(data_pool['height'][0]) if data_pool['height'] else np.nan
            w = clean_numeric(data_pool['weight'][0]) if data_pool['weight'] else np.nan
            if h > 0 and w > 0: bmi_v = round(w / ((h / 100) ** 2), 2)
        row['flag_obesity'] = (not pd.isna(bmi_v) and bmi_v >= 28)
        row['flag_smoke'] = check_positive_keyword(data_pool['smoke_status'], ['smoking', 'smoker'])
        row['flag_exercise_lack'] = check_positive_keyword(data_pool['exercise_status'], ['few', 'absent', 'lack'])
        row['flag_family'] = check_disease_with_negation(data_pool['family_history_stroke'] + gen_hist,
                                                         ["family_history", "genetic"])
        row['flag_stroke_hist'] = check_disease_with_negation(data_pool['history_stroke'] + gen_hist,
                                                              ["stroke"])
        row['flag_tia_hist'] = check_disease_with_negation(data_pool['history_tia'] + gen_hist,
                                                           ["transient_ischemic_attack", "tia"])

        # Result auxiliary columns
        row['raw_sbp'] = sbp
        row['raw_glu'] = glu
        row['raw_bmi'] = bmi_v
        row['raw_general_history'] = " / ".join(list(dict.fromkeys(gen_hist)))
        records.append(row)

    wide_df = pd.DataFrame(records)
    final = pd.merge(wide_df, base_df, on=['app_no', 'pat_id'], how='inner')

    results = []
    for _, r in final.iterrows():
        risk_flags = {
            'hypertension': r['flag_htn'], 'dyslipidemia': r['flag_dyslipidemia'],
            'diabetes': r['flag_dm'], 'af_valvular': r['flag_af'],
            'smoking': r['flag_smoke'], 'obesity': r['flag_obesity'],
            'exercise_lack': r['flag_exercise_lack'], 'family_history': r['flag_family']
        }
        warn_flags = {'prior_stroke': r['flag_stroke_hist'], 'prior_tia': r['flag_tia_hist']}
        level, advice, score, risk_code = get_risk_info(risk_flags, warn_flags)

        results.append({
            'model_name': '8+2 Stroke Screening Scale',
            'pat_id': r['pat_id'],
            'app_no': r['app_no'],
            'name': r.get('pat_name', 'unknown'),
            'sex': r.get('pat_sex', ''),
            'age': r.get('app_pat_age', '-'),
            'risk_level': level,
            'predicted_score': score,
            'code': risk_code,
            'gauge_reference': "0-2|3-5|6-8",
            'recommendation': advice,
            'BMI': r['raw_bmi'],
            'systolic_bp': r['raw_sbp'],
            'glucose (linkage-determined)': r['raw_glu'],
            'raw_past_history': r['raw_general_history'],
            'hypertension_determination': "positive" if r['flag_htn'] else "negative",
            'diabetes_determination': "positive" if r['flag_dm'] else "negative",
            'stroke_history_determination': "positive" if r['flag_stroke_hist'] else "negative"
        })

    if results:
        pd.DataFrame(results).to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"Assessment complete; results saved to {output_path}")


if __name__ == "__main__":
    process_data("base.csv", "item_result.csv", "stroke_8plus2_screening_result.csv")