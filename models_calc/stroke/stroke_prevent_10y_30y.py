import pandas as pd
import numpy as np
import re

# ==========================================
# 1. Field mapping
# ==========================================
FIELD_MAPPING = {
    'age': ['app_pat_age'],
    'gender': ['pat_sex'],
    'sbp': ['systolic_bp', 'blood_pressure'],
    'TC': ['total_cholesterol', 'CHOL'],
    'HDL-C': ['HDL_cholesterol', 'HDL', 'HDL-C'],
    'egfr': ['eGFR (CKD-EPI)', 'eGFR'],
    'scr': ['creatinine (Cr)', 'creatinine', 'serum_creatinine'],
    'has_diabetes': ['past_history', 'past_medical_history', 'type2_diabetes', 'prior_diabetes', 'diabetes'],
    'smoke': ['smoking_history'],
    'bp_treatment': ['past_history', 'past_medical_history', 'antihypertensive_treatment', 'on_antihypertensive'],
    'statin_use': ['on_lipid_lowering_drug', 'statin', 'lipid_lowering_therapy', 'past_history']
}


# ==========================================
# 2. Basic helpers and unit-alignment functions
# ==========================================
def normalize_str(s):
    if pd.isna(s):
        return ""
    return re.sub(r'[\(\)\（\）\[\]\【\】\{\}\s\|\、]', '', str(s)).lower()


def get_categories(name):
    norm_name = normalize_str(name)
    matched_keys = []
    for key, patterns in FIELD_MAPPING.items():
        for p in patterns:
            if normalize_str(p) == norm_name:
                matched_keys.append(key)
                break
    return matched_keys


def clean_numeric(val):
    if pd.isna(val) or str(val).strip() == "":
        return np.nan
    try:
        match = re.search(r"[-+]?\d*\.\d+|\d+", str(val))
        return float(match.group()) if match else np.nan
    except:
        return np.nan


def parse_sex(val):
    s = str(val).strip()
    if 'male' in s or 'M' in s.upper() or str(val) == '1':
        return 1  # Male
    return 0  # Female


def align_lipid_unit(val, unit_str):
    if pd.isna(val):
        return val

    unit_str = str(unit_str).strip().lower()
    if 'mmol' in unit_str:
        return round(val / 0.0259, 2)
    elif 'mg/dl' in unit_str or 'mg' in unit_str:
        return round(val, 2)

    if val < 20:
        return round(val / 0.0259, 2)

    return round(val, 2)


# ==========================================
# 3. Compute eGFR with the CKD-EPI 2021 formula
# ==========================================
def calc_egfr_from_scr(scr, age, sex):
    if pd.isna(scr) or pd.isna(age):
        return np.nan

    if scr > 50:
        scr = scr / 88.4

    if sex == 1:  # Male
        alpha = -0.302
        kappa = 0.9
        multiplier = 142
        gender_coeff = 1.0
    else:  # Female
        alpha = -0.241
        kappa = 0.7
        multiplier = 142
        gender_coeff = 1.012

    egfr = multiplier * (min(scr / kappa, 1) ** alpha) * \
           (max(scr / kappa, 1) ** -1.200) * \
           (0.9938 ** age) * gender_coeff
    return egfr


# ==========================================
# 4. Context analysis and history extraction
# ==========================================
def check_target_with_negation(combined_text, targets):
    if isinstance(targets, str):
        targets = [targets]
    text = re.sub(r'\s+', '', combined_text).lower()
    if not any(t in text for t in targets):
        return False
    clauses = re.split(r'[。；;\n]', text)
    for clause in clauses:
        for target in targets:
            if target in clause:
                if re.search(r'(denies|no|none|not found|not seen|not taking|not using|not eating)[^present]*?' + target, clause):
                    continue
                if re.search(target + r'[^\w]*?(no|denies|none|not taking|not using|not eating|negative)', clause):
                    continue
                return True
    return False


def parse_history_and_meds(key, val_list):
    positive_exact_matches = ['1', 'yes', 'present', 'true', 'positive']
    for v in val_list:
        v_str = str(v).strip().lower()
        if v_str in positive_exact_matches:
            return 1

    combined_text = " ".join([str(v) for v in val_list if not pd.isna(v)])

    if key == 'has_diabetes':
        targets = ['diabetes', 'glucose_lowering', 'insulin', 'metformin', 'acarbose', 'gliflozin', 'gliptin', 'glimepiride', 'voglibose']
    elif key == 'bp_treatment':
        targets = ['antihypertensive', 'dihydropyridine', 'sartan', 'pril', 'lol', 'thiazide', 'terazosin', 'prazosin']
    elif key == 'statin_use':
        targets = ['statin', 'lipid_lowering', 'atorvastatin', 'rosuvastatin', 'pitavastatin', 'simvastatin', 'lovastatin', 'pravastatin', 'fluvastatin', 'fibrate', 'ezetimibe', 'xuezhikang', 'zhibitai']
    elif key == 'smoke':
        targets = ['smoking', 'smoker', 'heavy_smoker']
    else:
        return 0

    if check_target_with_negation(combined_text, targets):
        return 1

    if key == 'bp_treatment':
        text_no_space = re.sub(r'\s+', '', combined_text).lower()
        clauses = re.split(r'[。；;\n]', text_no_space)
        for clause in clauses:
            if ('hypertension' in clause or 'high_normal_bp' in clause) and (
                    'medication' in clause or 'medication' in clause or 'medication' in clause or 'controlled' in clause):
                if not re.search(r'(not|none|no|refuses|denies)[^present]*?(medicated|medication|taking|treatment)', clause):
                    return 1
    return 0


def get_risk_code(risk_val):
    if pd.isna(risk_val): return "-"
    if risk_val < 5.0:
        return "I64.2.1"
    elif 5.0 <= risk_val < 10.0:
        return "I64.2.2"
    else:
        return "I64.2.3"

    # ==========================================


# 5. AHA PREVENT stroke risk computation core
# ==========================================
def calculate_stroke_risk(sex, age, sbp, tc, hdl, diabetes, smoker, egfr, treated_bp, statin, years=10):
    if pd.isna(age) or pd.isna(sbp) or pd.isna(tc) or pd.isna(hdl) or pd.isna(egfr):
        return np.nan

    tc_mmol = tc * 0.02586
    hdl_mmol = hdl * 0.02586
    non_hdl_mmol = tc_mmol - hdl_mmol

    age_spline = (age - 55) / 10
    non_hdl_spline = non_hdl_mmol - 3.5
    hdl_spline = (hdl_mmol - 1.3) / 0.3
    sbp_min110_spline = (min(sbp, 110) - 110) / 20
    sbp_max110_spline = (max(sbp, 110) - 130) / 20
    egfr_min60_spline = (min(egfr, 60) - 60) / -15
    egfr_max60_spline = (max(egfr, 60) - 90) / -15

    if sex == 0:
        score = (
                age_spline * 0.6907849 + non_hdl_spline * 0.0534279 + hdl_spline * -0.1055109 +
                sbp_min110_spline * -0.113078 + sbp_max110_spline * 0.3665217 +
                diabetes * 0.8013721 + smoker * 0.4187039 +
                egfr_min60_spline * 0.4539767 + egfr_max60_spline * 0.0515087 +
                treated_bp * 0.2494624 + statin * -0.0798829 +
                sbp_max110_spline * treated_bp * -0.0079039 +
                non_hdl_spline * statin * 0.0833101 +
                age_spline * non_hdl_spline * -0.0409242 +
                age_spline * hdl_spline * 0.016994 +
                age_spline * sbp_max110_spline * -0.1191213 +
                age_spline * diabetes * -0.2480549 +
                age_spline * smoker * -0.0998063 +
                age_spline * egfr_min60_spline * -0.1759075 +
                -4.409199
        )
    else:
        score = (
                age_spline * 0.722513 + non_hdl_spline * 0.0263348 + hdl_spline * -0.0248959 +
                sbp_min110_spline * -0.268104 + sbp_max110_spline * 0.3474634 +
                diabetes * 0.684699 + smoker * 0.3874844 +
                egfr_min60_spline * 0.3877827 + egfr_max60_spline * 0.0201965 +
                treated_bp * 0.232963 + statin * -0.1178935 +
                sbp_max110_spline * treated_bp * 0.0120926 +
                non_hdl_spline * statin * 0.155739 +
                age_spline * non_hdl_spline * 0.0141928 +
                age_spline * hdl_spline * -0.0111745 +
                age_spline * sbp_max110_spline * -0.1155391 +
                age_spline * diabetes * -0.2123743 +
                age_spline * smoker * -0.0824133 +
                age_spline * egfr_min60_spline * -0.180789 +
                -4.20881
        )

    risk = np.exp(score) / (1 + np.exp(score))
    return risk * 100


# ==========================================
# 6. Main data-processing routine
# ==========================================
def process_data(base_path, item_path, output_path):
    print("--- Starting AHA PREVENT risk assessment ---")

    def read_csv_safe(path):
        for enc in ['utf-8-sig', 'gbk', 'utf-8']:
            try:
                return pd.read_csv(path, encoding=enc, low_memory=False)
            except:
                continue
        return pd.DataFrame()

    base_df = read_csv_safe(base_path)
    item_df = read_csv_safe(item_path)

    if base_df.empty or item_df.empty:
        print("error: input file missing")
        return

    item_df['cats'] = item_df['res_itm_name'].apply(get_categories)
    filtered = item_df[item_df['cats'].map(len) > 0].dropna(subset=['res_value']).copy()

    base_df['app_no'] = base_df['app_no'].astype(str).str.strip()
    base_df['pat_id'] = base_df['pat_id'].astype(str).str.strip()

    base_info = base_df.set_index('app_no')[['app_pat_age', 'pat_sex']].to_dict('index')

    records = []
    grouped = filtered.groupby(['app_no', 'pat_id'])

    for (app_no, pat_id), group in grouped:
        row = {'app_no': str(app_no).strip(), 'pat_id': str(pat_id).strip()}

        has_unit_col = 'res_unit' in group.columns
        data_pool = {k: [] for k in FIELD_MAPPING.keys()}
        unit_pool = {k: [] for k in FIELD_MAPPING.keys()}

        for _, item in group.iterrows():
            cats = item['cats']
            for cat in cats:
                data_pool[cat].append(str(item['res_value']))
                if has_unit_col:
                    unit_pool[cat].append(item['res_unit'] if not pd.isna(item['res_unit']) else "")
                else:
                    unit_pool[cat].append("")

        for cat in FIELD_MAPPING.keys():
            row[f'raw_{cat}'] = " / ".join(data_pool.get(cat, []))

        row['sbp'] = clean_numeric("".join(data_pool.get('sbp', [])))

        raw_tc_val = clean_numeric("".join(data_pool.get('TC', [])))
        tc_units = unit_pool.get('TC', [])
        row['TC'] = align_lipid_unit(raw_tc_val, tc_units[0] if tc_units else "")

        raw_hdl_val = clean_numeric("".join(data_pool.get('HDL-C', [])))
        hdl_units = unit_pool.get('HDL-C', [])
        row['HDL-C'] = align_lipid_unit(raw_hdl_val, hdl_units[0] if hdl_units else "")

        raw_egfr = clean_numeric("".join(data_pool.get('egfr', [])))
        raw_scr = clean_numeric("".join(data_pool.get('scr', [])))

        if not pd.isna(raw_egfr):
            row['egfr'] = raw_egfr
        elif not pd.isna(raw_scr):
            p_info = base_info.get(row['app_no'], {})
            p_age = clean_numeric(p_info.get('app_pat_age'))
            p_sex = parse_sex(p_info.get('pat_sex'))
            row['egfr'] = calc_egfr_from_scr(raw_scr, p_age, p_sex)
        else:
            row['egfr'] = np.nan

        row['has_diabetes'] = parse_history_and_meds('has_diabetes', data_pool.get('has_diabetes', []))
        row['smoke'] = parse_history_and_meds('smoke', data_pool.get('smoke', []))
        row['bp_treatment'] = parse_history_and_meds('bp_treatment', data_pool.get('bp_treatment', []))
        row['statin_use'] = parse_history_and_meds('statin_use', data_pool.get('statin_use', []))

        records.append(row)

    wide_items = pd.DataFrame(records)
    final = pd.merge(wide_items, base_df, on=['app_no', 'pat_id'], how='inner')

    results = []
    print("Generating final assessment results...")

    for _, r in final.iterrows():
        age = clean_numeric(r.get('app_pat_age'))
        sex = parse_sex(r.get('pat_sex'))

        risk_score = calculate_stroke_risk(
            sex=sex, age=age, sbp=r['sbp'], tc=r['TC'], hdl=r['HDL-C'],
            diabetes=r['has_diabetes'], smoker=r['smoke'], egfr=r['egfr'],
            treated_bp=r['bp_treatment'], statin=r['statin_use']
        )

        code = get_risk_code(risk_score)
        if pd.isna(risk_score):
            risk_level, action = "-", "core indicators missing (e.g. BP, lipids, eGFR); complete the data."
        elif risk_score < 5.0:
            risk_level, action = "low risk", "continue a healthy lifestyle; regular rechecks."
        elif 5.0 <= risk_score < 10.0:
            risk_level, action = "medium risk", "intensify lifestyle intervention; control BP, lipids and other risk factors."
        else:
            risk_level, action = "high risk", "see a specialist promptly for necessary medication or medical intervention."

        results.append({
            'model_name': 'AHA PREVENT Stroke Risk Assessment',
            'pat_id': r['pat_id'],
            'app_no': r['app_no'],
            'pat_name': r.get('pat_name', r.get('name', 'unknown')),
            'sex': r.get('pat_sex', ''),
            'age': r.get('app_pat_age', '-'),
            'risk_level': risk_level,
            'prediction_horizon': 'next 10 years',
            'risk_probability': f"{risk_score:.2f}" if not pd.isna(risk_score) else "-",
            'code': code,
            'gauge_reference': "0%-5%|5%-10%|10%-100%",
            'systolic_bp': r['raw_sbp'],
            'total_cholesterol': r['raw_TC'],
            'HDL': r['raw_HDL-C'],
            'eGFR (computed)': f"{r['egfr']:.2f}" if pd.notna(r['egfr']) else "-",
            'raw_eGFR_text': r['raw_egfr'],

            # [Output format changed to the specified one]
            'creatinine': r['raw_scr'] if str(r.get('raw_scr', '')).strip() else "-",
            'diabetes_history': r['raw_has_diabetes'] if str(r.get('raw_has_diabetes', '')).strip() else "missing (defaulted to none)",
            'smoking_history': r['raw_smoke'] if str(r.get('raw_smoke', '')).strip() else "missing (defaulted to none)",
            'antihypertensive_treatment': r['raw_bp_treatment'] if str(r.get('raw_bp_treatment', '')).strip() else "missing (defaulted to none)",
            'lipid_lowering_therapy': r['raw_statin_use'] if str(r.get('raw_statin_use', '')).strip() else "missing (defaulted to none)"
        })

    if results:
        res_df = pd.DataFrame(results)
        res_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"--- Exported {len(res_df)} records to {output_path} ---")
    else:
        print("No valid results generated.")


if __name__ == "__main__":
    process_data("base.csv", "item_result.csv", "aha_prevent_stroke_results.csv")