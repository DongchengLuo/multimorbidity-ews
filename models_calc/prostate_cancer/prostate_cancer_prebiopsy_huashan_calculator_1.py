import pandas as pd
import numpy as np
import re

# ==========================================
# 1. Field mapping (see the precise field-matching reference table)
# ==========================================
FIELD_MAPPING = {
    'age': ['app_pat_age', 'age'],
    'psa_value': ['PSA', 'total_PSA', 'TPSA', 'psa'],
    'fPSA/tPSA': ['fPSA/tPSA', 'free_ratio'],
    'pv_value': ['prostate_volume', 'PV', 'prostate_size'],
    'dre_abnormal': ['digital_rectal_exam', 'DRE', 'prostate_palpation', 'digital_exam'],
    'trus_abnormal': ['transrectal_ultrasound', 'TRUS']
}


# ==========================================
# 2. Basic cleaning and utility functions
# ==========================================
def normalize_str(s):
    if pd.isna(s):
        return ""
    return re.sub(r'[\(\)\（\）\[\]\【\】\{\}\s]', '', str(s)).lower()


def get_category(name):
    norm_name = normalize_str(name)
    for key, patterns in FIELD_MAPPING.items():
        for p in patterns:
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


def parse_clinical_status(val):
    s = str(val)
    if any(kw in s for kw in ['positive', 'abnormal', 'nodule', 'mass', 'space_occupying_lesion', '1']):
        return 1
    return 0


# ==========================================
# 3. Huashan risk calculator core logic (RC1 & RC2)
# ==========================================
def calculate_huashan_risk(age, psa, f_t_ratio, pv, dre, trus):
    if psa <= 0 or pd.isna(psa):
        return None, {}

    log_psa = np.log(psa)
    pct_fpsa = f_t_ratio * 100

    # RC1 index computation
    rc1_pca_idx = (-7.552 + 0.05 * age + 1.636 * dre - 0.035 * pv + 3.675 * log_psa - 0.02 * pct_fpsa + 1.137 * trus)
    rc1_hg_idx = (-4.041 - 0.006 * age + 0.455 * dre + 1.325 * log_psa + 0.008 * pct_fpsa - 0.013 * pv + 1.513 * trus)

    # RC2 index computation
    rc2_pca_idx = (-7.983 + 0.051 * age + 2.333 * dre + 3.146 * log_psa - 0.047 * pct_fpsa)

    # Convert to probability
    prob_rc1_pca = 1 / (1 + np.exp(-rc1_pca_idx))
    prob_rc1_hg = 1 / (1 + np.exp(-rc1_hg_idx))
    prob_rc2_pca = 1 / (1 + np.exp(-rc2_pca_idx))

    calc_info = {
        'rc1_index': round(rc1_pca_idx, 3)
    }

    return (prob_rc1_pca, prob_rc1_hg, prob_rc2_pca), calc_info


# ==========================================
# 4. Main data-processing routine
# ==========================================
def process_data(base_path, item_path, output_path):
    print("--- Starting Huashan risk calculator (full output) ---")

    def read_csv_safe(path):
        for enc in ['gbk', 'utf-8-sig', 'utf-8']:
            try:
                return pd.read_csv(path, encoding=enc, low_memory=False)
            except:
                continue
        return pd.DataFrame()

    def get_first_valid(val_list):
        for v in val_list:
            num = clean_numeric(v)
            if not pd.isna(num):
                return num
        return np.nan

    base_df = read_csv_safe(base_path)
    item_df = read_csv_safe(item_path)

    if base_df.empty:
        print("Error: cannot read base.csv, or the file is empty.")
        return

    # 1. Extract indicators
    records = []
    if not item_df.empty:
        item_df['cat'] = item_df['res_itm_name'].apply(get_category)
        filtered = item_df.dropna(subset=['cat', 'res_value']).copy()

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            for (app_no, pat_id), group in filtered.groupby(['app_no', 'pat_id']):
                row = {'app_no': str(app_no).strip(), 'pat_id': str(pat_id).strip()}
                data_pool = {k: [] for k in FIELD_MAPPING.keys()}

                for _, item in group.iterrows():
                    data_pool[item['cat']].append(str(item['res_value']))

                row['psa_value'] = get_first_valid(data_pool.get('psa_value', []))
                row['f_t_ratio'] = get_first_valid(data_pool.get('fPSA/tPSA', []))
                row['pv_value'] = get_first_valid(data_pool.get('pv_value', []))

                dre_records = data_pool.get('dre_abnormal', [])
                row['dre_abnormal'] = 1 if any(parse_clinical_status(v) == 1 for v in dre_records) else (
                    0 if dre_records else np.nan)

                trus_records = data_pool.get('trus_abnormal', [])
                row['trus_abnormal'] = 1 if any(parse_clinical_status(v) == 1 for v in trus_records) else (
                    0 if trus_records else np.nan)

                records.append(row)

    wide_items = pd.DataFrame(records)

    # === Change 1: error-proof fallback ===
    # When records is empty (no patient indicators matched), create an empty table with base columns so later merges succeed
    if wide_items.empty:
        wide_items = pd.DataFrame(
            columns=['app_no', 'pat_id', 'psa_value', 'f_t_ratio', 'pv_value', 'dre_abnormal', 'trus_abnormal'])

    base_df['app_no'] = base_df['app_no'].astype(str).str.strip()
    base_df['pat_id'] = base_df['pat_id'].astype(str).str.strip()

    # === Change 2: use left join (how='left') ===
    # Ensure every patient in base.csv is retained even without matching items
    final = pd.merge(base_df, wide_items, on=['app_no', 'pat_id'], how='left')

    # 2. Evaluation logic and new fields
    results = []
    GAUGE_REF = "-10--2|-2-2|2-10"

    for _, r in final.iterrows():
        age = clean_numeric(r.get('app_pat_age'))
        psa = r.get('psa_value')
        f_t = r.get('f_t_ratio')
        pv = r.get('pv_value')
        dre = r.get('dre_abnormal')
        trus = r.get('trus_abnormal')

        # Check whether values are missing
        is_missing = pd.isna(age) or pd.isna(psa) or pd.isna(f_t) or pd.isna(pv) or pd.isna(dre) or pd.isna(trus)

        # === Change 3: output '-' when the result cannot be read ===
        if is_missing:
            risk_lvl, code, advice = '-', '-', '-'
            idx_display, p_rc1_display, p_rc1_hg_display, p_rc2_display = '-', '-', '-', '-'
            gauge_ref_display = '-'
        else:
            probs, calc_details = calculate_huashan_risk(age, psa, f_t, pv, dre, trus)

            if not probs:
                risk_lvl, code, advice = '-', '-', '-'
                idx_display, p_rc1_display, p_rc1_hg_display, p_rc2_display = '-', '-', '-', '-'
                gauge_ref_display = '-'
            else:
                p_rc1, p_rc1_hg, p_rc2 = probs
                idx = calc_details['rc1_index']

                if idx >= 2:
                    risk_lvl, code, advice = "high risk", "C61.2.3", "recommend biopsy"
                elif idx <= -2:
                    risk_lvl, code, advice = "low risk", "C61.2.1", "follow-up may be considered; avoid unnecessary biopsy for now"
                else:
                    risk_lvl, code, advice = "medium risk", "C61.2.2", "decide together with clinical presentation"

                idx_display = idx
                p_rc1_display = f"{p_rc1:.2%}"
                p_rc1_hg_display = f"{p_rc1_hg:.2%}"
                p_rc2_display = f"{p_rc2:.2%}"
                gauge_ref_display = GAUGE_REF

        # === Change 4: force 'missing' output when computed indicators are empty ===
        age_out = int(age) if pd.notna(age) else "missing"
        psa_out = f"{psa:.2f}" if pd.notna(psa) else "missing"
        ft_out = f"{f_t:.4f}" if pd.notna(f_t) else "missing"
        pv_out = f"{pv:.2f}" if pd.notna(pv) else "missing"
        dre_out = int(dre) if pd.notna(dre) else "missing"
        trus_out = int(trus) if pd.notna(trus) else "missing"

        results.append({
            'pat_id': r.get('pat_id'),
            'app_no': r.get('app_no'),
            'name': r.get('pat_name', 'unknown'),
            'sex': r.get('pat_sex', ''),
            'age': age_out,
            'total_PSA': psa_out,
            'free_ratio (f/t)': ft_out,
            'prostate_volume (PV)': pv_out,
            'abnormal_DRE': dre_out,
            'abnormal_TRUS': trus_out,
            'code': code,
            'risk_level': risk_lvl,
            'prediction_horizon': 'current',
            'gauge_reference': gauge_ref_display,
            'model_name': 'Prostate Cancer Risk Prediction (RC1)',
            'RC1_predicted_score': idx_display,
            'prostate_cancer_risk (RC1)': p_rc1_display,
            'high_grade_cancer_risk (RC1-HG)': p_rc1_hg_display,
            'baseline_clinical_risk (RC2)': p_rc2_display,
            'clinical_recommendation': advice
        })

    if results:
        res_df = pd.DataFrame(results)
        res_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"--- Processing complete: {len(res_df)} records (including missing), exported to {output_path} ---")
    else:
        print("--- no patient records found ---")


if __name__ == "__main__":
    process_data("base.csv", "item_result.csv", "huashan_rc_assessment_results.csv")
