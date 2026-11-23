import pandas as pd
import numpy as np
import math
import re

# ==========================================
# 1. Field mapping
# ==========================================
FIELD_MAPPING = {
    'age': ['app_pat_age'],
    'gender': ['pat_sex'],
    'ALB': ['albumin', 'ALB'],
    'TBIL': ['total_bilirubin', 'TBIL'],
    'plt': ['PLT', 'platelet_count'],
    'history_labels': ['past_history', 'past_medical_history', 'diagnosis'],
    'hbsag': ['HBsAg (immunoassay)', 'HBsAg', 'HBsAg (chemiluminescence)'],
    'hbv_dna': ['HBV DNA quantitative', 'HBV DNA (internal standard)'],
    'hcv_ab': ['anti_HCV', 'anti_HCV (chemiluminescence)'],
    'is_chb': ['chronic_hepatitis_b']
}


# ==========================================
# 2. Basic utility functions
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


def format_metric(val, precision=1):
    """Format indicator display: show 'missing' when absent, otherwise show with precision"""
    if pd.isna(val): return "missing"
    return f"{val:.{precision}f}"


def check_population_status(history_str, hbsag_raw, hbv_dna_raw, hcv_ab_raw):
    """Determine target-population status"""
    hist = str(history_str).upper()

    # 1. Exclusion
    if any(k in hist for k in ['liver_cancer', 'hepatocellular_carcinoma', 'HCC', 'liver_transplant']):
        return "excluded (liver cancer or transplant)"

    # 2. Summary of inclusion criteria
    found_evidence = False
    target_keywords = ['hepatitis_b', 'HBV', 'hepatitis_c', 'HCV', 'chronic_hepatitis', 'fatty_liver', 'NAFLD', 'alcoholic_liver_disease', 'cirrhosis']
    if any(k in hist for k in target_keywords): found_evidence = True

    if any(s in str(hbsag_raw) for s in ['positive', '+', 'pos']): found_evidence = True

    dna_num = clean_numeric(hbv_dna_raw)
    if not pd.isna(dna_num) and dna_num > 100: found_evidence = True

    if any(s in str(hcv_ab_raw) for s in ['positive', '+', 'pos']): found_evidence = True

    if found_evidence: return "meets"

    # If there is neither positive evidence nor history, mark as missing
    if hist.strip() == "" and not hbsag_raw and not hcv_ab_raw:
        return "undetermined (data missing)"

    return "not eligible (non-chronic liver disease)"


# ==========================================
# 3. aMAP risk scoring logic
# ==========================================
def calculate_amap_risk(age, sex_str, alb, tbil, plt, pop_status):
    if pop_status != "meets":
        return None  # do not compute for non-target population

    if any(pd.isna(x) for x in [age, alb, tbil, plt]) or pd.isna(sex_str):
        return None  # do not compute when core indicators are missing

    # Sex handling
    sex_val = 1 if 'male' in str(sex_str) or str(sex_str) == '1' else 0
    # Prevent log10(0) errors
    safe_tbil = tbil if tbil > 0 else 0.1

    albi_score = (math.log10(safe_tbil) * 0.66) + (alb * -0.085)
    amap_raw = (0.06 * age + 0.89 * sex_val + 0.48 * albi_score - 0.01 * plt + 7.4)
    amap_score = (amap_raw / 14.77) * 100

    if amap_score < 50:
        return {'score': amap_score, 'group': "low risk", 'prob': "0-0.8%", 'code': 'C22.1.1'}
    elif 50 <= amap_score <= 60:
        return {'score': amap_score, 'group': "medium risk", 'prob': "1.5-4.8%", 'code': 'C22.1.2'}
    else:
        return {'score': amap_score, 'group': "high risk", 'prob': "8.1-19.9%", 'code': 'C22.1.3'}


# ==========================================
# 4. Main data-processing routine
# ==========================================
def process_data(base_path, item_path, output_path):
    print("--- Starting aMAP liver cancer risk model (no-drop) ---")

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
        print("error: base table empty");
        return

    # 1. Pivot indicators to wide table
    wide_records = []
    if not item_df.empty:
        item_df['cat'] = item_df['res_itm_name'].apply(get_category)
        filtered = item_df.dropna(subset=['cat', 'res_value']).copy()

        for (app_no, pat_id), group in filtered.groupby(['app_no', 'pat_id']):
            data_pool = {k: [] for k in FIELD_MAPPING.keys()}
            for _, item in group.iterrows():
                data_pool[item['cat']].append(str(item['res_value']))

            wide_records.append({
                'app_no': str(app_no).strip(),
                'pat_id': str(pat_id).strip(),
                'ALB_val': next((clean_numeric(v) for v in data_pool['ALB'] if not pd.isna(clean_numeric(v))), np.nan),
                'TBIL_val': next((clean_numeric(v) for v in data_pool['TBIL'] if not pd.isna(clean_numeric(v))),
                                 np.nan),
                'plt_val': next((clean_numeric(v) for v in data_pool['plt'] if not pd.isna(clean_numeric(v))), np.nan),
                'hbsag_raw': " ".join(data_pool['hbsag']),
                'hbv_dna_raw': " ".join(data_pool['hbv_dna']),
                'hcv_ab_raw': " ".join(data_pool['hcv_ab']),
                'hist_raw': " ".join(data_pool['history_labels'])
            })

    wide_items = pd.DataFrame(wide_records)

    # 2. Core: left join so nobody in base_df is dropped
    base_df['app_no'] = base_df['app_no'].astype(str).str.strip()
    base_df['pat_id'] = base_df['pat_id'].astype(str).str.strip()

    if wide_items.empty:
        final = base_df.copy()
        # Fill missing wide-table columns
        for col in ['ALB_val', 'TBIL_val', 'plt_val', 'hbsag_raw', 'hbv_dna_raw', 'hcv_ab_raw', 'hist_raw']:
            final[col] = np.nan
    else:
        final = pd.merge(base_df, wide_items, on=['app_no', 'pat_id'], how='left')

    # 3. Loop computation
    results = []
    for _, r in final.iterrows():
        age = clean_numeric(r.get('app_pat_age'))
        sex = r.get('pat_sex', '')

        # Determine the population
        pop_status = check_population_status(
            str(r.get('hist_raw', '')),
            r.get('hbsag_raw', ''),
            r.get('hbv_dna_raw', ''),
            r.get('hcv_ab_raw', '')
        )

        # Run the model
        risk = calculate_amap_risk(age, sex, r.get('ALB_val'), r.get('TBIL_val'), r.get('plt_val'), pop_status)

        # Assemble output: strictly distinguish "missing" from "-"
        results.append({
            'pat_id': r['pat_id'],
            'app_no': r['app_no'],
            'name': r.get('pat_name', 'missing'),
            'sex': sex if pd.notna(sex) and str(sex).strip() != '' else 'missing',
            'age': int(age) if not pd.isna(age) else 'missing',
            'target_population_determination': pop_status,
            'ALB': format_metric(r.get('ALB_val')),
            'TBIL': format_metric(r.get('TBIL_val')),
            'PLT': format_metric(r.get('plt_val'), precision=0),
            'aMAP_score': f"{risk['score']:.2f}" if risk else "-",
            'risk_level': risk['group'] if risk else "-",
            'code': risk['code'] if risk else "-",
            '5y_cancer_probability': risk['prob'] if risk else "-",
            'model_name': 'Liver Cancer aMAP score'
        })

    # 4. Export
    res_df = pd.DataFrame(results)
    res_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"--- Assessment complete: {len(res_df)} records, exported to {output_path} ---")


if __name__ == "__main__":
    process_data("base.csv", "item_result.csv", "amap_hcc_final.csv")
