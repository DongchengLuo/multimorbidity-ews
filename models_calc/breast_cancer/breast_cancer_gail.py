import pandas as pd
import numpy as np
import re
import math

# ==========================================
# 1. Field mapping (see the precise field-matching reference table)
# ==========================================
FIELD_MAPPING = {
    'age': ['app_pat_age', 'age'],
    'age_at_menarche': ['menarche_age', 'age_at_menarche'],
    'age_first_birth': ['age_at_first_birth', 'age_first_birth'],
    'family_bc_history': ['breast_cancer_family_history', 'first_degree_cancer_count', 'numrel'],
    'biopsy_count': ['breast_biopsy_count', 'biopsy_count', 'nbiops'],
    'followup_years': ['follow_up_years', 'prediction_years']
}

MODEL_NAME = "Breast Cancer Risk Assessment Model (Gail Model)"
REFERENCE = "Gail MH, et al. (J Natl Cancer Inst. 1989)"
GAUGE_REF = "0-5|5-15|15-100"

# ==========================================
# 2. Basic cleaning and core logic
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

def get_gail_codes(agemen, nbiops, ageflb, numrel):
    agemen_c = 0 if agemen >= 14 else (1 if 12 <= agemen <= 13 else 2)
    nbiops_c = 0 if nbiops == 0 else (1 if nbiops == 1 else 2)
    ageflb_c = 0 if ageflb < 20 else (1 if 20 <= ageflb <= 24 else (2 if ageflb <= 29 else 3))
    numrel_c = 0 if numrel == 0 else (1 if numrel == 1 else 2)
    return agemen_c, nbiops_c, ageflb_c, numrel_c

def calculate_gail_risk(age, agemen_c, nbiops_c, ageflb_c, numrel_c, t=30):
    def get_rr(current_age):
        age_cat = 1 if current_age >= 50 else 0
        lor = (0.09401 * agemen_c + 0.52926 * nbiops_c + 0.21863 * ageflb_c + 0.95830 * numrel_c +
               0.01081 * age_cat - 0.28804 * (nbiops_c * age_cat) - 0.19081 * (ageflb_c * numrel_c))
        return math.exp(lor)

    rr_init = get_rr(age)
    rr_later = get_rr(50) if age < 50 and (age + t) > 50 else rr_init

    grid_40_30 = {(2.0, 2.0): 8.6, (5.0, 2.0): 11.9, (2.0, 5.0): 17.0}
    p00, p10, p01 = grid_40_30[(2.0, 2.0)], grid_40_30[(5.0, 2.0)], grid_40_30[(2.0, 5.0)]

    abs_risk = p00 + (rr_init - 2.0) * (p10 - p00) / 3.0 + (rr_later - 2.0) * (p01 - p00) / 3.0
    return rr_init, abs_risk

# ==========================================
# 3. Main data-processing routine
# ==========================================
def process_data(base_path, item_path, output_path):
    print(f"--- Starting {MODEL_NAME} (full base iteration) ---")

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
        print("Data file could not be read or is empty")
        return

    base_df['app_no'] = base_df['app_no'].astype(str).str.strip()
    base_df['pat_id'] = base_df['pat_id'].astype(str).str.strip()

    # Parse item data
    filtered = pd.DataFrame()
    if not item_df.empty:
        item_df['app_no'] = item_df['app_no'].astype(str).str.strip()
        item_df['pat_id'] = item_df['pat_id'].astype(str).str.strip()
        item_df['cat'] = item_df['res_itm_name'].apply(get_category)
        filtered = item_df.dropna(subset=['cat', 'res_value']).copy()

    # Iterate over each patient in base_df
    results = []
    for _, base_row in base_df.iterrows():
        app_no = str(base_row['app_no']).strip()
        pat_id = str(base_row['pat_id']).strip()

        # Extract this patient's indicator data
        data_pool = {k: [] for k in FIELD_MAPPING.keys()}
        if not filtered.empty:
            group = filtered[(filtered['app_no'] == app_no) & (filtered['pat_id'] == pat_id)]
            for _, item in group.iterrows():
                data_pool[item['cat']].append(item['res_value'])

        # Extract each indicator
        agemen = clean_numeric(data_pool['age_at_menarche'][0]) if data_pool['age_at_menarche'] else np.nan
        ageflb = clean_numeric(data_pool['age_first_birth'][0]) if data_pool['age_first_birth'] else np.nan
        nbiops = clean_numeric(data_pool['biopsy_count'][0]) if data_pool['biopsy_count'] else 0.0
        numrel = clean_numeric(data_pool['family_bc_history'][0]) if data_pool['family_bc_history'] else 0.0
        t = clean_numeric(data_pool['followup_years'][0]) if data_pool['followup_years'] else 30

        age = clean_numeric(base_row.get('app_pat_age'))

        # Attempt computation - requires age, menarche age and age at first birth
        can_calculate = not any(pd.isna(v) for v in [age, agemen, ageflb])

        if can_calculate:
            agemen_c, nbiops_c, ageflb_c, numrel_c = get_gail_codes(agemen, nbiops, ageflb, numrel)
            rr, abs_risk_pct = calculate_gail_risk(age, agemen_c, nbiops_c, ageflb_c, numrel_c, t=t)

            if abs_risk_pct > 15.0:
                risk_level, code = "high risk", "C50.2.3"
            elif abs_risk_pct >= 5.0:
                risk_level, code = "medium risk", "C50.2.2"
            else:
                risk_level, code = "low risk", "C50.2.1"

            results.append({
                'pat_id': pat_id,
                'app_no': app_no,
                'model_name': MODEL_NAME,
                'pat_name': base_row.get('pat_name', 'unknown'),
                'sex': base_row.get('pat_sex', ''),
                'age': int(age),
                'indicator_details': f"menarche:{agemen}, biopsy:{nbiops}, age_at_birth:{ageflb}, relatives:{numrel}",
                'relative_risk (RR)': round(rr, 2),
                f'{int(t)}-year absolute risk': f"{abs_risk_pct:.2f}%",
                'risk_level': risk_level,
                'code': code,
                'gauge_reference': GAUGE_REF,
                'recommendation': "recommend intensified screening" if risk_level != "low risk" else "regular follow-up",
                'reference_basis': REFERENCE
            })
        else:
            results.append({
                'pat_id': pat_id,
                'app_no': app_no,
                'model_name': MODEL_NAME,
                'pat_name': base_row.get('pat_name', 'unknown'),
                'sex': base_row.get('pat_sex', ''),
                'age': '-',
                'indicator_details': '-',
                'relative_risk (RR)': '-',
                '30-year absolute risk': '-',
                'risk_level': '-',
                'code': '-',
                'gauge_reference': GAUGE_REF,
                'recommendation': '-',
                'reference_basis': REFERENCE
            })

    if results:
        res_df = pd.DataFrame(results)
        res_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        valid_count = len(res_df[res_df['code'] != '-'])
        print(f"--- Assessment complete: {len(res_df)} records processed ---")
        print(f"--- Valid computed records: {valid_count} ---")
        print(f"--- Records not computable due to missing values: {len(res_df) - valid_count} ---")
        print(f"--- Results saved to: {output_path} ---")

if __name__ == "__main__":
    process_data(
        "base.csv",
        "item_result.csv",
        "gail_risk_final_C50.csv"
    )
