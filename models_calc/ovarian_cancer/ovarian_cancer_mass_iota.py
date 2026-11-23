import pandas as pd
import numpy as np
import re

# ==========================================
# 1. Field mapping (see the precise field-matching reference table)
# ==========================================
FIELD_MAPPING = {
    # --- Basic fields ---
    'age': ['app_pat_age', 'age'],

    # --- Clinical indicators ---
    'ca125': ['CA-125'],

    # --- Ultrasound morphological indicators ---
    'lesion_diameter': ['max_lesion_diameter', 'max_mass_diameter', 'mass_diameter'],
    'solid_diameter': ['max_solid_component_diameter', 'solid_part_diameter'],
    'locules_gt_10': ['locules_gt_10', 'multilocular_gt_10'],
    'papillary_count': ['papillary_projections_count', 'papillary_count', 'papillary_projections'],
    'acoustic_shadows': ['acoustic_shadowing', 'has_acoustic_shadowing'],
    'ascites': ['ascites'],

    # --- Center type ---
    'is_oncology_center': ['hospital_type', 'center_type', 'is_cancer_center']
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


def parse_boolean(val):
    s = str(val)
    if any(kw in s for kw in ['yes', 'present', '1', 'y', 'positive']): return 1
    return 0


# ==========================================
# 3. IOTA ADNEX model core computation logic
# ==========================================
def calculate_adnex_score(age, ca125, lesion_d, solid_d, loc_gt_10, pap_count, shadows, ascites, oncology_center):
    # Log transform (the formula requires log2)
    log2_b = np.log2(max(ca125, 0.1))
    log2_c = np.log2(max(lesion_d, 0.1))
    ratio = solid_d / lesion_d if lesion_d > 0 else 0
    ratio_sq = ratio ** 2

    # z1-z4 linear predictor coefficients
    z1 = (-7.577663 + 0.004506 * age + 0.111642 * log2_b + 0.372046 * log2_c + 6.967853 * ratio - 5.65588 * ratio_sq
          + 1.375079 * loc_gt_10 + 0.604238 * pap_count - 2.04157 * shadows + 0.971061 * ascites + 0.953043 * oncology_center)

    z2 = (-12.276041 + 0.017260 * age + 0.197249 * log2_b + 0.873530 * log2_c + 9.583053 * ratio - 5.83319 * ratio_sq
          + 0.791873 * loc_gt_10 + 0.400369 * pap_count - 1.87763 * shadows + 0.452731 * ascites + 0.452484 * oncology_center)

    z3 = (-14.915830 + 0.051239 * age + 0.765456 * log2_b + 0.430477 * log2_c + 10.37696 * ratio - 5.70975 * ratio_sq
          + 0.273692 * loc_gt_10 + 0.389874 * pap_count - 2.35516 * shadows + 1.348408 * ascites + 0.459021 * oncology_center)

    z4 = (-11.909267 + 0.033601 * age + 0.276166 * log2_b + 0.449025 * log2_c + 6.644939 * ratio - 2.30330 * ratio_sq
          + 0.899980 * loc_gt_10 + 0.215645 * pap_count - 2.49845 * shadows + 1.636407 * ascites + 0.80887 * oncology_center)

    # Compute the denominator and per-type probabilities
    denom = 1 + np.exp(z1) + np.exp(z2) + np.exp(z3) + np.exp(z4)
    probs = {
        'benign': 1 / denom,
        'Borderline': np.exp(z1) / denom,
        'Stage I': np.exp(z2) / denom,
        'Stage II-IV': np.exp(z3) / denom,
        'Metastatic': np.exp(z4) / denom
    }

    total_malignant_risk = 1 - probs['benign']
    return total_malignant_risk, probs, {'z1': z1, 'z2': z2, 'z3': z3, 'z4': z4}


# ==========================================
# 4. Main data-processing routine
# ==========================================
def process_data(base_path, item_path, output_path):
    print("--- Starting IOTA ADNEX ovarian tumor model (column output) ---")

    # Safe read
    def read_csv_safe(path):
        for enc in ['gbk', 'utf-8-sig', 'utf-8']:
            try:
                return pd.read_csv(path, encoding=enc, low_memory=False)
            except:
                continue
        return pd.DataFrame()

    # Helper: safely extract the first valid number
    def get_first_valid(val_list, default_val=np.nan):
        for v in val_list:
            num = clean_numeric(v)
            if not pd.isna(num):
                return num
        return default_val

    base_df = read_csv_safe(base_path)
    item_df = read_csv_safe(item_path)

    if base_df.empty:
        print("Error: cannot read base.csv, or the file is empty.")
        return

    # 1. Indicator extraction and conversion
    records = []
    if not item_df.empty:
        item_df['cat'] = item_df['res_itm_name'].apply(get_category)
        filtered = item_df.dropna(subset=['cat']).copy()

        for (app_no, pat_id), group in filtered.groupby(['app_no', 'pat_id']):
            row = {'app_no': str(app_no).strip(), 'pat_id': str(pat_id).strip()}
            data_pool = {k: [] for k in FIELD_MAPPING.keys()}
            for _, itm in group.iterrows():
                data_pool[itm['cat']].append(itm['res_value'])

            row['ca125'] = get_first_valid(data_pool.get('ca125', []), np.nan)
            row['lesion_d'] = get_first_valid(data_pool.get('lesion_diameter', []), np.nan)

            row['solid_d'] = get_first_valid(data_pool.get('solid_diameter', []), 0)
            row['pap_count'] = get_first_valid(data_pool.get('papillary_count', []), 0)

            row['loc_gt_10'] = parse_boolean(data_pool['locules_gt_10'][0]) if data_pool.get('locules_gt_10') else 0
            row['shadows'] = parse_boolean(data_pool['acoustic_shadows'][0]) if data_pool.get('acoustic_shadows') else 0
            row['ascites'] = parse_boolean(data_pool['ascites'][0]) if data_pool.get('ascites') else 0
            row['is_oncology'] = parse_boolean(data_pool['is_oncology_center'][0]) if data_pool.get(
                'is_oncology_center') else 0

            records.append(row)

    wide_df = pd.DataFrame(records)

    # Error-proof fallback
    if wide_df.empty:
        wide_df = pd.DataFrame(columns=[
            'app_no', 'pat_id', 'ca125', 'lesion_d', 'solid_d',
            'pap_count', 'loc_gt_10', 'shadows', 'ascites', 'is_oncology'
        ])

    if 'app_no' in base_df.columns: base_df['app_no'] = base_df['app_no'].astype(str).str.strip()
    if 'pat_id' in base_df.columns: base_df['pat_id'] = base_df['pat_id'].astype(str).str.strip()

    # Left join
    final = pd.merge(base_df, wide_df, on=['app_no', 'pat_id'], how='left')

    # 2. Batch computation
    final_results = []
    GAUGE_REF_STR = "0-10|10-50|50-100"

    for _, r in final.iterrows():
        age = clean_numeric(r.get('app_pat_age'))
        ca125 = r.get('ca125')
        lesion_d = r.get('lesion_d')
        solid_d = r.get('solid_d')
        pap_count = r.get('pap_count')
        loc_gt_10 = r.get('loc_gt_10')
        shadows = r.get('shadows')
        ascites = r.get('ascites')
        is_oncology = r.get('is_oncology')

        # Output "-" when the result is empty
        if pd.isna(age) or pd.isna(ca125) or pd.isna(lesion_d):
            risk_percent_str = "-"
            risk_level = "-"
            risk_code = "-"
            probs_str = "-"
            z_vals_str = "-"
        else:
            mal_risk, all_probs, z_vals = calculate_adnex_score(
                age, ca125, lesion_d, solid_d, loc_gt_10,
                pap_count, shadows, ascites, is_oncology
            )

            risk_percent = mal_risk * 100
            risk_percent_str = f"{risk_percent:.2f}%"
            probs_str = str({k: f"{v * 100:.1f}%" for k, v in all_probs.items()})
            z_vals_str = str(z_vals)

            if risk_percent < 10:
                risk_level = "low risk"
                risk_code = "C56.1.1"
            elif risk_percent < 50:
                risk_level = "medium risk"
                risk_code = "C56.1.2"
            else:
                risk_level = "high risk"
                risk_code = "C56.1.3"

        # ==========================================
        # Format columns independently: force "missing" for missing values
        # ==========================================
        age_out = int(age) if pd.notna(age) else "missing"
        ca125_out = f"{ca125:.2f}" if pd.notna(ca125) else "missing"
        lesion_d_out = f"{lesion_d:.2f}" if pd.notna(lesion_d) else "missing"
        solid_d_out = f"{solid_d:.2f}" if pd.notna(solid_d) else "missing"
        pap_count_out = int(pap_count) if pd.notna(pap_count) else "missing"

        # Booleans report missing when empty and show 1/0 when present
        loc_gt_10_out = int(loc_gt_10) if pd.notna(loc_gt_10) else "missing"
        shadows_out = int(shadows) if pd.notna(shadows) else "missing"
        ascites_out = int(ascites) if pd.notna(ascites) else "missing"
        is_oncology_out = int(is_oncology) if pd.notna(is_oncology) else "missing"

        final_results.append({
            'pat_id': r.get('pat_id'),
            'app_no': r.get('app_no'),
            'name': r.get('pat_name', '-'),

            # --- All raw indicators used in the computation, one column each ---
            'age': age_out,
            'CA125_value': ca125_out,
            'max_lesion_diameter': lesion_d_out,
            'max_solid_diameter': solid_d_out,
            'papillary_projections': pap_count_out,
            'locules_gt_10 (yes=1)': loc_gt_10_out,
            'acoustic_shadowing (yes=1)': shadows_out,
            'ascites (yes=1)': ascites_out,
            'cancer_center (yes=1)': is_oncology_out,
            # ----------------------------------------------

            'code': risk_code,
            'gauge_reference': GAUGE_REF_STR,
            'model_name': 'Ovarian Cancer Mass Screening IOTA ADNEX',
            'risk_probability': risk_percent_str,
            'risk_level': risk_level,
            'per_type_probability_details': probs_str,
            'intermediate_variables (z)': z_vals_str
        })

    if final_results:
        res_df = pd.DataFrame(final_results)
        res_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"--- Processing complete: {len(res_df)} records (including missing indicators) -> {output_path} ---")


if __name__ == "__main__":
    process_data(
        "base.csv",
        "item_result.csv",
        "iota_adnex_evaluation_C56_results.csv"
    )
