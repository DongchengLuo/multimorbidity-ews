import pandas as pd
import numpy as np
import re

# ==========================================
# 1. Field mapping (for the Lancet Model 5 prediction model)
# ==========================================
FIELD_MAPPING = {
    # --- Required basic fields ---
    'age': ['app_pat_age', 'age'],
    'gender': ['pat_sex', 'sex'],

    # --- Pulmonary function indicators (Model 5 core indicators) ---
    'mmef_percent': ['MMEF%pred', 'MMEF_percent_predicted', 'maximal_mid_expiratory_flow_percent_predicted', 'MMEF(%)', 'mmef'],
    'fef50_percent': ['FEF50%pred', 'FEF50_percent_predicted', 'FEF50(%)', 'fef50'],

    # --- Environmental exposure indicators ---
    'biomass_exposure': ['indoor_biofuel', 'biofuel_exposure', 'cooking_fuel', 'biomass', 'firewood']
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
            # Exact text match
            if normalize_str(p) == norm_name: return key
    return None


def clean_numeric(val):
    if pd.isna(val) or str(val).strip() == "": return np.nan
    try:
        match = re.search(r"[-+]?\d*\.\d+|\d+", str(val))
        return float(match.group()) if match else np.nan
    except:
        return np.nan


def parse_binary(val):
    s = normalize_str(val)
    # Exact text match
    if s in ['yes', 'present', '1', 'true']:
        return 1
    return 0


# ==========================================
# 3. Lancet Model 5 prediction core logic (binary version)
# ==========================================
def calculate_model5_risk(age, sex_str, mmef_val, fef50_val, bio_fuel_raw):
    """
    Compute PI and classify binary (low vs high)
    Cut-off Value: 3.630
    """
    debug_info = {}

    # 1. Age
    if pd.isna(age):
        return np.nan, "age_missing", np.nan, np.nan, debug_info

    # 2. Sex (Male=1, Female=0)
    sex_norm = normalize_str(sex_str)
    if sex_norm in ['male', '1']:
        sex_val = 1
    elif sex_norm in ['female', '0']:
        sex_val = 0
    else:
        return np.nan, "sex_missing_or_invalid", np.nan, np.nan, debug_info
    debug_info['sex_val'] = sex_val

    # 3. MMEF (< 65% pred = 1)
    mmef_num = clean_numeric(mmef_val)
    if pd.isna(mmef_num):
        return np.nan, "MMEF_missing", np.nan, np.nan, debug_info

    low_mmef = 1 if mmef_num < 65 else 0
    debug_info['mmef_status'] = f"value={mmef_num}, Low={low_mmef}"

    # 4. FEF50 (< 65% pred = 1)
    fef50_num = clean_numeric(fef50_val)
    if pd.isna(fef50_num):
        return np.nan, "FEF50_missing", np.nan, np.nan, debug_info

    low_fef50 = 1 if fef50_num < 65 else 0
    debug_info['fef50_status'] = f"value={fef50_num}, Low={low_fef50}"

    # 5. Biofuel
    if not bio_fuel_raw or str(bio_fuel_raw).strip() == "":
        return np.nan, "biofuel_exposure_missing", np.nan, np.nan, debug_info

    biomass_val = parse_binary(bio_fuel_raw)
    debug_info['biomass_val'] = biomass_val

    # --- Compute PI ---
    pi = (0.0394 * age) + \
         (0.9206 * sex_val) + \
         (0.7552 * low_mmef) + \
         (0.4541 * low_fef50) + \
         (0.7169 * biomass_val)

    # --- Binary risk tiering and code mapping ---
    if pi >= 3.630:
        risk_label = "high risk"
        code = "J44.1.2"
    else:
        risk_label = "low risk"
        code = "J44.1.1"

    # --- Estimate 5-year incidence probability ---
    try:
        prob = 1 - np.power(0.9729, np.exp(pi))
        prob_str = f"{prob:.1%}"
    except:
        prob_str = "-"

    return pi, risk_label, prob_str, code, debug_info


# ==========================================
# 4. Main data-processing routine
# ==========================================
def process_data(base_path, item_path, output_path):
    print("--- Starting Lancet Pre-COPD prediction model (Model 5 - binary) ---")

    GAUGE_REF_STR = "0-3.63|3.63-8.0"

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
        print("Error: base data file is empty or unreadable")
        return

    # 1. Pivot to wide table
    records = []
    if not item_df.empty:
        item_df['cat'] = item_df['res_itm_name'].apply(get_category)
        filtered = item_df.dropna(subset=['cat', 'res_value']).copy()

        filtered['app_no'] = filtered['app_no'].astype(str).str.strip()
        filtered['pat_id'] = filtered['pat_id'].astype(str).str.strip()

        for (app_no, pat_id), group in filtered.groupby(['app_no', 'pat_id']):
            row = {'app_no': app_no, 'pat_id': pat_id}
            data_pool = {k: [] for k in FIELD_MAPPING.keys()}

            for _, item in group.iterrows():
                data_pool[item['cat']].append(str(item['res_value']))

            mmef_vals = [v for v in (clean_numeric(x) for x in data_pool.get('mmef_percent', [])) if not pd.isna(v)]
            row['mmef_val'] = mmef_vals[0] if mmef_vals else np.nan

            fef50_vals = [v for v in (clean_numeric(x) for x in data_pool.get('fef50_percent', [])) if not pd.isna(v)]
            row['fef50_val'] = fef50_vals[0] if fef50_vals else np.nan

            bio_fuel_vals = [x for x in data_pool.get('biomass_exposure', []) if pd.notna(x) and str(x).strip() != ""]
            row['bio_fuel_raw'] = bio_fuel_vals[0] if bio_fuel_vals else ""

            records.append(row)

    wide_items = pd.DataFrame(records)

    # 2. Merge basic information
    if 'app_no' in base_df.columns:
        base_df['app_no'] = base_df['app_no'].astype(str).str.strip()
    if 'pat_id' in base_df.columns:
        base_df['pat_id'] = base_df['pat_id'].astype(str).str.strip()

    if not wide_items.empty:
        final = pd.merge(base_df, wide_items, on=['app_no', 'pat_id'], how='left')
    else:
        final = base_df

    # 3. Computation and formatted output
    results = []
    print(f"computing {len(final)} records...")

    for _, r in final.iterrows():
        age = clean_numeric(r.get('app_pat_age'))
        sex = r.get('pat_sex', '')
        mmef = r.get('mmef_val', np.nan)
        fef50 = r.get('fef50_val', np.nan)
        bio_fuel_raw = r.get('bio_fuel_raw', '')

        # === Change 1: default logic when biofuel data is missing ===
        if not bio_fuel_raw or str(bio_fuel_raw).strip() == "":
            bio_fuel_for_calc = '0'  # value passed to the model (defaults to no exposure)
            bio_fuel_str = "missing (defaulted to none)"  # text shown in the output table
        else:
            bio_fuel_for_calc = bio_fuel_raw
            bio_fuel_str = bio_fuel_raw
            # ==========================================

        # Run model computation (pass the processed bio_fuel_for_calc)
        pi, risk_level, prob, code, details = calculate_model5_risk(
            age, sex, mmef, fef50, bio_fuel_for_calc
        )

        app_no_val = r.get('app_no', '')
        pat_id_val = r.get('pat_id', '')
        pat_name_val = r.get('pat_name', 'unknown')

        # === Change 2: formatted text for missing indicators ===
        age_str = int(age) if pd.notna(age) else "missing"
        sex_str = sex if sex else "missing"
        mmef_str = f"{mmef:.1f}" if pd.notna(mmef) else "missing"
        fef50_str = f"{fef50:.1f}" if pd.notna(fef50) else "missing"

        # === Change 3: output "-" when indicators are missing and computation is skipped ===
        if pd.isna(pi):
            pi_str = "-"
            prob_str = "-"
            risk_str = "-"
            code_str = "-"
            advice_str = "-"
        else:
            pi_str = f"{pi:.4f}"
            prob_str = prob
            risk_str = risk_level
            code_str = code
            advice_str = "recommend closer follow-up; monitor small-airway function" if risk_level == "high risk" else "regular_checkup"

        # Assemble the final result
        res_row = {
            'app_no': app_no_val,
            'pat_id': pat_id_val,
            'pat_name': pat_name_val,
            'sex': sex_str,
            'age': age_str,
            'MMEF_percent_predicted (%)': mmef_str,
            'FEF50_percent_predicted (%)': fef50_str,
            'biofuel_exposure': bio_fuel_str,  # new column: show bio-environment inputs
            'prognostic_index (PI)': pi_str,
            'predicted_score': pi_str,
            'risk_probability': prob_str,
            'prediction_horizon': 'next 5 years',
            'risk_level': risk_str,
            'code': code_str,
            'gauge_reference': GAUGE_REF_STR,
            'recommendation': advice_str,
            'model_name': 'COPD incidence prediction model'
        }
        results.append(res_row)

    # 4. Export results
    if results:
        res_df = pd.DataFrame(results)
        try:
            res_df.to_csv(output_path, index=False, encoding='utf-8-sig')
            print(f"--- Assessment complete: {len(res_df)} results ---")
            print(f"File saved to: {output_path}")
        except Exception as e:
            print(f"Failed to export file: {e}")
    else:
        print("--- Assessment complete, but no valid results were generated ---")


if __name__ == "__main__":
    process_data(
        "base.csv",
        "item_result.csv",
        "lancet_pre_copd_prediction_result.csv"
    )