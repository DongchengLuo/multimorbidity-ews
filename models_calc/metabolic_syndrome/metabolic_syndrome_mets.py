import pandas as pd
import numpy as np
import re

# ==========================================
# 1. Field mapping (strictly per the precise field-matching reference table)
# ==========================================
FIELD_MAPPING = {
    'TG': ['triglyceride', 'TG'],
    'HDL-C': ['HDL_cholesterol', 'HDL', 'HDL-C'],
    'sbp': ['systolic_bp'],
    'dbp': ['diastolic_bp'],
    'bp': ['blood_pressure'],
    'FBG': ['GLU', 'fasting_glucose'],
    'wc': ['waist']
}


# ==========================================
# 2. Basic cleaning and exact-matching logic
# ==========================================
def get_category(name):
    """Force strict exact matching to prevent field contamination"""
    if pd.isna(name): return None
    s = str(name).strip()
    if not s: return None
    for key, patterns in FIELD_MAPPING.items():
        for pattern in patterns:
            if s == pattern:  # strictly equal
                return key
    return None


def standardize_value(val_str, unit_str, category):
    """
    Smart numeric extraction with unit conversion (unit inside res_value or in a separate res_unit field)
    """
    if pd.isna(val_str) or val_str is None or str(val_str).strip() == "":
        return np.nan

    val_str = str(val_str).strip()
    unit_str = str(unit_str).strip().lower() if pd.notna(unit_str) else ""

    # Use regex to separate the numeric part from a trailing unit
    match = re.search(r"([-+]?\d*\.\d+|\d+)\s*([a-zA-Z/%\u4e00-\u9fa5]+)?", val_str)
    if not match:
        return np.nan

    val = float(match.group(1))
    inline_unit = match.group(2)

    # Prefer the unit embedded in res_value when present
    if inline_unit:
        unit_str = inline_unit.lower()

    # --- Core indicator unit conversion logic ---
    if category == 'TG':
        # Common conversion: mg/dL -> mmol/L (divide by 88.57)
        if 'mg' in unit_str:
            val /= 88.57
    elif category == 'HDL-C':
        # Common conversion: mg/dL -> mmol/L (divide by 38.67)
        if 'mg' in unit_str:
            val /= 38.67
    elif category == 'FBG':
        # Common conversion: mg/dL -> mmol/L (divide by 18.015)
        if 'mg' in unit_str:
            val /= 18.015
    elif category in ['sbp', 'dbp']:
        # Common conversion: kPa -> mmHg (multiply by 7.5006)
        if 'kpa' in unit_str:
            val *= 7.5006

    return val


def parse_combined_bp(val_str, unit_str):
    """Handle blood pressure written in one field, e.g. 120/80 mmHg"""
    if pd.isna(val_str) or val_str is None or str(val_str).strip() == "":
        return np.nan, np.nan

    val_str = str(val_str).strip()
    unit_str = str(unit_str).strip().lower() if pd.notna(unit_str) else ""

    inline_unit_match = re.search(r"([a-zA-Z/]+)$", val_str)
    if inline_unit_match:
        unit_str = inline_unit_match.group(1).lower()

    nums = re.findall(r"[-+]?\d*\.\d+|\d+", val_str)
    if len(nums) >= 2:
        sbp = float(nums[0])
        dbp = float(nums[1])
        if 'kpa' in unit_str:
            sbp *= 7.5006
            dbp *= 7.5006
        return sbp, dbp
    return np.nan, np.nan


def parse_gender(val):
    if pd.isna(val) or val is None: return None
    s = str(val).strip()
    if s in ['female', '2']: return '2'
    if s in ['male', '1']: return '1'
    return None


def parse_ethnicity(val):
    if pd.isna(val) or val is None or str(val).strip() == "": return '1'
    s = str(val).strip()
    if s in ['Han', '1']: return '1'
    return '2'


# ==========================================
# 3. MetS core scoring logic (based on the 2023 CMEC model)
# ==========================================
def calculate_mets_score(age, gender, ethnicity, wc, tg, hdl, sbp, dbp, fbg):
    if any(pd.isna(x) for x in [age, wc, tg, hdl, sbp, dbp, fbg]) or gender is None or ethnicity is None:
        return "-", "-", "-"

    map_val = ((sbp - dbp) / 3.0) + dbp

    if ethnicity == '1':  # Han
        if gender == '1':  # male
            coeffs = (-2.9092, 0.0262, 0.3098, -0.944, 0.0097, 0.0745) if age < 60 else (-2.3741, 0.0264, 0.4933,
                                                                                         -0.999, 0.0054, 0.0821)
        else:  # female
            coeffs = (-2.4981, 0.0199, 0.5218, -0.8616, 0.0110, 0.1074) if age < 60 else (-0.5682, 0.0153, 0.4587,
                                                                                          -1.3567, 0.0036, 0.0688)
    else:  # ethnic minority
        if gender == '1':  # male
            coeffs = (-5.0941, 0.0427, 0.2038, -0.9073, 0.0172, 0.1101) if age < 60 else (-3.7784, 0.0399, 0.3234,
                                                                                          -1.0218, 0.0116, 0.0738)
        else:  # female
            coeffs = (-3.6762, 0.0189, 0.5237, -0.6562, 0.0147, 0.2064) if age < 60 else (-1.5436, 0.0212, 0.4698,
                                                                                          -1.3471, 0.0084, 0.0821)

    intc, b_wc, b_tg, b_hdl, b_map, b_fbg = coeffs

    score = (intc + (b_wc * wc) + (b_tg * tg) + (b_hdl * hdl) + (b_map * map_val) + (b_fbg * fbg))

    if score < -0.11:
        risk_level, code = "low risk", "E70.1.1"
    elif -0.11 <= score < 0.58:
        risk_level, code = "medium risk", "E70.1.2"
    else:
        risk_level, code = "high risk", "E70.1.3"

    return score, risk_level, code


# ==========================================
# 4. Main data-processing routine
# ==========================================
def process_mets_data(base_path, item_path, output_path):
    print("--- Starting MetS risk assessment (fully automated parsing) ---")

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
        print("error: cannot read base data file or it is empty.")
        return

    base_df['app_no'] = base_df['app_no'].astype(str).str.strip().str.lstrip('0')
    base_df['pat_id'] = base_df['pat_id'].astype(str).str.strip().str.lstrip('0')

    records = []
    if not item_df.empty:
        item_df['app_no'] = item_df['app_no'].astype(str).str.strip().str.lstrip('0')
        item_df['pat_id'] = item_df['pat_id'].astype(str).str.strip().str.lstrip('0')

        item_df['cat'] = item_df['res_itm_name'].apply(get_category)
        filtered = item_df.dropna(subset=['cat', 'res_value']).copy()

        # Check whether the raw table has a separate unit column
        has_unit_col = 'res_unit' in item_df.columns

        for (app_no, pat_id), group in filtered.groupby(['app_no', 'pat_id']):
            row = {'app_no': app_no, 'pat_id': pat_id}
            data_pool = {k: (None, None) for k in FIELD_MAPPING.keys()}

            for _, item in group.iterrows():
                cat = item['cat']
                if data_pool[cat] == (None, None):
                    v = item['res_value']
                    u = item['res_unit'] if has_unit_col else None
                    data_pool[cat] = (v, u)

            # Convert values and units
            row['wc'] = standardize_value(data_pool['wc'][0], data_pool['wc'][1], 'wc')
            row['TG'] = standardize_value(data_pool['TG'][0], data_pool['TG'][1], 'TG')
            row['HDL-C'] = standardize_value(data_pool['HDL-C'][0], data_pool['HDL-C'][1], 'HDL-C')
            row['FBG'] = standardize_value(data_pool['FBG'][0], data_pool['FBG'][1], 'FBG')

            sbp_val = standardize_value(data_pool['sbp'][0], data_pool['sbp'][1], 'sbp')
            dbp_val = standardize_value(data_pool['dbp'][0], data_pool['dbp'][1], 'dbp')
            bp_sbp, bp_dbp = parse_combined_bp(data_pool['bp'][0], data_pool['bp'][1])

            row['sbp'] = sbp_val if not pd.isna(sbp_val) else bp_sbp
            row['dbp'] = dbp_val if not pd.isna(dbp_val) else bp_dbp

            records.append(row)

    wide_items = pd.DataFrame(records) if records else pd.DataFrame(columns=['app_no', 'pat_id'])

    final = pd.merge(base_df, wide_items, on=['app_no', 'pat_id'], how='left')

    results = []
    GAUGE_REF = "-5.0 - -0.11 | -0.11 - 0.58 | 0.58 - 5.0"
    model_name = "MetS Metabolic Syndrome Prediction Model"

    def fmt_in(v, is_num=False):
        if pd.isna(v) or v is None or str(v).strip() == "": return "missing"
        return f"{v:.2f}" if is_num else str(v)

    for _, r in final.iterrows():

        # Read core patient data from base_df only
        age = standardize_value(r.get('app_pat_age'), None, 'age')  # reuse the number-extraction helper
        gender_raw = r.get('pat_sex')
        gender = parse_gender(gender_raw)
        ethnicity = parse_ethnicity(r.get('patient nation'))

        wc, tg, hdl, sbp, dbp, fbg = r.get('wc'), r.get('TG'), r.get('HDL-C'), r.get('sbp'), r.get('dbp'), r.get('FBG')

        score, risk_level, code = calculate_mets_score(
            age, gender, ethnicity, wc, tg, hdl, sbp, dbp, fbg
        )

        if score == "-":
            remark = "core physical or biochemical indicators missing; cannot compute"
            gauge = "-"
            suggestion = "-"
        else:
            remark = "data complete, computable"
            gauge = GAUGE_REF
            if risk_level == "high risk":
                suggestion = "strongly recommend intervening on metabolic syndrome risk factors"
            elif risk_level == "medium risk":
                suggestion = "lifestyle intervention and regular follow-up advised"
            else:
                suggestion = "current risk is low; maintain a healthy lifestyle"

        results.append({
            'pat_id': r.get('pat_id'),
            'app_no': r.get('app_no'),
            'pat_name': r.get('pat_name', 'unknown'),
            'model_name': model_name,
            'sex': gender_raw if pd.notna(gender_raw) else "missing",
            'age': int(age) if pd.notna(age) else "missing",
            'ethnicity': "Han" if ethnicity == '1' else "ethnic_minority",
            'waist (cm)': fmt_in(wc, True),
            'systolic_bp (mmHg)': fmt_in(sbp, True),
            'diastolic_bp (mmHg)': fmt_in(dbp, True),
            'FBG': fmt_in(fbg, True),
            'TG': fmt_in(tg, True),
            'HDL-C': fmt_in(hdl, True),
            'predicted_score': f"{score:.4f}" if score != "-" else "-",
            'risk_level': risk_level,
            'code': code if code != "-" else "E70.MISS",
            'gauge_reference': gauge,
            'recommendation': suggestion,
            'prediction_horizon': 'current' if score != "-" else "-",
            'assessment_note': remark
        })

    if results:
        res_df = pd.DataFrame(results)

        ordered_cols = [
            'pat_id', 'app_no', 'pat_name', 'model_name', 'code', 'gauge_reference',
            'predicted_score', 'risk_level', 'prediction_horizon', 'recommendation', 'assessment_note',
            'sex', 'age', 'ethnicity', 'waist (cm)', 'systolic_bp (mmHg)', 'diastolic_bp (mmHg)',
            'FBG', 'TG', 'HDL-C'
        ]
        final_cols = [c for c in ordered_cols if c in res_df.columns]

        res_df[final_cols].to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"--- Processing complete: {len(res_df)} records. ---")
    else:
        print("--- WARNING: no data generated. ---")


if __name__ == "__main__":
    process_mets_data("base.csv", "item_result.csv", "mets_assessment_result_perfect.csv")