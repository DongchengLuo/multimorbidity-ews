import pandas as pd
import numpy as np
import re
import math
from model_unit_utils import convert_measurement

# ==========================================
# 1. Field mapping
# ==========================================
FIELD_MAPPING = {
    'app_no': ['app_no', 'serial_number', 'outpatient_number'],
    'pat_id': ['pat_id', 'patient_id', 'id'],
    'pat_name': ['pat_name', 'name'],
    'age': ['app_pat_age', 'age', 'nl'],
    'gender': ['pat_sex', 'sex', 'gender'],
    'sbp': ['systolic_bp', 'systolic_bp (sbp)', 'sbp', 'sys_bp', 'systolic_bp, blood_pressure', 'blood_pressure', 'bp'],
    'tc': ['total_cholesterol', 'CHOL', 'tc', 'cholesterol'],
    'hdl': ['HDL', 'HDL-C', 'HDL_cholesterol'],
    'is_treated': ['antihypertensive_treatment', 'hypertension_treatment', 'past_history', 'is_treated', 'treated_hypertension'],
    'smoker': ['smoking_history', 'smoking_status', 'smoke', 'smoker', 'smoking'],
    'has_diabetes': ['diabetes', 'past_history', 'has_diabetes', 'diabetes_history']
}

# ==========================================
# 2. Model parameters and metadata
# ==========================================
MODEL_NAME = "Framingham Coronary Heart Disease (CHD)"
REFERENCE = "General Cardiovascular Risk Profile for Use in Primary Care: The Framingham Heart Study (Circulation, 2008)"
GAUGE_REF_STRING = "0%-100%"
GENERAL_CVD_GAUGE = "0%-6%|6%-20%|20%-100%"


# ==========================================
# 3. Data cleaning and parsing logic
# ==========================================
def normalize_str(s):
    if pd.isna(s):
        return ""
    return re.sub(r'[\(\)\（\）\[\]\【\】\{\}\s]', '', str(s)).lower()


def get_categories(name):
    norm_name = normalize_str(name)
    matched_keys = []
    for key, patterns in FIELD_MAPPING.items():
        for p in patterns:
            # Exact match
            if normalize_str(p) == norm_name:
                matched_keys.append(key)
    return matched_keys


def clean_numeric(val):
    if pd.isna(val) or str(val).strip() == "":
        return np.nan
    try:
        match = re.search(r"[-+]?\d*\.\d+|\d+", str(val))
        return float(match.group()) if match else np.nan
    except:
        return np.nan


def parse_gender(val):
    s = str(val).strip()
    if any(k in s for k in ['female', 'Female', 'F', '0', '2', 'Women']):
        return 'female'
    if any(k in s for k in ['male', 'Male', 'M', '1', 'Men']):
        return 'male'
    return np.nan


def parse_boolean_with_context(val, category):
    s = str(val).strip().lower()
    if pd.isna(val) or s == "":
        return np.nan

    if category == 'has_diabetes':
        if any(k in s for k in ['denies_diabetes', 'no_diabetes', 'no diabetes', 'false']):
            return 0
        if 'diabetes' in s:
            return 1
        return 0

    if category == 'is_treated':
        if any(k in s for k in ['no_antihypertensive', 'not_on_antihypertensive', 'not_receiving_antihypertensive', 'denies_antihypertensive_treatment', 'no treatment', 'untreated']):
            return 0
        if any(k in s for k in ['antihypertensive', 'hypertension_treatment', 'medication']):
            return 1
        if s in ['yes', 'y', '1', 'true']:
            return 1
        return 0

    if any(k in s for k in ['yes', 'y', '1', 'present', 'true']):
        return 1
    if any(k in s for k in ['no', 'n', '0', 'absent', 'false']):
        return 0
    return np.nan


def parse_bp_value(val):
    s = str(val).strip()
    if not s or s.lower() == 'nan':
        return np.nan
    if '/' in s:
        parts = s.split('/')
        if len(parts) >= 2:
            try:
                v1 = float(re.findall(r"[-+]?\d*\.\d+|\d+", parts[0])[0])
                v2 = float(re.findall(r"[-+]?\d*\.\d+|\d+", parts[1])[0])
                return max(v1, v2)
            except:
                pass
    try:
        match = re.search(r"[-+]?\d*\.\d+|\d+", s)
        if match:
            return float(match.group())
    except:
        pass
    return np.nan


# ==========================================
# 4. Core computation logic
# ==========================================
def calculate_framingham_points(gender, age, tc, hdl, sbp, is_treated, smoker, has_diabetes):
    points = 0
    if gender == 'female':
        if 30 <= age <= 34:
            points += 0
        elif 35 <= age <= 39:
            points += 2
        elif 40 <= age <= 44:
            points += 4
        elif 45 <= age <= 49:
            points += 5
        elif 50 <= age <= 54:
            points += 7
        elif 55 <= age <= 59:
            points += 8
        elif 60 <= age <= 64:
            points += 9
        elif 65 <= age <= 69:
            points += 10
        elif 70 <= age <= 74:
            points += 11
        elif age >= 75:
            points += 12
        if hdl >= 60:
            points += -2
        elif 50 <= hdl <= 59:
            points += -1
        elif 45 <= hdl <= 49:
            points += 0
        elif 35 <= hdl <= 44:
            points += 1
        elif hdl < 35:
            points += 2
        if tc < 160:
            points += 0
        elif 160 <= tc <= 199:
            points += 1
        elif 200 <= tc <= 239:
            points += 3
        elif 240 <= tc <= 279:
            points += 4
        elif tc >= 280:
            points += 5
        if not is_treated:
            if sbp < 120:
                points += -3
            elif 120 <= sbp <= 129:
                points += 0
            elif 130 <= sbp <= 139:
                points += 1
            elif 140 <= sbp <= 149:
                points += 2
            elif 150 <= sbp <= 159:
                points += 4
            elif sbp >= 160:
                points += 5
        else:
            if sbp < 120:
                points += -1
            elif 120 <= sbp <= 129:
                points += 2
            elif 130 <= sbp <= 139:
                points += 3
            elif 140 <= sbp <= 149:
                points += 4
            elif 150 <= sbp <= 159:
                points += 6
            elif sbp >= 160:
                points += 7
        if smoker:
            points += 3
        if has_diabetes:
            points += 4
    else:  # male
        if 30 <= age <= 34:
            points += 0
        elif 35 <= age <= 39:
            points += 2
        elif 40 <= age <= 44:
            points += 5
        elif 45 <= age <= 49:
            points += 6
        elif 50 <= age <= 54:
            points += 8
        elif 55 <= age <= 59:
            points += 10
        elif 60 <= age <= 64:
            points += 11
        elif 65 <= age <= 69:
            points += 12
        elif 70 <= age <= 74:
            points += 14
        elif age >= 75:
            points += 15
        if hdl >= 60:
            points += -2
        elif 50 <= hdl <= 59:
            points += -1
        elif 45 <= hdl <= 49:
            points += 0
        elif 35 <= hdl <= 44:
            points += 1
        elif hdl < 35:
            points += 2
        if tc < 160:
            points += 0
        elif 160 <= tc <= 199:
            points += 1
        elif 200 <= tc <= 239:
            points += 2
        elif 240 <= tc <= 279:
            points += 3
        elif tc >= 280:
            points += 4
        if not is_treated:
            if sbp < 120:
                points += -2
            elif 120 <= sbp <= 129:
                points += 0
            elif 130 <= sbp <= 139:
                points += 1
            elif 140 <= sbp <= 159:
                points += 2
            elif sbp >= 160:
                points += 3
        else:
            if sbp < 120:
                points += 0
            elif 120 <= sbp <= 129:
                points += 2
            elif 130 <= sbp <= 139:
                points += 3
            elif 140 <= sbp <= 159:
                points += 4
            elif sbp >= 160:
                points += 5
        if smoker:
            points += 4
        if has_diabetes:
            points += 3
    return points


def get_heart_age(gender, points):
    if gender == 'female':
        if points < 1:
            return "<30"
        mapping = {
            1: "31", 2: "34", 3: "36", 4: "39", 5: "42", 6: "45", 7: "48",
            8: "51", 9: "55", 10: "59", 11: "64", 12: "68", 13: "73", 14: "79"
        }
        return mapping.get(points, ">80")
    else:
        if points < 0:
            return "<30"
        mapping = {
            0: "30", 1: "32", 2: "34", 3: "36", 4: "38", 5: "40", 6: "42",
            7: "45", 8: "48", 9: "51", 10: "54", 11: "57", 12: "60",
            13: "64", 14: "68", 15: "72", 16: "76"
        }
        return mapping.get(points, ">80")


def calculate_chd_risk(age, gender, tc, hdl, sbp, is_treated, smoker, has_diabetes):
    # Change: return the placeholder "-" uniformly when anything is missing
    if any(pd.isna(x) for x in [age, gender, tc, hdl, sbp, is_treated, smoker, has_diabetes]) \
            or age <= 0 or tc <= 0 or hdl <= 0 or sbp <= 0:
        return {
            "cvd_risk": "-",
            "chd_risk": "-",
            "points": "-",
            "heart_age": "-",
            "risk_level": "-",
            "code": "-"
        }

    ln_age, ln_tc, ln_hdl, ln_sbp = math.log(age), math.log(tc), math.log(hdl), math.log(sbp)
    smoker_val = 1 if smoker else 0
    diabetes_val = 1 if has_diabetes else 0

    if gender == 'female':
        s10, mean_xb, calib_factor = 0.95012, 26.1931, 0.6086
        sum_beta_x = (
            (2.32888 * ln_age) +
            (1.20904 * ln_tc) +
            (-0.70833 * ln_hdl) +
            (0.52873 * smoker_val) +
            (0.69154 * diabetes_val)
        )
        sum_beta_x += (2.82263 * ln_sbp) if is_treated else (2.76157 * ln_sbp)
    else:
        s10, mean_xb, calib_factor = 0.88936, 23.9802, 0.7174
        sum_beta_x = (
            (3.06117 * ln_age) +
            (1.12370 * ln_tc) +
            (-0.93263 * ln_hdl) +
            (0.65451 * smoker_val) +
            (0.57367 * diabetes_val)
        )
        sum_beta_x += (1.99881 * ln_sbp) if is_treated else (1.93303 * ln_sbp)

    cvd_risk = 1 - pow(s10, math.exp(sum_beta_x - mean_xb))
    chd_risk = cvd_risk * calib_factor
    points = calculate_framingham_points(gender, age, tc, hdl, sbp, is_treated, smoker, has_diabetes)
    heart_age = get_heart_age(gender, points)

    if cvd_risk < 0.06:
        cvd_level = "overall_CVD_low_risk"
    elif cvd_risk < 0.20:
        cvd_level = "overall_CVD_medium_risk"
    else:
        cvd_level = "overall_CVD_high_risk"

    return {
        "cvd_risk": cvd_risk,
        "chd_risk": chd_risk,
        "points": points,
        "heart_age": heart_age,
        "risk_level": "continuous_CHD_risk",
        "cvd_level": cvd_level,
        "code": "-"
    }


# ==========================================
# 5. Main data-processing routine
# ==========================================
def process_data(base_path, item_path, output_path):
    print(f"--- Starting {MODEL_NAME} assessment ---")

    def read_csv_safe(path):
        for enc in ['gbk', 'utf-8-sig', 'utf-8']:
            try:
                return pd.read_csv(path, encoding=enc, low_memory=False, dtype=str)
            except:
                continue
        return pd.DataFrame()

    base_df = read_csv_safe(base_path)
    item_df = read_csv_safe(item_path)

    if base_df.empty:
        print("Error: base data file could not be read or is empty.")
        return

    # Clean the detail table
    if not item_df.empty:
        item_df['app_no'] = item_df['app_no'].astype(str).str.strip().str.lstrip('0')
        item_df['pat_id'] = item_df['pat_id'].astype(str).str.strip().str.lstrip('0')

    # 1. Wide-table processing of Item data
    records = []
    if not item_df.empty:
        item_df['cats'] = item_df['res_itm_name'].apply(get_categories)
        for (app_no, pat_id), group in item_df.groupby(['app_no', 'pat_id']):
            row = {'app_no': str(app_no).strip(), 'pat_id': str(pat_id).strip()}
            data_pool = {k: [] for k in FIELD_MAPPING.keys()}

            for _, item in group.iterrows():
                for cat in item['cats']:
                    data_pool[cat].append(item)

            def get_v(cat, parser):
                for item in data_pool.get(cat, []):
                    v = item.get('res_value')
                    res = parser(v, cat) if parser == parse_boolean_with_context else parser(v)
                    if pd.notna(res):
                        return res
                return np.nan

            def get_measurement(cat, indicator, target_unit, value_parser=clean_numeric):
                for item in data_pool.get(cat, []):
                    raw = item.get('res_value')
                    parsed = value_parser(raw)
                    result = convert_measurement(
                        parsed, item.get('res_unit', ''), indicator=indicator, target_unit=target_unit
                    )
                    if result.ok:
                        return result.value, raw, item.get('res_unit', ''), result.rule
                return np.nan, np.nan, '', 'no_recognizable_value_or_res_unit'

            row['sbp'], row['sbp_raw'], row['sbp_unit'], row['sbp_unit_note'] = get_measurement('sbp', 'blood_pressure', 'mmHg', parse_bp_value)
            row['tc'], row['tc_raw'], row['tc_unit'], row['tc_unit_note'] = get_measurement('tc', 'cholesterol', 'mg/dL')
            row['hdl'], row['hdl_raw'], row['hdl_unit'], row['hdl_unit_note'] = get_measurement('hdl', 'cholesterol', 'mg/dL')
            row['is_treated'] = get_v('is_treated', parse_boolean_with_context)
            row['smoker'] = get_v('smoker', parse_boolean_with_context)
            row['has_diabetes'] = get_v('has_diabetes', parse_boolean_with_context)
            records.append(row)

    wide_items = pd.DataFrame(records)

    # 2. Merge basic information with a left join
    base_df['app_no'] = base_df['app_no'].astype(str).str.strip().str.lstrip('0').replace('', '0')
    base_df['pat_id'] = base_df['pat_id'].astype(str).str.strip().str.lstrip('0').replace('', '0')

    base_parsed = []
    for _, r in base_df.iterrows():
        row_data = {}
        for field in FIELD_MAPPING.keys():
            for col in FIELD_MAPPING[field]:
                if col in r.index and pd.notna(r[col]):
                    row_data[field] = r[col]
                    break
        base_parsed.append(row_data)
    base_df = pd.DataFrame(base_parsed)

    # Change: force a left join with base_df as the base
    if not wide_items.empty:
        final = pd.merge(base_df, wide_items, on=['app_no', 'pat_id'], how='left', suffixes=('_base', '_item'))
    else:
        final = base_df.copy()

    # 3. Computation and output
    results = []

    # Formatting function; default to "-"
    def fmt_num(val):
        return "-" if pd.isna(val) else f"{val:.2f}"

    def fmt_int(val):
        return "-" if pd.isna(val) else f"{int(val)}"

    def fmt_bool(val):
        return "-" if pd.isna(val) else ("yes" if val else "no")

    for _, r in final.iterrows():
        def pick(k):
            if k in r.index and pd.notna(r[k]):
                return r[k]
            vi = r.get(f'{k}_item')
            vb = r.get(f'{k}_base')
            if pd.notna(vi):
                return vi
            if pd.notna(vb):
                return vb
            return np.nan

        app_no, pat_id, pat_name = pick('app_no'), pick('pat_id'), pick('pat_name')
        age = clean_numeric(pick('age'))
        gender = parse_gender(pick('gender'))
        sbp, tc, hdl = pick('sbp'), pick('tc'), pick('hdl')

        is_treated = pick('is_treated')
        smoker = pick('smoker')
        has_diabetes = pick('has_diabetes')

        # Compute risk (returns an all-"-" dict when data is missing)
        eval_res = calculate_chd_risk(age, gender, tc, hdl, sbp, is_treated, smoker, has_diabetes)

        cvd_risk_display = "-" if eval_res['cvd_risk'] == "-" else f"{eval_res['cvd_risk']:.2%}"
        chd_risk_display = "-" if eval_res['chd_risk'] == "-" else f"{eval_res['chd_risk']:.2%}"
        heart_age_display = "-" if eval_res['heart_age'] == "-" else f"{eval_res['heart_age']} years"
        points_display = "-" if eval_res['points'] == "-" else str(eval_res['points'])

        # Strictly restore output fields
        results.append({
            'pat_id': pat_id if pd.notna(pat_id) else "-",
            'app_no': app_no if pd.notna(app_no) else "-",
            'pat_name': pat_name if pd.notna(pat_name) else "-",
            'age': fmt_int(age),
            'gender': "-" if pd.isna(gender) else ('female' if gender == 'female' else 'male'),
            'sbp': fmt_num(sbp),
            'sbp_raw_unit': pick('sbp_unit') if pd.notna(pick('sbp_unit')) else '-',
            'tc': fmt_num(tc),
            'tc_raw_unit': pick('tc_unit') if pd.notna(pick('tc_unit')) else '-',
            'hdl': fmt_num(hdl),
            'hdl_raw_unit': pick('hdl_unit') if pd.notna(pick('hdl_unit')) else '-',
            'is_treated': fmt_bool(is_treated),
            'smoker': fmt_bool(smoker),
            'has_diabetes': fmt_bool(has_diabetes),
            'model_name': MODEL_NAME,
            'risk_probability': chd_risk_display,
            'prediction_horizon': 'next 10 years',
            'risk_level': eval_res['risk_level'],
            'overall_CVD_risk': "-" if eval_res['cvd_risk'] == "-" else cvd_risk_display,
            'overall_CVD_risk_tier': eval_res.get('cvd_level', '-'),
            'overall_CVD_tier_scale': GENERAL_CVD_GAUGE,
            'code': eval_res['code'],
            'heart_age': heart_age_display,
            'framingham_points': points_display,
            'gauge_reference': GAUGE_REF_STRING,
            'reference': REFERENCE
        })

    if results:
        pd.DataFrame(results).to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"--- Assessment complete: {len(results)} records processed. Results saved to: {output_path} ---")
    else:
        print("--- WARNING: no valid data records found ---")


if __name__ == "__main__":
    process_data("base.csv", "item_result.csv", "framingham_I25_result.csv")
