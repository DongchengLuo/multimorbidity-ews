import pandas as pd
import numpy as np
import re
import os

# ==========================================
# 1. Field-mapping configuration
# ==========================================
FIELD_MAPPING = {
    'age': ['app_pat_age', 'age'],
    'gender': ['pat_sex', 'sex'],
    'sbp': ['systolic_bp', 'blood_pressure', 'SBP'],
    'TC': ['total_cholesterol', 'CHOL', 'TC'],
    'has_diabetes': ['diabetes', 'past_history'],
    'smoke': ['smoking_history', 'smoking', 'smoke']
}


# ==========================================
# 2. Basic helper functions (data cleaning and extraction)
# ==========================================
def normalize_str(s):
    if pd.isna(s): return ""
    return re.sub(r'[\(\)\（\）\[\]\【\】\{\}\s]', '', str(s)).lower()


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
        match = re.search(r"[-+]?\d*\.\d+|\d+", str(val))
        return float(match.group()) if match else np.nan
    except:
        return np.nan


def parse_gender(val):
    s = str(val).strip().lower()
    if pd.isna(val) or s == "": return np.nan
    if any(x in s for x in ['female', '0', 'f', '2']): return 1
    if any(x in s for x in ['male', '1', 'm']): return 0
    return np.nan


def parse_bool_history(val, keywords):
    if pd.isna(val) or str(val).strip() == "":
        return 0, True
    combined = normalize_str(val)
    if any(neg in combined for neg in ['absent', 'no', 'not']):
        return 0, False
    if any(kw in combined for kw in keywords + ['yes', 'present', '1']):
        return 1, False
    return 0, False


# ==========================================
# 3. Globorisk lookup-table mapping logic
# ==========================================
def map_age_to_group(age):
    if 40 <= age <= 44: return '40-44'
    elif 45 <= age <= 49: return '45-49'
    elif 50 <= age <= 54: return '50-54'
    elif 55 <= age <= 59: return '55-59'
    elif 60 <= age <= 64: return '60-64'
    elif 65 <= age <= 69: return '65-69'
    elif 70 <= age <= 74: return '70-74'
    else: return None


def map_sbp_to_group(sbp):
    if sbp >= 170: return 180
    elif sbp >= 150: return 160
    elif sbp >= 130: return 140
    else: return 120


def map_tc_to_group(tc):
    tc_mmol = tc / 38.67 if tc > 30 else tc
    tc_round = round(tc_mmol)
    if tc_round < 3: return 3
    elif tc_round > 7: return 7
    return tc_round


def load_globorisk_chart(csv_path):
    if not os.path.exists(csv_path):
        return None
    try:
        df = pd.read_csv(csv_path)
        chart_dict = {}
        for _, row in df.iterrows():
            key = (
                int(row['gender']),
                int(row['diabetes']),
                int(row['smoke']),
                str(row['age_group']).strip()
            )
            sbp = int(row['sbp'])
            tc = int(row['tc'])
            risk = float(row['risk'])

            if key not in chart_dict:
                chart_dict[key] = {}
            if sbp not in chart_dict[key]:
                chart_dict[key][sbp] = {}
            chart_dict[key][sbp][tc] = risk
        return chart_dict
    except Exception as e:
        print(f"failed to read chart file: {e}")
        return None


def lookup_globorisk_10yr(chart_dict, age, is_female, sbp, tc, diabetes, smoke):
    if any(pd.isna(x) for x in [age, is_female, sbp, tc]):
        return None, "core indicators (age/sex/BP/total cholesterol) missing"

    age_group = map_age_to_group(age)
    if not age_group:
        return None, f"age ({int(age)} years) outside chart range (40-74)"

    sbp_group = map_sbp_to_group(sbp)
    tc_group = map_tc_to_group(tc)

    key = (int(is_female), int(diabetes), int(smoke), age_group)

    try:
        risk_prob = chart_dict[key][sbp_group][tc_group]
        return risk_prob, "chart lookup matched"
    except KeyError:
        return None, f"chart lacks the combination (SBP:{sbp_group}, TC:{tc_group})"


# ==========================================
# 4. Main data-processing routine
# ==========================================
def process_data(base_path, item_path, chart_path, output_path):
    print("--- Starting Globorisk CVD 10-year risk assessment (full base iteration) ---")

    # 1. Load official chart data
    chart_dict = load_globorisk_chart(chart_path)
    if not chart_dict:
        print(f"error: ensure a valid chart data file exists in the same directory [{chart_path}]")
        return

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
        print("Error: base table could not be read or is empty.")
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
        data_pool = {k: None for k in FIELD_MAPPING.keys()}
        if not filtered.empty:
            group = filtered[(filtered['app_no'] == app_no) & (filtered['pat_id'] == pat_id)]
            for _, item in group.iterrows():
                cat, val = item['cat'], str(item['res_value'])
                if data_pool[cat] is None:
                    data_pool[cat] = val

        # Parse each indicator
        pat_age = clean_numeric(data_pool['age']) if data_pool['age'] else clean_numeric(base_row.get('app_pat_age'))
        pat_sex = parse_gender(data_pool['gender']) if data_pool['gender'] else parse_gender(base_row.get('pat_sex'))
        sbp = clean_numeric(data_pool['sbp'])
        tc = clean_numeric(data_pool['TC'])

        dm_val, dm_missing = parse_bool_history(data_pool.get('has_diabetes'), ['diabetes'])
        smoke_val, smoke_missing = parse_bool_history(data_pool.get('smoke'), ['smoking', 'cigarette'])

        # Core lookup
        prob, status_msg = lookup_globorisk_10yr(chart_dict, pat_age, pat_sex, sbp, tc, dm_val, smoke_val)

        # Format output
        age_str = int(pat_age) if pd.notna(pat_age) else "missing"
        sex_str = ('female' if pat_sex == 1 else 'male') if pd.notna(pat_sex) else "missing"
        sbp_str = sbp if pd.notna(sbp) else "missing"
        tc_str = tc if pd.notna(tc) else "missing"
        dm_str = "missing (defaulted to no)" if dm_missing else ('yes' if dm_val else 'no')
        smoke_str = "missing (defaulted to no)" if smoke_missing else ('yes' if smoke_val else 'no')

        if prob is not None:
            prob_str = f"{prob}%"
            level_str = "high risk" if prob >= 10.0 else "low/medium risk"
            code_str = "I25.3.3" if prob >= 10.0 else "I25.3.1"
        else:
            prob_str = "-"
            level_str = "-"
            code_str = "-"

        results.append({
            'pat_id': pat_id,
            'app_no': app_no,
            'pat_name': base_row.get('pat_name', 'unknown'),
            'age': age_str,
            'sex': sex_str,
            'systolic_bp (mmHg)': sbp_str,
            'total_cholesterol': tc_str,
            'diabetes_history': dm_str,
            'smoking_history': smoke_str,
            'prediction_horizon': 'next 10 years',
            'risk_probability': prob_str,
            'risk_level': level_str,
            'code': code_str,
            'assessment_status': status_msg,
            'model_name': 'Globorisk China exact-match version'
        })

    # Export
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
        base_path="base.csv",
        item_path="item_result.csv",
        chart_path="globorisk_china_chart.csv",
        output_path="globorisk_cvd_10yr.csv"
    )
