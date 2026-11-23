import pandas as pd
import numpy as np
import re
from model_unit_utils import convert_measurement

# ==========================================
# 1. Field mapping (based on the Hisayama gastric cancer risk score)
# ==========================================
FIELD_MAPPING = {
    # --- Basic demographic information ---
    'age': ['app_pat_age', 'age'],
    'gender': ['pat_sex', 'sex'],
    'pat_name': ['pat_name', 'name'],  # explicitly add the name mapping

    # --- Laboratory indicators ---
    # H. pylori antibody
    'hp_positive': ['h_pylori_antibody', 'HP_antibody', 'h_pylori_serology', 'Hp', '13C', '14C'],

    # --- Pepsinogen I, II and ratio ---
    'pg_i': ['pepsinogen_I', 'pepsinogen_i', 'pg1', 'pgi', 'pg i'],
    'pg_ii': ['pepsinogen_II', 'pepsinogen_ii', 'pg2', 'pgii', 'pg ii'],
    'pgr': [' pepsinogen_I_II', 'pgi/pgii', 'pg1/pg2', 'pepsinogen_ratio', 'pg_ratio', 'pg1/2'],

    # HbA1c
    'hba1c': ['HbA1c', 'hba1c', 'glycated'],

    # --- Behavioral habits ---
    'smoke': ['smoking_history', 'smoking_status', 'current_smoker', 'smoke']
}

# Gauge reference ticks (low | medium | high | very high)
GAUGE_REF_STRING = "0-7|8-14"


# ==========================================
# 2. Basic cleaning and matching logic
# ==========================================
def normalize_str(s):
    """String normalization"""
    if pd.isna(s): return ""
    return re.sub(r'[\(\)\（\）\[\]\【\】\{\}\s]', '', str(s)).lower()


def get_category(name):
    """Exact-match the standard label from the field name"""
    norm_name = normalize_str(name)
    for key, patterns in FIELD_MAPPING.items():
        for p in patterns:
            if normalize_str(p) == norm_name: return key  # [change] exact match: use == instead of in
    return None


def clean_numeric(val):
    """Extract numeric value"""
    if pd.isna(val) or str(val).strip() == "": return np.nan
    try:
        match = re.search(r"[-+]?\d*\.\d+|\d+", str(val))
        return float(match.group()) if match else np.nan
    except:
        return np.nan


def parse_bool_status(val_list):
    """Parse boolean status (True/False/None)"""
    s = " ".join([str(v) for v in val_list]).lower()
    if not s or s.strip() == "": return None

    positive_keywords = ['positive', '+', 'yes', 'present', 'true', '1']
    negative_keywords = ['negative', '-', 'no', 'absent', 'false', '0']

    for kw in positive_keywords:
        if kw in s: return True
    for kw in negative_keywords:
        if kw in s: return False
    return None


# ==========================================
# 3. Gastric cancer risk scoring core logic (Hisayama Study)
# ==========================================
def calculate_gastric_risk_score(age, sex, hp_status, ag_status, hba1c, smoking):
    score = 0
    debug_info = {}

    # --- Data completeness check (do not compute when indicators are missing) ---
    if pd.isna(age) or sex is None or hba1c is None or pd.isna(hba1c) or smoking is None:
        return None, {"error": "basic_indicator_missing"}

    # Special case: atrophic gastritis and HP status
    if ag_status is True:
        pass
    elif ag_status is False:
        if hp_status is None: return None, {"error": "non-atrophic gastritis; clarify HP status"}
    else:
        return None, {"error": "atrophic gastritis status unknown"}

    # 1. Age
    if 40 <= age <= 49:
        s = 0
    elif 50 <= age <= 59:
        s = 2
    elif 60 <= age <= 69:
        s = 3
    elif age >= 70:
        s = 2
    else:
        s = 0
    score += s
    debug_info['age_score'] = s

    # 2. Sex
    is_male = str(sex).lower() in ['male', 'm', '1']
    s = 3 if is_male else 0
    score += s
    debug_info['sex_score'] = s

    # 3. HP & AG
    if ag_status is True:
        s = 5
    elif hp_status is True:
        s = 2
    else:
        s = 0
    score += s
    debug_info['hp_ag_score'] = s

    # 4. HbA1c
    s = 2 if hba1c >= 6.5 else 0
    score += s
    debug_info['hba1c_score'] = s

    # 5. Smoking
    s = 1 if smoking else 0
    score += s
    debug_info['smoke_score'] = s

    return score, debug_info


def get_risk_meta(score):
    if score is None: return "unknown", ""

    if score < 8:
        return "low risk", "C16.3.1"
    return "high risk", "C16.3.3"


# ==========================================
# 4. Main data-processing routine
# ==========================================
def process_data(base_path, item_path, output_path):
    print("--- Starting Hisayama gastric cancer risk score model ---")

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

    if item_df.empty:
        print("detail data is empty.")

    # 1. Pivot to wide table
    item_df['cat'] = item_df['res_itm_name'].apply(get_category)
    filtered = item_df.dropna(subset=['cat', 'res_value']).copy()

    records = []
    # Group by app_no and pat_id
    for (app_no, pat_id), group in filtered.groupby(['app_no', 'pat_id']):
        row = {'app_no': str(app_no).strip(), 'pat_id': str(pat_id).strip()}

        data_pool = {k: [] for k in FIELD_MAPPING.keys()}
        for _, item in group.iterrows():
            if item['cat']:
                data_pool[item['cat']].append(item)

        def values(category):
            return [str(item.get('res_value', '')) for item in data_pool.get(category, [])]

        def first_measurement(category, indicator, target_unit, allow_unitless=False):
            for item in data_pool.get(category, []):
                result = convert_measurement(
                    item.get('res_value'), item.get('res_unit', ''),
                    indicator=indicator, target_unit=target_unit,
                    allow_unitless=allow_unitless
                )
                if result.ok:
                    return result.value, item.get('res_value'), item.get('res_unit', ''), result.rule
            return np.nan, '-', '', 'no_recognizable_value_or_res_unit'

        row['hba1c'], hba1c_raw, hba1c_unit, hba1c_note = first_measurement('hba1c', 'percentage', '%')
        row['hp_positive_bool'] = parse_bool_status(values('hp_positive'))
        pg1_val, pg1_raw, pg1_unit, pg1_note = first_measurement('pg_i', 'pepsinogen', 'ng/mL')
        pg2_val, pg2_raw, pg2_unit, pg2_note = first_measurement('pg_ii', 'pepsinogen', 'ng/mL')
        pgr_val, pgr_raw, pgr_unit, pgr_note = first_measurement('pgr', 'ratio_value', 'ratio', allow_unitless=True)
        
        effective_pgr = pgr_val if not pd.isna(pgr_val) else (pg1_val / pg2_val if not pd.isna(pg1_val) and not pd.isna(pg2_val) and pg2_val > 0 else np.nan)
        
        if not pd.isna(pg1_val) and not pd.isna(effective_pgr):
            row['atrophic_bool'] = True if (pg1_val <= 70 and effective_pgr <= 3.0) else False
        else:
            row['atrophic_bool'] = None

        row['smoke_bool'] = parse_bool_status(values('smoke'))

        # Record raw values for traceability
        row['raw_hba1c'], row['hba1c_unit'] = str(hba1c_raw), hba1c_unit
        row['raw_hp'] = " ".join(values('hp_positive'))
        row['raw_pg1'], row['pg1_unit'] = str(pg1_raw), pg1_unit
        row['raw_pg2'], row['pg2_unit'] = str(pg2_raw), pg2_unit
        row['raw_pgr'], row['pgr_unit'] = str(pgr_raw), pgr_unit
        row['raw_smoke'] = " ".join(values('smoke'))
        row['unit_note'] = f"PGI[{pg1_note}]；PGII[{pg2_note}]；PGR[{pgr_note}]；HbA1c[{hba1c_note}]"

        records.append(row)

    wide_items = pd.DataFrame(records)

    # 2. Merge basic information
    base_df['app_no'] = base_df['app_no'].astype(str).str.strip() 
    base_df['pat_id'] = base_df['pat_id'].astype(str).str.strip()

    if wide_items.empty:
        print("no valid exam indicator data extracted.")
        wide_items = pd.DataFrame(columns=['app_no', 'pat_id'])

    final = pd.merge(base_df, wide_items, on=['app_no', 'pat_id'], how='left')

    # 3. Scoring and result generation
    results = []
    for _, r in final.iterrows():
        age = clean_numeric(r.get('app_pat_age'))
        sex = r.get('pat_sex', '')
        pat_name = r.get('pat_name', 'unknown')

        hp_status = r.get('hp_positive_bool')
        ag_status = r.get('atrophic_bool')
        hba1c = r.get('hba1c')
        smoking = r.get('smoke_bool')

        score, details = calculate_gastric_risk_score(
            age, sex, hp_status, ag_status, hba1c, smoking
        )

        if score is not None:
            risk_label, risk_code = get_risk_meta(score)
        else:
            score = '-'
            risk_label = '-'
            risk_code = '-'

        results.append({
            'pat_id': r['pat_id'],
            'app_no': r['app_no'],
            'pat_name': pat_name,
            'model_name': 'Hisayama Gastric Cancer Risk Score',
            'prediction_horizon': 'next 14 years',
            'code': risk_code,
            'gauge_reference': GAUGE_REF_STRING,
            'predicted_score': score,
            'risk_level': risk_label,
            'sex': sex,
            'age': int(age) if not pd.isna(age) else '-',
            'h_pylori_antibody': r.get('raw_hp', '-'),
            'pepsinogen_I': r.get('raw_pg1', '-'),
            'pepsinogen_II': r.get('raw_pg2', '-'),
            'pepsinogen_ratio': r.get('raw_pgr', '-'),
            'HbA1c': r.get('raw_hba1c', '-'),
            'PGI_raw_unit': r.get('pg1_unit', '-'),
            'PGII_raw_unit': r.get('pg2_unit', '-'),
            'PGR_raw_unit': r.get('pgr_unit', '-'),
            'HbA1c_raw_unit': r.get('hba1c_unit', '-'),
            'unit handling': r.get('unit_note', '-'),
            'smoking_history': r.get('raw_smoke', '-'),
        })

    # 4. Export files
    if results:
        res_df = pd.DataFrame(results)
        res_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"--- Assessment complete: {len(res_df)} results generated, exported to {output_path} ---")
    else:
        print(f"--- Assessment complete but no results (likely due to missing key indicators) ---")


if __name__ == "__main__":
    # Replace with the actual file path
    process_data(
        "base.csv",
        "item_result.csv",
        "gastric_cancer_risk_result.csv"
    )
