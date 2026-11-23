import pandas as pd
import numpy as np
import re
import warnings

# Ignore all warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. Field mapping
# ==========================================
FIELD_MAPPING = {
    'age': ['app_pat_age', 'age'],
    'gender': ['pat_sex', 'sex'],
    'ebv_162215': ['EBV162215', 'EBV_162215_C', 'variant_162215'],
    'ebv_162476': ['EBV162476', 'EBV_162476_C', 'variant_162476'],
    'ebv_163364': ['EBV163364', 'EBV_163364_T', 'variant_163364'],
    'rs2860580': ['rs2860580', 'rs2860580_C', 'HLA-A_gene'],
    'rs2894207': ['rs2894207', 'rs2894207_T', 'HLA-B/C_gene'],
    'smoke': ['smoking_history', 'smoking_status', 'smoke'],
    'salted_fish': ['salted_fish_intake_frequency', 'salted_fish_consumption', 'salted_fish'],
    'education_level': ['education_level', 'education'],
    'npc_family_history': ['npc_family_history', 'respiratory_tumor_family_history']
}


# ==========================================
# 2. Basic cleaning logic (core change: add missing-value handling)
# ==========================================
def normalize_str(s):
    if pd.isna(s) or str(s).strip() == "": return None
    return re.sub(r'[\(\)\（\）\[\]\【\】\{\}\s]', '', str(s)).lower()


def get_category(name):
    norm_name = normalize_str(name)
    if not norm_name: return None
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


def parse_genetic_risk(val, risk_allele):
    """
    Parse genotype:
    - return np.nan (missing) when the raw data is empty
    - return 1 when a risk allele exists or the record is 1
    - otherwise return 0
    """
    s = normalize_str(val)
    if s is None:
        return np.nan  # key change: no longer default to 0

    if risk_allele.lower() in s or s in ['1', '1.0', 'yes', 'present']:
        return 1
    return 0


# ==========================================
# 3. Core scoring logic
# ==========================================
def calculate_npc_crs_score(ebv_data, host_data, epi_data):
    score = 0.0
    details = {}

    ebv_score = (ebv_data.get('162215', 0) * 2.155 +
                 ebv_data.get('162476', 0) * 2.204 +
                 ebv_data.get('163364', 0) * 1.892)
    score += ebv_score
    details['ebv_genetic_score'] = round(ebv_score, 3)

    host_score = (host_data.get('rs2860580', 0) * 0.582 +
                  host_data.get('rs2894207', 0) * 0.621)
    score += host_score
    details['host_genetic_score'] = round(host_score, 3)

    epi_score = (epi_data.get('family', 0) * 1.0 +
                 epi_data.get('fish', 0) * 0.4 +
                 epi_data.get('smoke', 0) * 0.2 +
                 epi_data.get('edu', 0) * -0.3)
    score += epi_score
    details['epidemiology_score'] = round(epi_score, 3)

    return round(score, 3), details


# ==========================================
# 4. Main data-processing routine
# ==========================================
def process_npc_screening(base_path, item_path, output_path):
    print("--- Starting NPC strict model (genetic-missing interception) ---")

    def read_csv_safe(path):
        for enc in ['gbk', 'utf-8-sig', 'utf-8']:
            try:
                return pd.read_csv(path, encoding=enc, low_memory=False)
            except:
                continue
        return pd.DataFrame()

    base_df = read_csv_safe(base_path)
    item_df = read_csv_safe(item_path)
    if base_df.empty or item_df.empty:
        print("Data file read failed")
        return

    item_df['cat'] = item_df['res_itm_name'].apply(get_category)
    filtered = item_df.dropna(subset=['cat', 'res_value']).copy()

    records = []
    for _, base_row in base_df.iterrows():
        app_no = str(base_row['app_no']).strip()
        pat_id = str(base_row['pat_id']).strip()

        row = {
            'app_no': app_no, 'pat_id': pat_id,
            'pat_name': base_row.get('pat_name', 'unknown'),
            'pat_sex': base_row.get('pat_sex', ''),
        }

        age_val = base_row['app_pat_age'] if 'app_pat_age' in base_df.columns else np.nan
        row['app_pat_age'] = clean_numeric(age_val)

        group = filtered[(filtered['app_no'] == app_no) & (filtered['pat_id'] == pat_id)]
        # Aggregate all indicator values for this patient
        data_pool = {k: " ".join([str(v) for v in vals]) for k, vals in group.groupby('cat')['res_value']}

        # Extract genetic indicators (if data_pool lacks the key, parse_genetic_risk receives None and returns np.nan)
        row['ebv_162215'] = parse_genetic_risk(data_pool.get('ebv_162215'), 'C')
        row['ebv_162476'] = parse_genetic_risk(data_pool.get('ebv_162476'), 'C')
        row['ebv_163364'] = parse_genetic_risk(data_pool.get('ebv_163364'), 'T')
        row['rs2860580'] = parse_genetic_risk(data_pool.get('rs2860580'), 'C')
        row['rs2894207'] = parse_genetic_risk(data_pool.get('rs2894207'), 'T')

        # Extract epidemiology (default 0 when missing, since such data is rarely complete; adjust per genetic logic if needed)
        row['family'] = 1 if any(kw in data_pool.get('npc_family_history', "") for kw in ['yes', 'present', '1']) else 0
        fish_str = data_pool.get('salted_fish', "")
        row['fish'] = 2 if "week" in fish_str else (1 if "month" in fish_str else 0)
        smoke_str = data_pool.get('smoke', "")
        row['smoke'] = 2 if "current_smoker" in smoke_str or "current" in smoke_str else (
            1 if "quit" in smoke_str or "former" in smoke_str else 0)
        edu_str = data_pool.get('education_level', "")
        row['edu'] = 3 if "primary_school" in edu_str or "not_exposed" in edu_str else (
            2 if "junior_high" in edu_str else (1 if "high school" in edu_str else 0))

        records.append(row)

    # --- Tiered score output ---
    results = []
    for r in records:
        # Check required fields: age + 5 genetic indicators
        genetic_values = [r['ebv_162215'], r['ebv_162476'], r['ebv_163364'], r['rs2860580'], r['rs2894207']]

        is_genetic_missing = any(pd.isna(v) for v in genetic_values)
        is_age_missing = pd.isna(r['app_pat_age'])

        if is_age_missing or is_genetic_missing:
            reason = []
            if is_age_missing: reason.append("age_missing")
            if is_genetic_missing: reason.append("genetic_variant_missing")

            score, ebv_score, risk, code_val, details_str = "-", "-", "-", "-", f"to_supplement: {','.join(reason)}"
            suggestion = "complete the basic information or genetic test report"
        else:
            score, details = calculate_npc_crs_score(
                {'162215': r['ebv_162215'], '162476': r['ebv_162476'], '163364': r['ebv_163364']},
                {'rs2860580': r['rs2860580'], 'rs2894207': r['rs2894207']},
                {'family': r['family'], 'fish': r['fish'], 'smoke': r['smoke'], 'edu': r['edu']}
            )
            ebv_score = details['ebv_genetic_score']
            details_str = str(details)

            if score >= 3.88:
                risk, code_val, suggestion = "high risk", "C11.1.3", "EBV serology screening and specialist consultation advised"
            else:
                risk, code_val, suggestion = "low risk", "C11.1.1", "regular_checkup"

        results.append({
            'pat_id': r['pat_id'],
            'pat_name': r['pat_name'],
            'sex': r['pat_sex'],
            'age': int(r['app_pat_age']) if not pd.isna(r['app_pat_age']) else "missing",
            'code': code_val,
            'risk_level': risk,
            'predicted_score': score,
            'recommendation': suggestion,
            'EBV_genetic_score': ebv_score,
            'score_details': details_str
        })

    res_df = pd.DataFrame(results)
    res_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"--- Assessment complete: {len(res_df)} processed, {len(res_df[res_df['code'] != '-'])} valid ---")


if __name__ == "__main__":
    process_npc_screening("base.csv", "item_result.csv", "npc_crs_screening_result.csv")
