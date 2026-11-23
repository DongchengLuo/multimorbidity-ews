import pandas as pd
import numpy as np
import re
import math

# ==========================================
# 1. Field mapping (strict exact match; breast density removed)
# ==========================================
FIELD_MAPPING = {
    'age': ['app_pat_age', 'age'],
    'race': ['race', 'ethnicity'],
    'family_hx': ['breast_cancer_family_history', 'family_history', 'family_hx'],
    'biopsy_hx': ['breast_biopsy_history', 'biopsy_history', 'biopsy_hx']
}

MODEL_NAME = "Breast Cancer Surveillance Consortium (BCSC) Density Model"
REFERENCE = "Tice JA, et al. (Ann Intern Med. 2008)"
GAUGE_REF = "0-1.0|1.0-2.49|2.5-100"  # dashboard percentage banding

# ==========================================
# 2. Basic cleaning and core logic
# ==========================================
INCIDENCE_COEFS = {
    'White': (-0.000007231, 0.001129372, -0.044330046, 0.520553050),
    'Black': (-0.000004654, 0.000689680, -0.023186530, 0.218993144),
    'Asian': (-0.000003314, 0.000421460, -0.009405533, 0.004344518),
    'Hispanic': (-0.000003928, 0.000585266, -0.020678768, 0.210555571)
}

MORTALITY_COEFS = {
    'White': (0.003695745, 0.08844949),
    'Black': (0.011685888, 0.077263338),
    'Asian': (0.01333534, 0.09479173),
    'Hispanic': (0.002435331, 0.091388373)
}

ADJUSTMENT_FACTORS = {
    'White': {35: 0.7182, 45: 0.7565, 55: 0.8246, 65: 0.9069, 75: 0.9134},
    'Black': {35: 0.7470, 45: 0.7810, 55: 0.8486, 65: 0.9201, 75: 0.9334},
    'Asian': {35: 0.6341, 45: 0.6742, 55: 0.7530, 65: 0.8866, 75: 0.9127},
    'Hispanic': {35: 0.7535, 45: 0.7950, 55: 0.8780, 65: 0.9615, 75: 0.9865}
}


def normalize_str(s):
    """Text cleaning for exact matching"""
    if pd.isna(s):
        return ""
    return re.sub(r'[\(\)\（\）\[\]\【\】\{\}\s]', '', str(s)).lower()


def get_category(name):
    """Exact field-name matching"""
    norm_name = normalize_str(name)
    for key, patterns in FIELD_MAPPING.items():
        for p in patterns:
            if normalize_str(p) == norm_name:
                return key
    return None


def clean_numeric(val):
    """Extract the numeric value without any default filling"""
    if pd.isna(val) or str(val).strip() == "":
        return np.nan
    try:
        match = re.search(r"[-+]?\d*\.\d+|\d+", str(val))
        return float(match.group()) if match else np.nan
    except:
        return np.nan


def get_model_race(display_race):
    """Map the exam system's ethnicity text to the coefficient classes supported by the BCSC model"""
    if pd.isna(display_race):
        return 'Asian'
    val = str(display_race)
    if 'white' in val or 'White' in val: return 'White'
    if 'black' in val or 'Black' in val: return 'Black'
    if 'hispanic' in val or 'Hispanic' in val: return 'Hispanic'
    return 'Asian'


def calculate_bcsc_risk(age, model_race, birads, fam_hx, biop_hx):
    """Core math for the BCSC 5-year risk computation"""
    a, b, c, d = INCIDENCE_COEFS[model_race]
    n, m = MORTALITY_COEFS[model_race]

    base_5yr_incidence = 0.0
    p_risk = 1.0

    for x in range(int(age), int(age) + 5):
        i_x = (a * (x ** 3) + b * (x ** 2) + c * x + d) / 100000.0
        d_x = (n * math.exp(m * x)) / 100000.0
        base_5yr_incidence += (i_x * p_risk)
        p_risk = p_risk - i_x - d_x

    if age < 45: bucket = 35
    elif age < 55: bucket = 45
    elif age < 65: bucket = 55
    elif age < 75: bucket = 65
    else: bucket = 75
    adj_factor = ADJUSTMENT_FACTORS[model_race][bucket]

    adjusted_incidence = base_5yr_incidence * adj_factor

    if age < 65:
        hr_dict = {1: 0.48107, 2: 1.0, 3: 1.5506, 4: 2.0119}
    else:
        hr_dict = {1: 0.65706, 2: 1.0, 3: 1.3881, 4: 1.4499}

    adjusted_incidence *= hr_dict.get(birads, np.nan)

    hx_fam_mult = 1.454 if fam_hx >= 1 else 0.938
    hx_biop_mult = 1.495 if biop_hx >= 1 else 0.906

    final_risk_decimal = adjusted_incidence * hx_fam_mult * hx_biop_mult
    return final_risk_decimal * 100


# ==========================================
# 3. Main data-processing routine
# ==========================================
def process_data(base_path, item_path, output_path):
    print(f"--- Starting {MODEL_NAME} ---")

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
        print("Base data file could not be read or is empty")
        return

    # -------- Extract breast density data from item_result.csv --------
    density_df = pd.DataFrame()
    if not item_df.empty:
        density_mask = item_df['res_itm_name'].str.contains('breast_density_birads', na=False)
        if density_mask.any():
            density_df = item_df[density_mask][['app_no', 'pat_id', 'res_value']].copy()
            density_df['app_no'] = density_df['app_no'].astype(str).str.strip()
            density_df['pat_id'] = density_df['pat_id'].astype(str).str.strip()
            density_df['birads'] = density_df['res_value'].apply(clean_numeric)
            density_df = density_df.drop_duplicates(subset=['app_no', 'pat_id'])
    # ------------------------------------------

    # 1. Indicator extraction (age, family history, biopsy history, race only)
    records = []
    if not item_df.empty:
        item_df['cat'] = item_df['res_itm_name'].apply(get_category)
        filtered = item_df.dropna(subset=['cat', 'res_value']).copy()

        for (app_no, pat_id), group in filtered.groupby(['app_no', 'pat_id']):
            row = {'app_no': str(app_no).strip(), 'pat_id': str(pat_id).strip()}
            data_pool = {k: [] for k in FIELD_MAPPING.keys()}

            for _, item in group.iterrows():
                data_pool[item['cat']].append(item['res_value'])

            row['age'] = clean_numeric(data_pool['age'][0]) if data_pool['age'] else np.nan
            row['fam_hx'] = clean_numeric(data_pool['family_hx'][0]) if data_pool['family_hx'] else np.nan
            row['biop_hx'] = clean_numeric(data_pool['biopsy_hx'][0]) if data_pool['biopsy_hx'] else np.nan

            raw_race = data_pool['race'][0] if data_pool['race'] else np.nan
            row['display_race'] = "Han" if pd.isna(raw_race) or str(raw_race).strip() == "" else raw_race
            row['model_race'] = get_model_race(row['display_race'])

            records.append(row)

    wide_items = pd.DataFrame(records)

    base_df['app_no'] = base_df['app_no'].astype(str).str.strip()
    base_df['pat_id'] = base_df['pat_id'].astype(str).str.strip()
    
    # Merge the base table with wide-table extraction results
    if not wide_items.empty:
        final = pd.merge(wide_items, base_df, on=['app_no', 'pat_id'], how='outer')
    else:
        final = base_df.copy()
        for col in ['age', 'fam_hx', 'biop_hx', 'display_race', 'model_race']:
            final[col] = np.nan
        final['display_race'] = "Han"
        final['model_race'] = "Asian"

    # Merge standalone density data (left join)
    if not density_df.empty:
        final = pd.merge(final, density_df[['app_no', 'pat_id', 'birads']], on=['app_no', 'pat_id'], how='left')
    else:
        final['birads'] = np.nan

    # 2. Scoring and result generation
    results = []
    for _, r in final.iterrows():
        age = r['age'] if pd.notna(r['age']) else clean_numeric(r.get('app_pat_age'))
        birads = r['birads'] # rely entirely on the density value in the separate file

        age_str = str(int(age)) if pd.notna(age) else "missing"
        birads_str = str(int(birads)) if pd.notna(birads) else "missing"
        fam_hx_str = str(int(r['fam_hx'])) if pd.notna(r['fam_hx']) else "missing"
        biop_hx_str = str(int(r['biop_hx'])) if pd.notna(r['biop_hx']) else "missing"

        # Do not compute when any core indicator is missing
        if any(pd.isna(v) for v in [age, birads, r['fam_hx'], r['biop_hx']]):
            abs_risk_pct = np.nan
        else:
            abs_risk_pct = calculate_bcsc_risk(age, r['model_race'], birads, r['fam_hx'], r['biop_hx'])

        if pd.isna(abs_risk_pct):
            risk_str = "-"
            risk_level = "-"
            code = "-"
            advice = "-"
        else:
            risk_str = f"{abs_risk_pct:.2f}%"
            if abs_risk_pct >= 2.5:
                risk_level, code = "high risk", "C50.5.3"
            elif abs_risk_pct >= 1.0:
                risk_level, code = "medium risk", "C50.5.2"
            else:
                risk_level, code = "low risk", "C50.5.1"
            advice = "recommend further imaging based on clinical indications" if "high" in risk_level else "recommend routine screening follow-up"

        results.append({
            'pat_id': r['pat_id'],
            'app_no': r['app_no'],
            'model_name': MODEL_NAME,
            'pat_name': r.get('pat_name', 'unknown'),
            'sex': r.get('pat_sex', ''),
            'age': age_str,
            'race': r['display_race'],
            'BI-RADS': birads_str,
            'family_history': fam_hx_str,
            'biopsy_history': biop_hx_str,
            'risk_probability': risk_str,
            'risk_level': risk_level,
            'code': code,
            'gauge_reference': GAUGE_REF,
            'recommendation': advice,
            'reference_basis': REFERENCE
        })

    # 3. Output
    if results:
        res_df = pd.DataFrame(results)
        # Drop completely empty records (e.g. base row only, no matched indicator)
        res_df = res_df.dropna(subset=['pat_id', 'app_no'])
        res_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"--- Assessment complete: {len(res_df)} records processed to {output_path} ---")
    else:
        print("--- Assessment complete: no valid cases ---")


if __name__ == "__main__":
    process_data(
        "base.csv",
        "item_result.csv",
        "bcsc_risk_final_C50.csv"
    )
