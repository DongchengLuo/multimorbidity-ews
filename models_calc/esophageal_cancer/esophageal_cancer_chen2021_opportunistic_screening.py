import pandas as pd
import numpy as np
import re
import math

# ==========================================
# 1. Field mapping (exact match)
# ==========================================
FIELD_MAPPING = {
    'age': ['app_pat_age', 'age'],
    'gender': ['pat_sex', 'sex'],
    'residential_area': ['residence', 'household_registration', 'rural', 'residential_area'],
    'BMI': ['(body mass index) BMI', 'bmi', 'BMI'],
    'height': ['height'],
    'weight': ['weight'],
    'smoke': ['smoking_history', 'smoking_status'],
    'cigarettes_per_day': ['daily_cigarettes', 'cigarettes_per_day'],
    'years_smoked': ['smoking_years', 'smoking_duration', 'years_smoked'],
    'pickled_food': ['salt_preserved_food', 'pickled_vegetables', 'salt_preserved_food_preference', 'pickled_food'],
    'tooth_loss': ['tooth_loss', 'missing_teeth_count'],
    'family_ec_history': ['esophageal_cancer_family_history', 'family_tumor_history', 'family_ec_history']
}

# ==========================================
# 2. Basic cleaning and parsing logic
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

# ==========================================
# 3. ESCC core scoring logic
# ==========================================
def calculate_escc_score(age, gender, residence_raw, bmi, pack_years, pickled_raw, tooth_raw, family_raw):
    """
    Core scoring: return None and "-" when a key parameter is NaN
    """
    # Check whether core computation items are missing
    params = [age, gender, residence_raw, bmi, pack_years, pickled_raw, tooth_raw, family_raw]
    if any(pd.isna(p) or str(p).strip() == "" for p in params):
        return None, "-", "-", "-"

    score = 0.0
    # 1. Age
    if 50 <= age <= 59: score += 4.0
    elif 60 <= age <= 69: score += 6.5
    elif age > 69: score += 9.5

    # 2. Sex
    if "male" in str(gender): score += 2.5

    # 3. Residence
    if any(kw in str(residence_raw) for kw in ['farming', 'township', 'village']): score += 1.0

    # 4. BMI
    if bmi <= 22: score += 1.5

    # 5. Smoking pack-years
    if pack_years > 30: score += 2.0
    elif pack_years > 0: score += 1.5

    # 6. Salt-preserved food
    if any(kw in str(pickled_raw) for kw in ['yes', 'often', 'high', '3']): score += 1.5

    # 7. Tooth loss
    tooth_num = clean_numeric(tooth_raw)
    is_tooth_risk = (tooth_num > 4) if not pd.isna(tooth_num) else ("yes" in str(tooth_raw))
    if is_tooth_risk: score += 1.5

    # 8. Family history
    if any(kw in str(family_raw) for kw in ['present', 'yes', 'esophagus']): score += 1.5

    # Risk tiering
    if score >= 9:
        return score, "high risk", "C15.1.3", "recommend gastroscopy"
    else:
        return score, "low risk", "C15.1.1", "regular follow-up"

# ==========================================
# 4. Main data-processing routine
# ==========================================
def process_data(base_path, item_path, output_path):
    print("--- Starting esophageal cancer risk model (keep-all mode) ---")

    def read_csv_safe(path):
        for enc in ['gbk', 'utf-8-sig', 'utf-8']:
            try: return pd.read_csv(path, encoding=enc, low_memory=False)
            except: continue
        return pd.DataFrame()

    base_df = read_csv_safe(base_path)
    item_df = read_csv_safe(item_path)

    if base_df.empty:
        print("error: failed to read base data")
        return

    # 1. Pivot indicator items to a wide table
    records = []
    if not item_df.empty:
        item_df['cat'] = item_df['res_itm_name'].apply(get_category)
        filtered = item_df.dropna(subset=['cat', 'res_value']).copy()

        for (app_no, pat_id), group in filtered.groupby(['app_no', 'pat_id']):
            row = {'app_no': str(app_no).strip(), 'pat_id': str(pat_id).strip()}
            data_pool = {k: [] for k in FIELD_MAPPING.keys()}
            for _, item in group.iterrows():
                data_pool[item['cat']].append(str(item['res_value']))

            # BMI extraction and computation
            bmi_val = clean_numeric(data_pool['BMI'][0]) if data_pool['BMI'] else np.nan
            if pd.isna(bmi_val):
                h = clean_numeric(data_pool['height'][0]) if data_pool['height'] else np.nan
                w = clean_numeric(data_pool['weight'][0]) if data_pool['weight'] else np.nan
                if h > 0 and w > 0:
                    h_m = h / 100.0 if h > 3 else h
                    bmi_val = w / (h_m * h_m)
            row['bmi_val'] = bmi_val

            # Pack-year logic
            smoke_hist = "".join(data_pool['smoke'])
            cigs = clean_numeric(data_pool['cigarettes_per_day'][0]) if data_pool['cigarettes_per_day'] else np.nan
            yrs = clean_numeric(data_pool['years_smoked'][0]) if data_pool['years_smoked'] else np.nan
            
            if any(kw in smoke_hist for kw in ["absent", "no"]):
                row['pack_years'] = 0.0
            elif not pd.isna(cigs) and not pd.isna(yrs):
                row['pack_years'] = (cigs / 20.0) * yrs
            else:
                row['pack_years'] = np.nan

            row['residence_raw'] = data_pool['residential_area'][0] if data_pool['residential_area'] else ""
            row['pickled_raw'] = data_pool['pickled_food'][0] if data_pool['pickled_food'] else ""
            row['tooth_raw'] = data_pool['tooth_loss'][0] if data_pool['tooth_loss'] else ""
            row['family_raw'] = data_pool['family_ec_history'][0] if data_pool['family_ec_history'] else ""
            records.append(row)

    wide_df = pd.DataFrame(records) if records else pd.DataFrame(columns=['app_no', 'pat_id'])

    # 2. Merge (left join)
    base_df['app_no'] = base_df['app_no'].astype(str).str.strip()
    base_df['pat_id'] = base_df['pat_id'].astype(str).str.strip()
    final = pd.merge(base_df, wide_df, on=['app_no', 'pat_id'], how='left')

    # 3. Loop evaluation
    results = []
    GAUGE_REF = "0-8|9-25"

    for _, r in final.iterrows():
        age = clean_numeric(r.get('app_pat_age'))
        gender = str(r.get('pat_sex', '')).strip()
        
        bmi = r.get('bmi_val')
        pack_years = r.get('pack_years')
        residence = r.get('residence_raw', '')
        pickled = r.get('pickled_raw', '')
        tooth = r.get('tooth_raw', '')
        family = r.get('family_raw', '')

        # Run scoring
        score, risk_level, risk_code, suggestion = calculate_escc_score(
            age, gender, residence, bmi, pack_years, pickled, tooth, family
        )

        def fmt_in(v, is_num=False):
            if pd.isna(v) or v == "": return "missing"
            return f"{v:.1f}" if is_num else str(v)

        results.append({
            'app_no': r.get('app_no'),
            'pat_id': r.get('pat_id'),
            'pat_name': r.get('pat_name', 'unknown'),
            # --- Raw indicators (missing shown as "missing") ---
            'age': int(age) if pd.notna(age) else "missing",
            'sex': gender if gender else "missing",
            'residence': fmt_in(residence),
            'BMI': fmt_in(bmi, True),
            'pack_years': fmt_in(pack_years, True),
            'salt_preserved_food': fmt_in(pickled),
            'dental_status': fmt_in(tooth),
            'family_history': fmt_in(family),
            # --- Model results (missing shown as "-") ---
            'predicted_score': score if score is not None else "-",
            'risk_level': risk_level,
            'code': risk_code,
            'gauge_reference': GAUGE_REF if score is not None else "-",
            'clinical_recommendation': suggestion,
            'model_name': 'esophageal squamous cell carcinoma risk model'
        })

    if results:
        pd.DataFrame(results).to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"--- Assessment complete: {len(results)} records processed to {output_path} ---")

if __name__ == "__main__":
    process_data("base.csv", "item_result.csv", "escc_risk_results.csv")
