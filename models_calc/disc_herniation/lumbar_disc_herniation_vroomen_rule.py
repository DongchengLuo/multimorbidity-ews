import pandas as pd
import numpy as np
import re
import os

# ==========================================
# 1. Field mapping (for Vroomen's Prediction Rule)
# ==========================================
FIELD_MAPPING = {
    'age': ['app_pat_age', 'age'],
    'duration_days': ['disease_duration', 'pain_duration', 'days_since_onset', 'duration'],
    'paroxysmal_pain': ['paroxysmal_pain', 'pain_character', 'paroxysmal'],
    'pain_worse_in_leg': ['leg_pain_worse_than_back', 'leg_pain_severity', 'primary_pain_site'],
    'typical_dermatomal': ['dermatomal_distribution', 'typical_radicular_pain', 'radiating_pain_distribution'],
    'pain_worse_strain': ['worse_with_cough', 'worse_with_sneeze', 'worse_with_straining', 'worse_with_increased_abdominal_pressure'],
    'finger_floor_dist': ['finger_floor_distance', 'finger_to_floor_distance', 'FFD', 'finger_floor'],
    'paresis': ['muscle_weakness', 'lower_limb_weakness', 'paresis', 'foot_drop']
}


# ==========================================
# 2. Basic cleaning and logic utilities
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


def parse_boolean(val_list):
    if not val_list: return None
    text = str(val_list[0]).lower()  # take the first value
    if not text.strip() or text == "nan":
        return None
    true_keywords = ['yes', 'present', 'true', 'positive', '1', '+', 'right', 'left', 'bilateral']
    false_keywords = ['no', 'absent', 'false', 'negative', '0', '-', 'normal']
    for kw in true_keywords:
        if kw in text: return True
    for kw in false_keywords:
        if kw in text: return False
    return None


def get_first_numeric(val_list):
    """Take the first valid number directly; no averaging"""
    if not val_list: return np.nan
    return clean_numeric(val_list[0])


# ==========================================
# 3. Vroomen model core scoring logic
# ==========================================
def calculate_vroomen_prob(age, duration, is_paroxysmal, is_leg_worse, is_dermatomal, is_strain_worse, ffd, is_paresis):
    inputs = [age, duration, is_paroxysmal, is_leg_worse, is_dermatomal, is_strain_worse, ffd, is_paresis]

    # Mark as not computable when any core indicator is missing
    if any(pd.isna(x) or x is None for x in inputs):
        return None

    intercept = -3.511
    logit = intercept

    if 41 <= age <= 50:
        logit += np.log(1.8)
    elif age >= 51:
        logit += np.log(2.8)

    if 15 <= duration <= 30:
        logit += np.log(2.2)
    elif duration > 30:
        logit += np.log(0.8)

    if is_paroxysmal: logit += np.log(1.8)
    if is_leg_worse: logit += np.log(4.5)
    if is_dermatomal: logit += np.log(3.2)
    if is_strain_worse: logit += np.log(2.0)

    if 5 <= ffd <= 24:
        logit += np.log(1.4)
    elif ffd >= 25:
        logit += np.log(2.8)

    if is_paresis: logit += np.log(5.2)

    return 1 / (1 + np.exp(-logit))


# ==========================================
# 4. Main data-processing routine
# ==========================================
def process_data(base_path, item_path, output_path):
    print("--- Starting Vroomen model (full-output standard) ---")

    def read_csv_safe(path):
        if not os.path.exists(path):
            print(f"file not found: {path}")
            return pd.DataFrame()
        for enc in ['gbk', 'utf-8-sig', 'utf-8']:
            try:
                return pd.read_csv(path, encoding=enc, low_memory=False)
            except:
                continue
        return pd.DataFrame()

    base_df = read_csv_safe(base_path)
    item_df = read_csv_safe(item_path)

    if base_df.empty:
        print("error: base information table is empty.")
        return

    # Use app_no uniformly as the join key
    base_df['app_no'] = base_df['app_no'].astype(str).str.strip()
    base_df['pat_id'] = base_df['pat_id'].astype(str).str.strip()

    records = []
    
    # ================================================================
    # Core fix: only clean data when the table is non-empty and has the required columns
    # ================================================================
    if not item_df.empty and 'res_itm_name' in item_df.columns and 'res_value' in item_df.columns:
        item_df['cat'] = item_df['res_itm_name'].apply(get_category)
        filtered = item_df.dropna(subset=['cat', 'res_value']).copy()

        if not filtered.empty:
            for (app_no, pat_id), group in filtered.groupby(['app_no', 'pat_id']):
                # Use app_no as the dict key during extraction
                row = {'app_no': str(app_no).strip(), 'pat_id': str(pat_id).strip()}
                data_pool = {k: [] for k in FIELD_MAPPING.keys()}
                for _, item in group.iterrows():
                    data_pool[item['cat']].append(str(item['res_value']))

                row['duration_days'] = get_first_numeric(data_pool.get('duration_days', []))
                row['finger_floor_dist'] = get_first_numeric(data_pool.get('finger_floor_dist', []))
                row['paroxysmal_pain'] = parse_boolean(data_pool.get('paroxysmal_pain', []))
                row['pain_worse_in_leg'] = parse_boolean(data_pool.get('pain_worse_in_leg', []))
                row['typical_dermatomal'] = parse_boolean(data_pool.get('typical_dermatomal', []))
                row['pain_worse_strain'] = parse_boolean(data_pool.get('pain_worse_strain', []))
                row['paresis'] = parse_boolean(data_pool.get('paresis', []))
                records.append(row)

    wide_items = pd.DataFrame(records)

    # Fallback: initialize an empty table with standard headers when extraction is empty
    if wide_items.empty:
        wide_items = pd.DataFrame(columns=[
            'app_no', 'pat_id', 'duration_days', 'finger_floor_dist',
            'paroxysmal_pain', 'pain_worse_in_leg', 'typical_dermatomal',
            'pain_worse_strain', 'paresis'
        ])

    # Switch to a left join
    final = pd.merge(base_df, wide_items, on=['app_no', 'pat_id'], how='left')

    results = []
    GAUGE_REF = "0%-30%|30%-62.5%|62.5%-100%"

    for _, r in final.iterrows():
        age = clean_numeric(r.get('app_pat_age'))

        # Compute probability
        prob = calculate_vroomen_prob(
            age, r.get('duration_days'), r.get('paroxysmal_pain'),
            r.get('pain_worse_in_leg'), r.get('typical_dermatomal'),
            r.get('pain_worse_strain'), r.get('finger_floor_dist'), r.get('paresis')
        )

        # Handle missing results
        if prob is None:
            risk = "-"
            code = "-"
            prob_str = "-"
        else:
            prob_str = f"{prob * 100:.1f}%"
            if prob < 0.30:
                risk = "low risk"
                code = "M51.1.1"
            elif prob < 0.625:
                risk = "medium risk"
                code = "M51.1.2"
            else:
                risk = "high risk"
                code = "M51.1.3"

        p_name = r.get('pat_name') if not pd.isna(r.get('pat_name')) else r.get('name', 'unknown')

        # Format raw indicators separately: force "missing" for empty values and map booleans to "yes"/"no"
        age_out = int(age) if pd.notna(age) else "missing"
        duration_out = f"{r.get('duration_days'):.0f}" if pd.notna(r.get('duration_days')) else "missing"
        ffd_out = f"{r.get('finger_floor_dist'):.1f}" if pd.notna(r.get('finger_floor_dist')) else "missing"

        def fmt_bool(val):
            if pd.isna(val) or val is None: return "missing"
            return "yes" if val else "no"

        results.append({
            'model_name': 'Vroomen Rule (2002)',
            'pat_id': r['pat_id'],
            'app_no': r['app_no'],  # map back to the previously named app_no field
            'pat_name': p_name,
            'code': code,
            'risk_note': risk,
            'risk_probability': prob_str,
            'gauge_reference': GAUGE_REF,
            'sex': r.get('pat_sex', ''),

            # --- Separately split indicator columns, all filled precisely ---
            'age': age_out,
            'duration_days': duration_out,
            'finger_floor_distance (cm)': ffd_out,
            'leg_pain_worse_than_back': fmt_bool(r.get('pain_worse_in_leg')),
            'typical_dermatomal_distribution': fmt_bool(r.get('typical_dermatomal')),
            'worse_with_abdominal_pressure': fmt_bool(r.get('pain_worse_strain')),
            'paroxysmal_pain': fmt_bool(r.get('paroxysmal_pain')),
            'paresis': fmt_bool(r.get('paresis'))
        })

    if results:
        out_df = pd.DataFrame(results)
        # Specify column order
        cols = ['model_name', 'pat_id', 'app_no', 'pat_name', 'code', 'risk_note', 'risk_probability', 'gauge_reference', 'sex',
                'age',
                'duration_days', 'finger_floor_distance (cm)', 'leg_pain_worse_than_back', 'typical_dermatomal_distribution', 'worse_with_abdominal_pressure', 'paroxysmal_pain', 'paresis']

        final_cols = [c for c in cols if c in out_df.columns]
        out_df[final_cols].to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"--- Assessment complete, results exported to {output_path} ---")
        print(f"--- {len(results)} records processed (including those with missing indicators) ---")
    else:
        print("--- error: no valid results generated. ---")


if __name__ == "__main__":
    process_data("base.csv", "item_result.csv", "vroomen_M51_prediction_results.csv")
