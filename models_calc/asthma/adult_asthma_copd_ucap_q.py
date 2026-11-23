import pandas as pd
import numpy as np
import re
import os

# ==========================================
# 1. Field mapping (for the UCAP-Q model)
# ==========================================
FIELD_MAPPING = {
    # --- Basic demographics ---
    'age': ['app_pat_age', 'age'],
    'gender': ['pat_sex', 'sex'],
    'name': ['pat_name', 'name'],

    # --- Smoking-related ---
    'smoke_status': ['smoking_status', 'smoker'],
    'cigarettes_per_day': ['daily_cigarettes', 'cigarettes_per_day', 'cigs_per_day'],
    'years_smoked': ['smoking_years', 'years_smoked'],

    # --- Exposures and medication ---
    'smoke_exposure': ['secondhand_smoke_exposure', 'passive_smoking', 'exposure_smoke'],
    'paint_fumes': ['occupational_exposure_chemicals', 'paint_or_smoke_exposure', 'occupational_exposure'],
    'salbutamol': ['salbutamol_use', 'ventolin_use', 'use_salbutamol'],

    # --- Symptom features ---
    'wheeze_morning': ['worse_wheezing_in_morning', 'morning_wheezing', 'morning_wheeze'],
    'attacks': ['severe_attack_count', 'chest_problem_frequency', 'attack_count'],

    # --- Scoring items ---
    'cough_score': ['cough_score'],
    'sleep_score': ['sleep_score', 'sleep_disturbed_by_lung_disease'],
    'chest_tightness': ['chest_tightness_score', 'chest_tightness'],

    # --- Quality of life / functional limitation ---
    'vigorous_limit': ['limited_by_vigorous_activity', 'vigorous_limit'],
    'tired_freq': ['fatigue_frequency', 'feeling_tired', 'tired_frequency'],
    'work_cut': ['reduced_work_due_to_emotion', 'work_cut_down']
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
    s = normalize_str(val)
    if s in ['1', 'yes', 'present', 'true']: return 1
    return 0


def parse_attacks(val):
    s = str(val).strip()
    if pd.isna(val) or s == "": return 0
    if ">3" in s or "greater_than_3" in s or "more_than_3" in s: return 4
    num = clean_numeric(s)
    if not pd.isna(num):
        if num > 3: return 4
        return int(num)
    return 0


def parse_frequency_level(val):
    s = str(val).strip()
    if pd.isna(val) or s == "": return 0
    if "always" in s or "all_of_the_time" in s or "all" in s.lower(): return 4
    if "most_of_the_time" in s or "often" in s or "most" in s.lower(): return 3
    if "sometimes" in s or "some" in s.lower(): return 2
    if "a_little" in s or "a_little_of_the_time" in s or "little" in s.lower(): return 1
    if "never" in s or "absent" in s or "none" in s.lower(): return 0
    num = clean_numeric(s)
    if not pd.isna(num): return min(4, int(num))
    return 0


def parse_limit_level(val):
    s = str(val).strip()
    if pd.isna(val) or s == "": return 0
    if "very_much" in s or "a_lot" in s or "lot" in s.lower(): return 2
    if "a_little" in s or "little" in s.lower(): return 1
    if "absent" in s or "no" in s or "no" in s.lower(): return 0
    num = clean_numeric(s)
    if not pd.isna(num): return min(2, int(num))
    return 0


# ==========================================
# 3. UCAP-Q core scoring logic
# ==========================================
def calculate_ucap_risk(age, pack_years, smoke_exposure, paint_fumes, salbutamol,
                        wheeze_morning, attacks, cough, sleep, chest_tightness,
                        vigorous_limit, tired, work_cut):
    # Coefficients
    beta_asthma = {
        'const': -9.53375, 'age': -0.0044, 'pack_years': 0.0022, 'exposure': -0.4176,
        'paint': 0.9238, 'salb': 0.4024, 'wheeze': 0.5882, 'attacks': 0.8841,
        'cough': 1.448, 'sleep': 0.1511, 'chest': 0.8968, 'limit': 0.0877,
        'tired': -0.2959, 'work': 1.0958
    }
    beta_copd = {
        'const': -1.9512, 'age': 0.0420, 'pack_years': 0.0308, 'exposure': 1.2363,
        'paint': 0.0116, 'salb': 0.8238, 'wheeze': 0.5788, 'attacks': -0.1735,
        'cough': 0.8255, 'sleep': 1.4280, 'chest': -0.1256, 'limit': 0.9037,
        'tired': 1.1871, 'work': 0.0011
    }

    l_asthma = (beta_asthma['const'] + beta_asthma['age'] * age +
                beta_asthma['pack_years'] * pack_years + beta_asthma['exposure'] * smoke_exposure +
                beta_asthma['paint'] * paint_fumes + beta_asthma['salb'] * salbutamol +
                beta_asthma['wheeze'] * wheeze_morning + beta_asthma['attacks'] * attacks +
                beta_asthma['cough'] * cough + beta_asthma['sleep'] * sleep +
                beta_asthma['chest'] * chest_tightness + beta_asthma['limit'] * vigorous_limit +
                beta_asthma['tired'] * tired + beta_asthma['work'] * work_cut)

    l_copd = (beta_copd['const'] + beta_copd['age'] * age +
              beta_copd['pack_years'] * pack_years + beta_copd['exposure'] * smoke_exposure +
              beta_copd['paint'] * paint_fumes + beta_copd['salb'] * salbutamol +
              beta_copd['wheeze'] * wheeze_morning + beta_copd['attacks'] * attacks +
              beta_copd['cough'] * cough + beta_copd['sleep'] * sleep +
              beta_copd['chest'] * chest_tightness + beta_copd['limit'] * vigorous_limit +
              beta_copd['tired'] * tired + beta_copd['work'] * work_cut)

    exp_asthma = np.exp(l_asthma)
    exp_copd = np.exp(l_copd)
    exp_none = 1.0

    sum_exp = exp_asthma + exp_copd + exp_none
    prob_asthma = exp_asthma / sum_exp
    prob_copd = exp_copd / sum_exp

    return prob_asthma, prob_copd


# ==========================================
# 4. Main data-processing routine
# ==========================================
def process_data(base_path, item_path, output_path):
    print("--- Starting UCAP-Q (asthma/COPD) model ---")

    def read_csv_safe(path):
        if not os.path.exists(path):
            return pd.DataFrame()
        for enc in ['gbk', 'utf-8-sig', 'utf-8']:
            try:
                return pd.read_csv(path, encoding=enc, low_memory=False)
            except:
                continue
        print(f"cannot read file or file not found: {path}")
        return pd.DataFrame()

    base_df = read_csv_safe(base_path)
    item_df = read_csv_safe(item_path)

    if base_df.empty:
        print("base data unreadable or empty (Base table must contain patients); aborting.")
        return

    # 1. Pivot to wide table
    records = []
    if not item_df.empty and 'res_itm_name' in item_df.columns:
        item_df['cat'] = item_df['res_itm_name'].apply(get_category)
        filtered = item_df.dropna(subset=['cat', 'res_value']).copy()

        for (app_no, pat_id), group in filtered.groupby(['app_no', 'pat_id']):
            row = {'app_no': str(app_no).strip(), 'pat_id': str(pat_id).strip()}
            data_pool = {k: [] for k in FIELD_MAPPING.keys()}

            for _, item in group.iterrows():
                data_pool[item['cat']].append(str(item['res_value']))

            # Extract values and handle averaging (guard against empty-array warnings)
            row['age_val'] = clean_numeric(" ".join(data_pool.get('age', [])))
            
            def get_mean_val(key):
                vals = [clean_numeric(v) for v in data_pool.get(key, [])]
                valid_vals = [v for v in vals if not pd.isna(v)]
                return np.mean(valid_vals) if valid_vals else np.nan

            cigs = get_mean_val('cigarettes_per_day')
            yrs = get_mean_val('years_smoked')
            smoke_status = " ".join(data_pool.get('smoke_status', []))

            if "absent" in smoke_status or "no" in smoke_status:
                row['pack_years'] = 0.0
            else:
                if pd.isna(cigs): cigs = 0
                if pd.isna(yrs): yrs = 0
                row['pack_years'] = (cigs / 20.0) * yrs

            row['smoke_exposure'] = parse_boolean(" ".join(data_pool.get('smoke_exposure', [])))
            row['paint_fumes'] = parse_boolean(" ".join(data_pool.get('paint_fumes', [])))
            row['salbutamol'] = parse_boolean(" ".join(data_pool.get('salbutamol', [])))
            row['wheeze_morning'] = parse_boolean(" ".join(data_pool.get('wheeze_morning', [])))
            row['attacks'] = parse_attacks(" ".join(data_pool.get('attacks', [])))
            row['vigorous_limit'] = parse_limit_level(" ".join(data_pool.get('vigorous_limit', [])))
            row['tired'] = parse_frequency_level(" ".join(data_pool.get('tired_freq', [])))
            row['work_cut'] = parse_frequency_level(" ".join(data_pool.get('work_cut', [])))
            
            row['cough'] = get_mean_val('cough_score') if not pd.isna(get_mean_val('cough_score')) else 0
            row['sleep'] = get_mean_val('sleep_score') if not pd.isna(get_mean_val('sleep_score')) else 0
            row['chest'] = get_mean_val('chest_tightness') if not pd.isna(get_mean_val('chest_tightness')) else 0

            records.append(row)

    # Flag whether questionnaire indicator data exists
    if records:
        wide_items = pd.DataFrame(records)
        wide_items['has_data'] = True
    else:
        wide_items = pd.DataFrame(columns=['app_no', 'pat_id', 'has_data'])

    # 2. Format the Base table for merging
    base_df['app_no'] = base_df['app_no'].astype(str).str.strip()
    base_df['pat_id'] = base_df['pat_id'].astype(str).str.strip()

    # Change: use a left join
    final = pd.merge(base_df, wide_items, on=['app_no', 'pat_id'], how='left')
    final['has_data'] = final['has_data'].fillna(False)

    # 3. Computation and export
    results = []
    GAUGE_REF_STR = "0-5%|6-29%|30-100%"

    for _, r in final.iterrows():
        age = clean_numeric(r.get('app_pat_age'))
        if pd.isna(age): age = r.get('age_val')

        # === Core decision: score when questionnaire data exists and age (required) is valid; otherwise output "-" ===
        if r['has_data'] and not pd.isna(age):
            prob_asthma, prob_copd = calculate_ucap_risk(
                age, r.get('pack_years', 0.0), r['smoke_exposure'], r['paint_fumes'],
                r['salbutamol'], r['wheeze_morning'], r['attacks'], r['cough'],
                r['sleep'], r['chest'], r['vigorous_limit'], r['tired'], r['work_cut']
            )

            max_prob = max(prob_asthma, prob_copd)
            max_prob_pct = max_prob * 100

            # Generate codes and risk tiers
            if max_prob_pct < 6:
                risk_code = 'J45.2.1'
                risk_desc = 'low risk'
                suggestion = "maintain_a_healthy_lifestyle"
            elif 6 <= max_prob_pct < 30:
                risk_code = 'J45.2.2'
                risk_desc = 'medium risk'
                suggestion = "suspected risk; consult a respiratory physician"
            else:
                risk_code = 'J45.2.3'
                risk_desc = 'high risk'
                suggestion = "high risk; strongly recommend pulmonary function testing"
            
            # Format output values
            age_out = int(age)
            pack_years_out = f"{r.get('pack_years', 0.0):.1f}"
            prob_asthma_out = f"{prob_asthma * 100:.2f}%"
            prob_copd_out = f"{prob_copd * 100:.2f}%"
            gauge_ref_out = GAUGE_REF_STR

        else:
            # Use "-" when data or a key factor (age) is missing
            age_out = int(age) if not pd.isna(age) else 'missing'
            pack_years_out = '-'
            prob_asthma_out = '-'
            prob_copd_out = '-'
            risk_code = '-'
            risk_desc = '-'
            suggestion = '-'
            gauge_ref_out = '-'

        results.append({
            'app_no': r.get('app_no', ''), 
            'pat_id': r.get('pat_id', ''), 
            'pat_name': r.get('pat_name', 'unknown'),
            'model_name': 'UCAP-Q',
            'sex': r.get('pat_sex', ''),
            'age': age_out,
            'pack_years': pack_years_out,
            'asthma_probability (%)': prob_asthma_out,
            'copd_probability (%)': prob_copd_out,
            'code': risk_code,
            'gauge_reference': gauge_ref_out, 
            'risk_level': risk_desc,
            'recommendation': suggestion
        })

    if results:
        res_df = pd.DataFrame(results)
        res_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"--- Assessment complete: {len(res_df)} results generated, exported to {output_path} ---")
        print(res_df[['pat_name', 'asthma_probability (%)', 'copd_probability (%)', 'code', 'risk_level']].head(5))
    else:
        print("--- Assessment complete, but no valid results were generated ---")


if __name__ == "__main__":
    process_data(
        "base.csv",
        "item_result.csv",
        "ucap_q_screening_result.csv"
    )
