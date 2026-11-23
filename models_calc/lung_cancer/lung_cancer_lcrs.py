import pandas as pd
import numpy as np
import re
import math

# ==========================================
# 1. Field mapping
# ==========================================
FIELD_MAPPING = {
    'age': ['app_pat_age', 'age'],
    'gender': ['pat_sex', 'sex'],
    'BMI': ['(body mass index) BMI', 'bmi', 'BMI'],
    'height': ['height'],
    'weight': ['weight'],
    'smoke': ['smoking_history', 'smoking_status'],
    'cigarettes_per_day': ['daily_cigarettes', 'cigarettes_per_day', 'cpd'],
    'years_smoked': ['smoking_years'],
    'has_chronic_bronchitis': ['has_chronic_bronchitis', 'chronic_bronchitis', 'emphysema', 'past_history'],
    'residential_area': ['residence', 'household_type'],
    'frequent_cough': ['frequent_cough'],
    'previous_cancer': ['prior_cancer_history', 'tumor_history'],
    'family_cancer_history': ['two_or_more_first_degree_cancer', 'family_cancer_history', 'first_degree_cancer'],
    'edu': ['education_level', 'education'],
    'inhale': ['smoke_inhaled', 'smoke_inhalation'],
    'years_quit': ['years_since_quit'],
    'exercise': ['regular_exercise', 'physical_inactivity', 'exercise_status']
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
            if normalize_str(p) in norm_name: return key
    return None


def clean_numeric(val):
    if pd.isna(val) or str(val).strip() == "": return np.nan
    try:
        match = re.search(r"[-+]?\d*\.\d+|\d+", str(val))
        return float(match.group()) if match else np.nan
    except:
        return np.nan


def parse_binary(val):
    if pd.isna(val) or str(val).strip() == "": return np.nan
    s = normalize_str(val)
    if any(kw in s for kw in ['yes', 'present', '1', 'urban', 'meets']): return 1
    if any(kw in s for kw in ['no', 'absent', '0', 'rural', 'not_met']): return 0
    return np.nan


def parse_edu(val):
    if pd.isna(val) or str(val).strip() == "": return np.nan
    s = normalize_str(val)
    if any(kw in s for kw in ['junior_high', 'primary_school', 'illiterate', 'vocational_school']): return 2
    if 'high school' in s: return 1
    if any(kw in s for kw in ['college', 'bachelor', 'postgraduate', 'university']): return 0
    return np.nan


# ==========================================
# 3. LCRS core scoring logic
# ==========================================
def calculate_lcrs_score(data):
    smoke = data.get('smoke')
    age = data.get('age')

    if pd.isna(smoke) or pd.isna(age):
        return None, "missing core indicator: [age] or [smoking_status]"

    score = 0.0
    debug_info = {}

    essential_common = ['height', 'BMI', 'residential_area', 'frequent_cough',
                        'has_chronic_bronchitis', 'previous_cancer', 'family_cancer_history']
    for field in essential_common:
        if pd.isna(data.get(field)):
            return None, f"missing common core indicator: [{field}]"

    if smoke == 1:
        model_name = "LCRS-EverSmoker"
        missing_smoker_fields = [f for f in ['edu', 'cigarettes_per_day', 'years_smoked', 'inhale', 'years_quit'] if
                                 pd.isna(data.get(f))]
        if missing_smoker_fields:
            return None, f"smoker missing detail indicators: {','.join(missing_smoker_fields)}"

        if 40 <= age <= 44:
            s = 25.4
        elif 45 <= age <= 49:
            s = 46.5
        elif 50 <= age <= 54:
            s = 71.1
        elif 55 <= age <= 59:
            s = 86.6
        elif 60 <= age <= 64:
            s = 99.2
        elif 65 <= age <= 69:
            s = 114.8
        elif 70 <= age <= 80:
            s = 120.9
        else:
            s = 0
        score += s
        debug_info['age_s'] = s

        if data['residential_area'] == 1: score += 19.0
        if data['edu'] == 2:
            score += 20.7
        elif data['edu'] == 1:
            score += 17.9

        h = data['height']
        if 160 <= h <= 164:
            score += 1.0
        elif 165 <= h <= 169:
            score += 5.8
        elif h >= 170:
            score += 6.1

        bmi = data['BMI']
        if 18.5 <= bmi <= 23.9:
            score += 9.1
        elif bmi < 18.5:
            score += 25.1

        if data['frequent_cough'] == 1: score += 12.8
        if data['has_chronic_bronchitis'] == 1: score += 15.1
        if data['previous_cancer'] == 1: score += 26.8
        if data['family_cancer_history'] == 1: score += 13.6

        cpd = data['cigarettes_per_day']
        if 10 <= cpd <= 14:
            score += 5.5
        elif 15 <= cpd <= 19:
            score += 10.5
        elif 20 <= cpd <= 24:
            score += 18.3
        elif 25 <= cpd <= 29:
            score += 24.8
        elif 30 <= cpd <= 34:
            score += 26.5
        elif cpd >= 35:
            score += 28.9

        yrs_s = data['years_smoked']
        if 10 <= yrs_s <= 19:
            score += 15.7
        elif 20 <= yrs_s <= 29:
            score += 22.5
        elif 30 <= yrs_s <= 39:
            score += 35.1
        elif 40 <= yrs_s <= 49:
            score += 47.5
        elif yrs_s >= 50:
            score += 54.5

        if data['inhale'] == 1: score += 10.0
        if data['years_quit'] <= 5 and yrs_s > 0: score += 8.5

        beta, mean_lcrs, cutoff = 0.022, 157.3, 166.2
        s0_10y = 0.9877333
    else:
        model_name = "LCRS-NeverSmoker"
        if pd.isna(data.get('exercise')):
            return None, "non-smoker missing detail indicator: [exercise]"

        if 40 <= age <= 44:
            s = 6.6
        elif 45 <= age <= 49:
            s = 9.7
        elif 50 <= age <= 54:
            s = 13.1
        elif 55 <= age <= 59:
            s = 15.5
        elif 60 <= age <= 64:
            s = 19.4
        elif 65 <= age <= 69:
            s = 21.1
        elif 70 <= age <= 80:
            s = 21.4
        else:
            s = 0
        score += s
        debug_info['age_s'] = s

        if data['residential_area'] == 1: score += 1.4

        h = data['height']
        if 150 <= h <= 154:
            score += 1.0
        elif 155 <= h <= 159:
            score += 1.2
        elif h >= 160:
            score += 1.9

        bmi = data['BMI']
        if 18.5 <= bmi <= 23.9:
            score += 1.3
        elif bmi < 18.5:
            score += 3.0

        if data['exercise'] == 1: score += 1.0
        if data['frequent_cough'] == 1: score += 1.5
        if data['has_chronic_bronchitis'] == 1: score += 3.2
        if data['previous_cancer'] == 1: score += 6.5
        if data['family_cancer_history'] == 1: score += 2.3

        beta, mean_lcrs, cutoff = 0.122, 14.7, 21.2
        s0_10y = 0.9948929

    prob = (1 - math.pow(s0_10y, math.exp(beta * (score - mean_lcrs)))) * 100
    prob_cutoff = (1 - math.pow(s0_10y, math.exp(beta * (cutoff - mean_lcrs)))) * 100
    gauge_ref = f"0%-{prob_cutoff:.2f}%|{prob_cutoff:.2f}%-2.00%|2.00%-100%"

    if score < cutoff:
        risk_level, code = "low risk", "C34.2.1"
    else:
        if prob < 2.0:
            risk_level, code = "medium risk", "C34.2.2"
        else:
            risk_level, code = "high risk", "C34.2.3"

    return (score, prob, risk_level, code, model_name, debug_info, gauge_ref), "data complete, computable"


# ==========================================
# 4. Main data-processing routine
# ==========================================
def process_lcrs_evaluation(base_path, item_path, output_path):
    print("--- Starting LCRS batch screening model (transparent full version) ---")

    def read_csv_safe(path):
        for enc in ['gbk', 'utf-8-sig', 'utf-8']:
            try:
                return pd.read_csv(path, encoding=enc, low_memory=False)
            except:
                continue
        return pd.DataFrame()

    base_df = read_csv_safe(base_path)
    item_df = read_csv_safe(item_path)
    if base_df.empty: return

    # Clean base-table IDs
    base_df['app_no'] = base_df['app_no'].astype(str).str.strip()
    base_df['pat_id'] = base_df['pat_id'].astype(str).str.strip()

    # 1. Pivot to wide table
    records = []
    if not item_df.empty:
        item_df['app_no'] = item_df['app_no'].astype(str).str.strip()
        item_df['pat_id'] = item_df['pat_id'].astype(str).str.strip()

        item_df['cat'] = item_df['res_itm_name'].apply(get_category)
        filtered = item_df.dropna(subset=['cat', 'res_value']).copy()

        def get_first_valid(val_list, parser=None):
            for v in val_list:
                if pd.isna(v) or str(v).strip() == "": continue
                parsed = parser(v) if parser else clean_numeric(v)
                if pd.notna(parsed): return parsed
            return np.nan

        for (app_no, pat_id), group in filtered.groupby(['app_no', 'pat_id']):
            row = {'app_no': app_no, 'pat_id': pat_id}
            data_pool = {k: [] for k in FIELD_MAPPING.keys()}
            for _, item in group.iterrows():
                data_pool[item['cat']].append(str(item['res_value']))

            w = get_first_valid(data_pool['weight'])
            h = get_first_valid(data_pool['height'])
            bmi = get_first_valid(data_pool['BMI'])

            if pd.isna(bmi) and pd.notna(w) and pd.notna(h) and h > 0:
                bmi = round(w / ((h / 100) ** 2), 2)

            row.update({
                'item_age': get_first_valid(data_pool['age']),
                'BMI': bmi,
                'height': h,
                'weight': w,
                'smoke': get_first_valid(data_pool['smoke'], parse_binary),
                'residential_area': get_first_valid(data_pool['residential_area'], parse_binary),
                'frequent_cough': get_first_valid(data_pool['frequent_cough'], parse_binary),
                'has_chronic_bronchitis': get_first_valid(data_pool['has_chronic_bronchitis'], parse_binary),
                'previous_cancer': get_first_valid(data_pool['previous_cancer'], parse_binary),
                'family_cancer_history': get_first_valid(data_pool['family_cancer_history'], parse_binary),
                'edu': get_first_valid(data_pool['edu'], parse_edu),
                'cigarettes_per_day': get_first_valid(data_pool['cigarettes_per_day']),
                'years_smoked': get_first_valid(data_pool['years_smoked']),
                'inhale': get_first_valid(data_pool['inhale'], parse_binary),
                'years_quit': get_first_valid(data_pool['years_quit']),
                'exercise': get_first_valid(data_pool['exercise'], parse_binary)
            })
            records.append(row)

    wide_df = pd.DataFrame(records) if records else pd.DataFrame(columns=['app_no', 'pat_id'])

    # 2. Merge: left join, keep all
    final_merged = pd.merge(base_df, wide_df, on=['app_no', 'pat_id'], how='left')

    def fmt_in(val, is_bool=False, bool_type="yes_no", edu=False):
        if pd.isna(val) or str(val).strip() == "": return "missing"
        if is_bool:
            if bool_type == "city": return "urban" if val == 1 else "rural"
            return "yes/present" if val == 1 else "no/absent"
        if edu:
            if val == 2: return "junior_high_or_below"
            if val == 1: return "high school"
            if val == 0: return "college_or_above"
        return str(val)

    # 3. Batch scoring and result generation
    results = []
    for _, r in final_merged.iterrows():
        age = clean_numeric(r.get('app_pat_age'))
        if pd.isna(age): age = r.get('item_age')

        r['age'] = age

        eval_res, remark = calculate_lcrs_score(r)

        # Assemble the dict (definition order here is the exported column order)
        row_data = {
            'pat_id': r['pat_id'],
            'app_no': r.get('app_no', ''),
            'pat_name': r.get('pat_name', 'unknown'),
            'model_name': 'Lung Cancer Risk Score (LCRS)',
            'code': '-',
            'gauge_reference': '-',
            'population_type': 'unknown',
            'lcrs_total_score': '-',
            'risk_probability': '-',
            'risk_level': '-',
            'prediction_horizon': 'next 10 years',
            'assessment_note': remark,
            'sex': r.get('pat_sex', 'missing'),
            'age': int(age) if pd.notna(age) else "missing",
            'BMI': fmt_in(r.get('BMI')),
            'height (cm)': fmt_in(r.get('height')),
            'weight (kg)': fmt_in(r.get('weight')),
            'residence': fmt_in(r.get('residential_area'), is_bool=True, bool_type="city"),
            'education_level': fmt_in(r.get('edu'), edu=True),
            'smoking_status': fmt_in(r.get('smoke'), is_bool=True),
            'cigarettes_per_day': fmt_in(r.get('cigarettes_per_day')),
            'smoking_years': fmt_in(r.get('years_smoked')),
            'smoke_inhalation': fmt_in(r.get('inhale'), is_bool=True),
            'years_since_quit': fmt_in(r.get('years_quit')),
            'regular_exercise': fmt_in(r.get('exercise'), is_bool=True),
            'frequent_cough': fmt_in(r.get('frequent_cough'), is_bool=True),
            'chronic_bronchitis': fmt_in(r.get('has_chronic_bronchitis'), is_bool=True),
            'prior_cancer_history': fmt_in(r.get('previous_cancer'), is_bool=True),
            'family_cancer_history': fmt_in(r.get('family_cancer_history'), is_bool=True)
        }

        # Overwrite the placeholder with real data when computation succeeds
        if eval_res:
            score, prob, risk, code, model, details, gauge_ref = eval_res
            row_data['code'] = code
            row_data['gauge_reference'] = gauge_ref
            row_data['population_type'] = model
            row_data['lcrs_total_score'] = round(score, 1)
            row_data['risk_probability'] = f"{prob:.2f}%"
            row_data['risk_level'] = risk

        results.append(row_data)

    if results:
        res_df = pd.DataFrame(results)
        # Output directly without forced reordering
        res_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"Assessment complete: {len(res_df)} results (including missing-data records), exported to {output_path}")


if __name__ == "__main__":
    process_lcrs_evaluation("base.csv", "item_result.csv", "lcrs_risk_results_perfect.csv")