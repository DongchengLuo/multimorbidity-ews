import pandas as pd
import numpy as np
import re

# ==========================================
# 1. Field mapping
# ==========================================
FIELD_MAPPING = {
    'age': ['app_pat_age', 'age'],
    'gender': ['pat_sex', 'sex'],
    'ethnicity': ['patient nation', 'ethnicity', 'race'],
    'parental_asthma': ['parental_asthma', 'family_asthma'],
    'eczema': ['eczema', 'eczema_before_3y', 'eczema_before_3'],
    'wheeze_no_cold': ['non_cold_wheeze', 'wheeze_outside_cold', 'wheezing_apart_from_colds'],
    'early_wheeze': ['early_wheeze', 'wheeze_before_3y', 'early_wheezing'],
    'spt_polysensitization': ['skin_prick_test', 'SPT_positive', 'polysensitization', 'sensitization_ge_2']
}


# ==========================================
# 2. Basic utility functions
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


def is_positive(val):
    if pd.isna(val): return False
    s = str(val).lower()
    pos_keywords = ['yes', 'present', 'positive', '1', 'high', 'true']
    neg_keywords = ['no', 'absent', 'negative', '0', 'not', 'false']
    if any(kw in s for kw in pos_keywords):
        if any(kw in s for kw in neg_keywords): return False
        return True
    return False


# ==========================================
# 3. PARS core scoring logic
# ==========================================
def calculate_pars_score(parental_asthma, eczema, wheeze_no_cold, early_wheeze, is_african, spt_poly):
    score = 0
    details = {}
    factors = [
        ('parental_asthma', parental_asthma, 2),
        ('eczema_before_3y', eczema, 2),
        ('non_cold_wheeze', wheeze_no_cold, 3),
        ('early_wheeze', early_wheeze, 3),
        ('african_american', is_african, 2),
        ('polysensitization', spt_poly, 2)
    ]
    for name, condition, weight in factors:
        points = weight if condition else 0
        score += points
        details[name] = points
    return score, details


def get_risk_probability(score):
    prob_map = {
        0: "3%", 1: "3%",
        2: "6%", 3: "8%", 4: "11%",
        5: "15%", 6: "19%", 7: "25%", 8: "32%",
        9: "40%", 10: "49%", 11: "58%", 12: "66%", 13: "70%", 14: "79%"
    }
    if score <= 4:
        level = "low risk"
    elif score <= 8:
        level = "medium risk"
    else:
        level = "high risk"
    return prob_map.get(score, "unknown"), level


def get_risk_code(level):
    if level == "low risk":
        return "J45.4.1"
    elif level == "medium risk":
        return "J45.4.2"
    elif level == "high risk":
        return "J45.4.3"
    return ""


# ==========================================
# 4. Age-dependent interpretation logic
# ==========================================
def get_age_specific_interpretation(current_age, risk_prob, risk_level):
    if pd.isna(current_age):
        return "school_age (7-10 years)", f"Based on the PARS model, the child's asthma risk at school age is {risk_prob} ({risk_level})."

    age = int(current_age)

    if age < 3:
        target = "at_age_7"
        desc = f"The child is still in early childhood. If current risk factors persist, the predicted probability of developing asthma by age 7 is {risk_prob}."
    elif 3 <= age < 7:
        target = "at_age_7"
        desc = f"The child is currently {age} years old. Based on early indicators, the predicted asthma risk at age 7 is {risk_prob}, in the {risk_level} group."
    elif 7 <= age <= 10:
        target = "at_age_10"
        desc = f"The child is currently {age} years old. Based on PARS validation in older children, the predicted risk of persistent or incident asthma at age 10 is {risk_prob}."
    else:
        target = "retrospective_assessment (7-10 years)"
        desc = f"The child is over 10 years old. This score reflects the asthma risk the child would have had at school age ({target}) ({risk_prob}); it can help trace etiology or assess atopic predisposition."

    return target, desc


# ==========================================
# 5. Main batch-processing routine
# ==========================================
def process_pars_targeted(base_path, item_path, output_path):
    print("--- Starting PARS age-targeted prediction (full base iteration) ---")

    GAUGE_REF_STR = "0-4|5-8|9-14"
    MODEL_NAME = 'Childhood Asthma Risk Score (PARS)'

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
        print("Error: base data file is empty or unreadable.")
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
        data_pool = {k: [] for k in FIELD_MAPPING.keys()}
        if not filtered.empty:
            group = filtered[(filtered['app_no'] == app_no) & (filtered['pat_id'] == pat_id)]
            for _, item in group.iterrows():
                data_pool[item['cat']].append(str(item['res_value']))

        parental_asthma = any(is_positive(v) for v in data_pool.get('parental_asthma', []))
        eczema = any(is_positive(v) for v in data_pool.get('eczema', []))
        wheeze_no_cold = any(is_positive(v) for v in data_pool.get('wheeze_no_cold', []))
        early_wheeze = any(is_positive(v) for v in data_pool.get('early_wheeze', []))
        spt_poly = any(is_positive(v) for v in data_pool.get('spt_polysensitization', []))

        eth_str = " ".join(data_pool.get('ethnicity', [])).lower()
        is_african = any(kw in eth_str for kw in ['african', 'black'])

        age = clean_numeric(base_row.get('app_pat_age'))

        # The PARS model needs no strict numeric indicators; defaulting all booleans to False allows computation
        score, details = calculate_pars_score(
            parental_asthma, eczema, wheeze_no_cold,
            early_wheeze, is_african, spt_poly
        )
        prob, level = get_risk_probability(score)
        code = get_risk_code(level)
        target_label, interpretation = get_age_specific_interpretation(age, prob, level)

        results.append({
            'pat_id': pat_id,
            'app_no': app_no,
            'pat_name': base_row.get('pat_name', 'unknown'),
            'assessment_model_name': MODEL_NAME,
            'gender': base_row.get('pat_sex', ''),
            'current_age': age if not pd.isna(age) else "-",
            'pars_total_score': score,
            'risk_level': level,
            'code': code,
            'risk_probability': prob,
            'gauge_reference': GAUGE_REF_STR,
            'predicted_target_node': target_label,
            'targeted_interpretation': interpretation,
            'score_details': str(details)
        })

    if results:
        cols = ['pat_id', 'app_no', 'pat_name', 'code', 'gauge_reference', 'risk_level', 'pars_total_score', 'risk_probability',
                'targeted_interpretation', 'predicted_target_node', 'assessment_model_name', 'current_age', 'gender', 'score_details']
        df_out = pd.DataFrame(results)
        final_cols = [c for c in cols if c in df_out.columns]
        df_out[final_cols].to_csv(output_path, index=False, encoding='utf-8-sig')
        valid_count = len(df_out[df_out['code'] != '-'])
        print(f"--- Assessment complete: {len(df_out)} records ---")
        print(f"--- Valid computed records: {valid_count} ---")
        print(f"--- not computable due to missing values: {len(df_out) - valid_count} ---")
        print(f"--- Results saved to: {output_path} ---")


if __name__ == "__main__":
    process_pars_targeted(
        "base.csv",  
        "item_result.csv", 
        "pars_targeted_output.csv"
    )
