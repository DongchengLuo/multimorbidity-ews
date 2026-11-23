import pandas as pd
import numpy as np
import re
import os

# ==========================================
# 1. Field mapping (AD8 cognitive impairment screening scale)
# ==========================================
FIELD_MAPPING = {
    'q1_judgment': ['judgment_problems', 'q1', 'difficulty_deciding', 'problems with judgment'],
    'q2_interest': ['loss_of_interest', 'q2', 'reduced_interest_in_activities', 'loss of interest'],
    'q3_repeat': ['repeating_same_things', 'q3', 'repeating_questions', 'repeating the same things'],
    'q4_learning': ['difficulty_learning_tools', 'q4', 'cannot_use_appliances', 'trouble learning'],
    'q5_date': ['forgets_date', 'q5', 'temporal_disorientation', 'forgetting the correct month'],
    'q6_financial': ['difficulty_handling_finances', 'q6', 'difficulty_with_accounts', 'financial affairs'],
    'q7_appointment': ['forgets_appointments', 'q7', 'poor_memory', 'remembering appointments'],
    'q8_daily_memory': ['daily_memory_problems', 'q8', 'thinking_memory_problems', 'daily problems with thinking']
}

# ==========================================
# 2. Basic cleaning and parsing logic
# ==========================================
def normalize_str(s):
    if pd.isna(s): return ""
    return re.sub(r'[\(\)\（\）\[\]\【\】\{\}\s\_\-]', '', str(s)).lower()

def get_category(name):
    norm_name = normalize_str(name)
    for key, patterns in FIELD_MAPPING.items():
        for p in patterns:
            if normalize_str(p) in norm_name: return key
    return None

def parse_ad8_response(val):
    if pd.isna(val) or str(val).strip() == "": return np.nan
    s = str(val).lower()
    if any(x in s for x in ['yes', 'present', 'change', '1', 'positive']):
        if 'nochange' in s or 'no_change' in s: return 0
        return 1
    if any(x in s for x in ['no', 'absent', '0', 'normal', 'negative']):
        return 0
    return np.nan

# ==========================================
# 3. Core scoring logic
# ==========================================
def calculate_ad8_score(items_dict):
    keys = ['q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7', 'q8']
    if any(pd.isna(items_dict.get(k)) for k in keys):
        return np.nan, "indicator_missing"
    
    score = sum(items_dict[k] for k in keys)
    return score, "computed"

# ==========================================
# 4. Main data-processing routine
# ==========================================
def process_data(base_path, item_path, output_path):
    print("--- Starting AD8 cognitive screening model (full base iteration) ---")

    GAUGE_REF_STR = "0-1|2|3-8"

    def read_csv_safe(path):
        for enc in ['gbk', 'utf-8-sig', 'utf-8']:
            try: return pd.read_csv(path, encoding=enc, low_memory=False)
            except: continue
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

        # Map the scores of the 8 questions
        row_map = {}
        row_map['q1'] = parse_ad8_response(" ".join(data_pool['q1_judgment'])) if data_pool['q1_judgment'] else np.nan
        row_map['q2'] = parse_ad8_response(" ".join(data_pool['q2_interest'])) if data_pool['q2_interest'] else np.nan
        row_map['q3'] = parse_ad8_response(" ".join(data_pool['q3_repeat'])) if data_pool['q3_repeat'] else np.nan
        row_map['q4'] = parse_ad8_response(" ".join(data_pool['q4_learning'])) if data_pool['q4_learning'] else np.nan
        row_map['q5'] = parse_ad8_response(" ".join(data_pool['q5_date'])) if data_pool['q5_date'] else np.nan
        row_map['q6'] = parse_ad8_response(" ".join(data_pool['q6_financial'])) if data_pool['q6_financial'] else np.nan
        row_map['q7'] = parse_ad8_response(" ".join(data_pool['q7_appointment'])) if data_pool['q7_appointment'] else np.nan
        row_map['q8'] = parse_ad8_response(" ".join(data_pool['q8_daily_memory'])) if data_pool['q8_daily_memory'] else np.nan

        score, status = calculate_ad8_score(row_map)

        if pd.isna(score):
            results.append({
                'model_name': 'ad8_score',
                'pat_id': pat_id,
                'app_no': app_no,
                'pat_name': base_row.get('pat_name', 'unknown'),
                'predicted_score': '-',
                'risk_level': '-',
                'code': '-',
                'prediction_horizon': 'current',
                'gauge_reference': GAUGE_REF_STR,
                'clinical_recommendation': '-',
                'sex': base_row.get('pat_sex', '-'),
                'age': base_row.get('app_pat_age', '-')
            })
        else:
            display_score = int(score)
            if score >= 3:
                risk_level, risk_code = "high risk", "F03.1.3"
                suggestion = "suspected dementia; neurology evaluation advised"
            elif score == 2:
                risk_level, risk_code = "medium risk", "F03.1.2"
                suggestion = "suspected mild cognitive impairment (MCI); regular follow-up advised"
            else:
                risk_level, risk_code = "low risk", "F03.1.1"
                suggestion = "current cognitive function is good"

            results.append({
                'model_name': 'ad8_score',
                'pat_id': pat_id,
                'app_no': app_no,
                'pat_name': base_row.get('pat_name', 'unknown'),
                'predicted_score': display_score,
                'risk_level': risk_level,
                'code': risk_code,
                'prediction_horizon': 'current',
                'gauge_reference': GAUGE_REF_STR,
                'clinical_recommendation': suggestion,
                'sex': base_row.get('pat_sex', '-'),
                'age': base_row.get('app_pat_age', '-')
            })

    # Export
    if results:
        res_df = pd.DataFrame(results)
        res_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        valid_count = len(res_df[res_df['code'] != '-'])
        print(f"--- Assessment complete: {len(res_df)} records processed ---")
        print(f"--- Valid computed records: {valid_count} ---")
        print(f"--- Records not computable due to missing values: {len(res_df) - valid_count} ---")
        print(f"--- Results saved to: {os.path.abspath(output_path)} ---")

if __name__ == "__main__":
    process_data(
        base_path="base.csv", 
        item_path="item_result.csv", 
        output_path="ad8_screening_result.csv"
    )
