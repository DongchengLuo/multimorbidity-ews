import pandas as pd
import numpy as np
import re
import os

# ==========================================
# 1. Field-mapping configuration (for the long-table Item structure)
# ==========================================
FIELD_MAPPING = {
    # --- GA2LEN core scoring fields (Q1) ---
    'ever_asthma': ['ever_asthma', 'q1', 'asthma_history', 'prior_asthma_history'],

    # --- A2 core scoring fields (Q2-Q3, Q11) ---
    'doc_confirm': ['doc_confirm', 'physician_diagnosed', 'q2', 'physician_confirmed_asthma'],
    'still_asthma': ['still_asthma', 'current_asthma', 'q3'],
    'sob_rest': ['sob_rest', 'dyspnea_at_rest', 'q11', 'shortness_of_breath_rest'],

    # --- Shared fields (Q5-Q9) ---
    'attack_12m': ['attack_12m', 'episodes_past_12m', 'q5', 'asthma_attack', 'episodes_within_past_year'],
    'meds_current': ['meds_current', 'current_medication', 'q6', 'asthma_meds', 'currently_on_medication'],
    'wheeze_12m': ['wheeze_12m', 'wheeze_past_12m', 'q7', 'wheezing', 'wheeze_within_past_year'],
    'wheeze_no_cold': ['wheeze_no_cold', 'no_cold_wheeze', 'q8'],
    'breathless_wheeze': ['breathless_wheeze', 'wheeze_with_dyspnea', 'q9', 'breathless_with_wheeze']
}


# ==========================================
# 2. Basic utility functions
# ==========================================
def normalize_str(s):
    """Normalize the string: strip symbols, lowercase"""
    if pd.isna(s): return ""
    return re.sub(r'[\(\)\（\）\[\]\【\】\{\}\s\_\-]', '', str(s)).lower()


def get_category(name):
    """Match the standard label from the field name"""
    norm_name = normalize_str(name)
    for key, patterns in FIELD_MAPPING.items():
        for p in patterns:
            if normalize_str(p) in norm_name: return key
    return None


def parse_yes_no(val_list):
    """Parse a boolean from the extracted value list: 1 (yes), 0 (no/missing)"""
    if not val_list: return 0
    combined_text = " ".join([str(v) for v in val_list]).lower()
    if any(kw in combined_text for kw in ['no', 'absent', 'none', 'not', '0', 'false', 'negative']):
        return 0
    true_set = ['1', '1.0', 'yes', 'y', 'present', 'true', 't', 'positive']
    for kw in true_set:
        if kw in combined_text:
            return 1
    return 0


# ==========================================
# 3. Asthma score computation and logic
# ==========================================
VARS_A2 = [
        'doc_confirm', 'still_asthma', 'attack_12m', 'meds_current',
        'wheeze_12m', 'wheeze_no_cold', 'breathless_wheeze', 'sob_rest'
]
VARS_GA2LEN = [
        'ever_asthma', 'attack_12m', 'meds_current',
        'wheeze_12m', 'wheeze_no_cold', 'breathless_wheeze'
]


def calculate_asthma_score(record):
    """The source has two independent questionnaires: prefer A2 when complete, fall back to GA2LEN only when A2 is incomplete."""

    def get_val(v):
        val = record.get(v, 0)
        return 0 if pd.isna(val) else int(val)

    if all(bool(record.get(f'{v}_present', False)) for v in VARS_A2):
        return 'A2', sum(get_val(v) for v in VARS_A2)
    if all(bool(record.get(f'{v}_present', False)) for v in VARS_GA2LEN):
        return 'GA2LEN', sum(get_val(v) for v in VARS_GA2LEN)
    return None, None


def get_risk_assessment(model, score):
    """Apply the A2 and GA2LEN cut-offs from the source separately; do not merge the two models' results."""
    probable = score >= 4
    possible = score >= (2 if model == 'A2' else 1)
    if probable:
        return 'highly_likely_asthma', 'J45.1.3'
    if possible:
        return 'possible_asthma', 'J45.1.2'
    return 'asthma_excluded', 'J45.1.1'


# ==========================================
# 4. Main data-processing routine
# ==========================================
def process_data(base_path, item_path, output_path):
    print("--- Starting adult asthma screening (A2/GA2LEN) model ---")

    def read_csv_safe(path):
        if not os.path.exists(path):
            return pd.DataFrame()
        for enc in ['utf-8-sig', 'gbk', 'utf-8']:
            try:
                return pd.read_csv(path, encoding=enc, low_memory=False)
            except:
                continue
        return pd.DataFrame()

    base_df = read_csv_safe(base_path)
    item_df = read_csv_safe(item_path)

    if base_df.empty:
        print(f"base data unreadable or empty (Base table must contain patients): {base_path}")
        return

    # 1. Pivot indicators to a wide table
    records = []
    if not item_df.empty and 'res_itm_name' in item_df.columns:
        item_df['cat'] = item_df['res_itm_name'].apply(get_category)
        filtered = item_df.dropna(subset=['cat', 'res_value']).copy()

        # Aggregate data per person
        for (app_no, pat_id), group in filtered.groupby(['app_no', 'pat_id']):
            row = {'app_no': str(app_no).strip(), 'pat_id': str(pat_id).strip()}
            data_pool = {k: [] for k in FIELD_MAPPING.keys()}

            for _, item in group.iterrows():
                data_pool[item['cat']].append(str(item['res_value']))

            # Extract each asthma-related indicator
            for key in FIELD_MAPPING.keys():
                raw_vals = [str(v) for v in data_pool.get(key, []) if str(v).strip() and not pd.isna(v)]
                row[f'{key}_raw'] = "、".join(raw_vals) if raw_vals else 'missing'
                row[key] = parse_yes_no(data_pool.get(key, []))
                row[f'{key}_present'] = bool(raw_vals)

            records.append(row)
            
    # Initialize an empty DF with the base join keys when no indicators are extracted
    if records:
        wide_items = pd.DataFrame(records)
        wide_items['has_data'] = True  # <--- flag: this patient has indicator data
    else:
        wide_items = pd.DataFrame(columns=['app_no', 'pat_id', 'has_data'])

    # 2. Merge basic information (Base table)
    base_df['app_no'] = base_df['app_no'].astype(str).str.strip()
    base_df['pat_id'] = base_df['pat_id'].astype(str).str.strip()

    # Use a left join to keep all patients in the Base table
    final = pd.merge(base_df, wide_items, on=['app_no', 'pat_id'], how='left')
    
    # Fill has_data; patients absent from wide_items are False (no data)
    final['has_data'] = final['has_data'].fillna(False)

    # 3. Score and generate the final output
    processed_records = []
    # Helper: handle NaN from missing indicator data after a left join
    def get_raw_text(r, col_name):
        val = r.get(col_name)
        if pd.isna(val) or str(val).strip() == '':
            return 'missing'
        return str(val)

    for _, r in final.iterrows():
        # === Core change logic ===
        # Decide whether to compute a score or output "-" based on whether exam indicators exist
        if r['has_data']:
            selected_model, selected_score = calculate_asthma_score(r)
            if selected_model is not None:
                risk_level, code = get_risk_assessment(selected_model, selected_score)
                current_gauge_ref = '0-1|2-3|4-8' if selected_model == 'A2' else '0|1-3|4-6'
            else:
                selected_model, selected_score, code, risk_level, current_gauge_ref = '-', '-', '-', '-', '-'
        else:
            # For patients with no data at all, use "-" as the core result placeholder
            selected_model = '-'
            selected_score = '-'
            code = '-'
            risk_level = '-'
            current_gauge_ref = '-'

        out_row = {
            # --- Identity and system-required fields ---
            'pat_id': r.get('pat_id', ''),
            'app_no': r.get('app_no', ''),
            'pat_name': r.get('pat_name', 'unknown'),
            'model_name': 'Adult Asthma Screening (A2 first)',
            
            # --- Core output indicators ---
            'code': code,
            'gauge_reference': current_gauge_ref,
            'model_used': selected_model,
            'predicted_score': selected_score,
            'risk_level': risk_level,

            # --- Raw questionnaire data per item (still outputs 'missing') ---
            'ever_asthma': get_raw_text(r, 'ever_asthma_raw'),
            'physician_ever_diagnosed': get_raw_text(r, 'doc_confirm_raw'),
            'current_asthma': get_raw_text(r, 'still_asthma_raw'),
            'episodes_past_12m': get_raw_text(r, 'attack_12m_raw'),
            'current_medication': get_raw_text(r, 'meds_current_raw'),
            'wheeze_past_12m': get_raw_text(r, 'wheeze_12m_raw'),
            'no_cold_wheeze': get_raw_text(r, 'wheeze_no_cold_raw'),
            'wheeze_with_dyspnea': get_raw_text(r, 'breathless_wheeze_raw'),
            'dyspnea_at_rest': get_raw_text(r, 'sob_rest_raw')
        }
        processed_records.append(out_row)

    # 4. Export CSV
    if processed_records:
        res_df = pd.DataFrame(processed_records)
        res_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"--- Processing complete: {len(res_df)} records written to {output_path} ---")
        print(f"sample data:\n{res_df[['pat_name', 'model_used', 'predicted_score', 'risk_level']].head(5)}")
    else:
        print("no valid data generated.")


if __name__ == "__main__":
    # Configure paths
    base_file = 'base.csv'
    item_file = 'item_result.csv'
    output_file = 'asthma_screening_result.csv'

    process_data(base_file, item_file, output_file)
