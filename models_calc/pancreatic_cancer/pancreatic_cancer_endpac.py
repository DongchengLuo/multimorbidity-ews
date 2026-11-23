import pandas as pd
import numpy as np
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model_unit_utils import convert_measurement

# ==========================================
# 1. Field mapping
# ==========================================
FIELD_MAPPING = {
    'age': ['app_pat_age', 'age'],  
    'gender': ['pat_sex', 'sex'],  
    'weight': ['weight'],  
    'bg_current': ['current_glucose', 'glucose_at_nodm', 'glucose_at_diabetes_diagnosis', 'glucose_at_nod', 'fpg_current', 'bg_current', 'fasting_glucose', 'GLU'],
    'bg_1y': ['glucose_one_year_ago', 'prior_glucose', 'fpg_1y_ago', 'glucose_pre_1y', 'bg_1y', 'FBG_one_year_ago'],
    'weight_1y': ['weight_one_year_ago', 'prior_weight', 'weight_1y_ago', 'weight_pre_1y'],
    'weight_change': ['weight_change', 'weight_diff']
}

# ==========================================
# 2. Basic cleaning and matching utility functions
# ==========================================
def normalize_str(s):
    if pd.isna(s): return ""
    return re.sub(r'[\(\)\（\）\[\]\【\】\{\}\s]', '', str(s)).lower()

def get_category(name):
    norm_name = normalize_str(name)
    for key, patterns in FIELD_MAPPING.items():
        for p in patterns:
            # Exact match
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
# 3. END-PAC core scoring logic
# ==========================================
def glucose_category(bg_mg_dl):
    if bg_mg_dl < 100: return 1
    if bg_mg_dl <= 109: return 2
    if bg_mg_dl <= 125: return 3
    if bg_mg_dl <= 160: return 4
    return 5


def calculate_end_pac_score(bg_current, bg_1y, weight_change, age_at_nod):
    # Missing-value check: return the "-" placeholder when any key indicator is missing
    if any(pd.isna(x) for x in [bg_current, bg_1y, weight_change, age_at_nod]):
        return "-", "-"

    score_total = 0
    debug_info = {}

    # --- A. Glucose-change score: glucose category at NOD minus category ~1 year earlier ---
    current_category = glucose_category(bg_current)
    previous_category = glucose_category(bg_1y)
    s_a = current_category - previous_category
    score_total += s_a
    debug_info['A_bg_change_score'] = s_a
    debug_info['A_bg_categories'] = f"current {current_category} - one year ago {previous_category}"

    # --- B. Weight-change score ---
    wc = weight_change
    if wc <= -6.0: s_b = 6
    elif -5.9 <= wc <= -4.0: s_b = 4
    elif -3.9 <= wc <= -2.0: s_b = 2
    elif -1.9 <= wc <= 1.9: s_b = 0
    elif 2.0 <= wc <= 3.9: s_b = -2
    elif 4.0 <= wc <= 5.9: s_b = -4
    else: s_b = -6
    score_total += s_b
    debug_info['B_weight_score'] = s_b

    # --- C. Age score ---
    if age_at_nod < 60: s_c = -1
    elif 60 <= age_at_nod <= 69: s_c = 0
    else: s_c = 1
    score_total += s_c
    debug_info['C_age_score'] = s_c

    return score_total, debug_info

# ==========================================
# 4. Main data-processing routine
# ==========================================
def process_data(base_path, item_path, output_path):
    print("--- Starting END-PAC pancreatic cancer risk model ---")

    def read_csv_safe(path):
        for enc in ['utf-8-sig', 'gb18030', 'gbk', 'utf-8']:
            try: return pd.read_csv(path, encoding=enc, low_memory=False)
            except: continue
        return pd.DataFrame()

    base_df = read_csv_safe(base_path)
    item_df = read_csv_safe(item_path)

    if base_df.empty:
        print("Error: base data file could not be read or is empty")
        return

    # [Important fix]: strip redundant leading zeros in the base table that break matching
    base_df['app_no'] = base_df['app_no'].astype(str).str.strip().str.lstrip('0')
    base_df['pat_id'] = base_df['pat_id'].astype(str).str.strip().str.lstrip('0')

    # 1. Pivot to wide table
    records = []
    if not item_df.empty:
        item_df['app_no'] = item_df['app_no'].astype(str).str.strip().str.lstrip('0')
        item_df['pat_id'] = item_df['pat_id'].astype(str).str.strip().str.lstrip('0')
        
        item_df['cat'] = item_df['res_itm_name'].apply(get_category)
        filtered = item_df.dropna(subset=['cat', 'res_value']).copy()

        for (app_no, pat_id), group in filtered.groupby(['app_no', 'pat_id']):
            row = {'app_no': app_no, 'pat_id': pat_id}
            data_pool = {k: [] for k in FIELD_MAPPING.keys()}
            
            for _, item in group.iterrows():
                data_pool[item['cat']].append(str(item['res_value']))

            # Take the first valid number, not the mean
            def get_first_num(lst):
                nums = [clean_numeric(v) for v in lst if pd.notna(clean_numeric(v))]
                return nums[0] if nums else np.nan

            unit_notes, unit_errors = [], []
            for bg_field in ['bg_current', 'bg_1y']:
                cat_rows = group[group['cat'] == bg_field]
                converted = np.nan
                errors = []
                for _, measurement in cat_rows.iterrows():
                    result = convert_measurement(
                        measurement.get('res_value'),
                        measurement.get('res_unit', ''),
                        indicator='glucose',
                        target_unit='mg/dL',
                    )
                    if result.ok:
                        converted = result.value
                        unit_notes.append(f"{bg_field}: {result.rule}")
                        break
                    errors.append(f"{bg_field}: {result.error}")
                if pd.isna(converted):
                    unit_errors.extend(errors)
                row[bg_field] = converted
            row['unit_conversion'] = '；'.join(unit_notes)
            row['unit_error'] = '；'.join(unit_errors)
            
            wc_direct = get_first_num(data_pool.get('weight_change', []))
            if pd.notna(wc_direct):
                row['weight_change'] = wc_direct
            else:
                w_curr = get_first_num(data_pool.get('weight', []))
                w_prev = get_first_num(data_pool.get('weight_1y', []))
                if pd.notna(w_curr) and pd.notna(w_prev):
                    row['weight_change'] = w_curr - w_prev
                else:
                    row['weight_change'] = np.nan

            row['age_from_item'] = get_first_num(data_pool.get('age', []))
            records.append(row)

    wide_items = pd.DataFrame(records) if records else pd.DataFrame(columns=['app_no', 'pat_id'])

    # 2. Merge basic information (left join, keep everyone)
    final = pd.merge(base_df, wide_items, on=['app_no', 'pat_id'], how='left')

    # 3. Scoring and result generation
    results = []
    for _, r in final.iterrows():
        # Fetch the variables used for computation
        age = clean_numeric(r.get('app_pat_age'))
        if pd.isna(age):
            age = r.get('age_from_item', np.nan)

        bg_current = r.get('bg_current', np.nan)
        bg_1y = r.get('bg_1y', np.nan)
        weight_change = r.get('weight_change', np.nan)
        app_no = r.get('app_no', '')

        # Call the core computation
        score, details = calculate_end_pac_score(bg_current, bg_1y, weight_change, age)

        # Result analysis and risk coding (use "-" when not computable)
        if score == "-":
            risk_level = "-"
            risk_code = "-"
            risk_criteria = "-"
            calc_status = "input_incomplete"
            no_calc_reason = "missing current glucose, glucose one year ago, weight change or age"
        else:
            calc_status = "computed"
            no_calc_reason = ""
            if score >= 3:
                risk_level = "high risk"
                risk_code = "C25.1.3"
                risk_criteria = "total score >= 3"
            elif 1 <= score <= 2:
                risk_level = "medium risk"
                risk_code = "C25.1.2"
                risk_criteria = "1 <= total score <= 2"
            else:  
                risk_level = "low risk"
                risk_code = "C25.1.1"
                risk_criteria = "total score <= 0"

        def fmt_in(v, is_num=False):
            if pd.isna(v) or v is None or v == "": return "missing"
            return f"{v:.1f}" if is_num else str(v)

        results.append({
            'model_name': 'Pancreatic Cancer END-PAC',
            'app_no': app_no, 
            'pat_id': r.get('pat_id', ''),
            'pat_name': r.get('pat_name', 'unknown'),
            'sex': r.get('pat_sex', 'missing'),
            'age': int(age) if pd.notna(age) else "missing",
            # --- Raw indicators (missing shown as "missing") ---
            'current_glucose (mg/dL)': fmt_in(bg_current, True),
            'glucose_one_year_ago (mg/dL)': fmt_in(bg_1y, True),
            'glucose_category_change_score': (
                glucose_category(bg_current) - glucose_category(bg_1y)
                if pd.notna(bg_current) and pd.notna(bg_1y) else 'missing'
            ),
            'weight_change (kg)': fmt_in(weight_change, True),
            'unit_conversion': r.get('unit_conversion', ''),
            'unit_error': r.get('unit_error', ''),
            # --- Model results (missing shown as "-") ---
            'predicted_score': score,
            'risk_level': risk_level,
            'code': risk_code,
            'prediction_horizon':'next 3 years',
            'risk_criteria': risk_criteria,
            'calculation_status': calc_status,
            'not_computed_reason': no_calc_reason,
            'score_details': str(details) if details != "-" else "-"
        })

    # 4. Export results
    if results:
        res_df = pd.DataFrame(results)
        res_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"--- Assessment complete: {len(res_df)} complete results, exported to {output_path} ---")
    else:
        print("--- Assessment complete, no valid results ---")

if __name__ == "__main__":
    process_data("base.csv", "item_result.csv", "end_pac_screening_result.csv")
