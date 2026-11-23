import pandas as pd
import numpy as np
import re
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model_unit_utils import convert_measurement

# ==========================================
# 1. Field-mapping configuration (unchanged)
# ==========================================
FIELD_MAPPING = {
    # --- Basic demographic indicators ---
    'age': ['app_pat_age', 'age'],
    'gender': ['pat_sex', 'sex'],

    # --- ROMA core computation indicators ---
    'CA125': ['CA-125'],
    'FSH': ['FSH'],
    'HE4': ['HE4'],

    # --- Menopause determination indicators ---
    'menopause_status': ['menopausal', 'menopause', 'menopause_status']
}


# ==========================================
# 2. Basic cleaning utilities
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


# ==========================================
# 3. ROMA core computation logic
# ==========================================
def calculate_roma_score(age, he4, ca125, fsh_val, status_text):
    """
    Compute the ROMA value and return the risk tier
    Return None, "indicator missing" when absent
    """
    debug_info = {"model_name": "ROMA"}

    # --- A. Determine menopausal status ---
    status = str(status_text or '').strip().lower()
    if status in {'no', '0'} or any(kw in status for kw in ['premenopausal', 'still_menstruating', 'pre-menopausal']):
        is_post = False
        status_rule = 'definite_premenopausal'
    elif status in {'yes', '1', 'menopause'} or any(kw in status for kw in ['postmenopausal', 'amenorrhea_gt_1y', 'post-menopausal']):
        is_post = True
        status_rule = 'definite_postmenopausal'
    elif age > 55:
        is_post = True
        status_rule = 'menopausal status unknown and age >55; classified postmenopausal per source'
    elif age < 48:
        is_post = False
        status_rule = 'menopausal status unknown and age <48; classified premenopausal per source'
    elif not pd.isna(fsh_val):
        is_post = fsh_val >= 22
        status_rule = 'age 48-55 with unknown status; classified by FSH 22 IU/L cut-off'
    else:
        return None, 'menopausal status undetermined', debug_info

    debug_info['menopause_group'] = 'postmenopausal' if is_post else 'premenopausal'
    debug_info['menopause_rule'] = status_rule

    # --- B. Compute risk score ---
    if pd.isna(he4) or pd.isna(ca125) or he4 <= 0 or ca125 <= 0:
        return None, "indicator_missing", debug_info

    ln_he4 = math.log(he4)
    ln_ca125 = math.log(ca125)

    if is_post:
        # Postmenopausal formula (threshold 27.7%)
        pi = -8.09 + (1.04 * ln_he4) + (0.732 * ln_ca125)
        threshold = 27.7
    else:
        # Premenopausal formula (threshold 13.1%)
        pi = -12.0 + (2.38 * ln_he4) + (0.0626 * ln_ca125)
        threshold = 13.1

    # Compute predicted probability PP (0-100)
    pp = (1 / (1 + math.exp(-pi))) * 100 if pi >= 0 else (math.exp(pi) / (1 + math.exp(pi))) * 100

    # Determine risk (binary classification)
    risk_level = "high risk" if pp > threshold else "low risk"

    return round(pp, 2), risk_level, debug_info


# ==========================================
# 4. Main data-processing routine
# ==========================================
def process_data(base_path, item_path, output_path):
    print("--- Starting ROMA risk screening engine (full output) ---")

    def read_csv_safe(path):
        for enc in ['gbk', 'utf-8-sig', 'utf-8']:
            try:
                return pd.read_csv(path, encoding=enc, low_memory=False)
            except:
                continue
        return pd.DataFrame()

    # Helper: safely extract a valid number
    def get_first_valid(val_list, default_val=np.nan):
        for v in val_list:
            num = clean_numeric(v)
            if not pd.isna(num):
                return num
        return default_val

    base_df = read_csv_safe(base_path)
    item_df = read_csv_safe(item_path)

    if base_df.empty:
        print("error: base.csv unreadable or empty")
        return

    # 1. Pivot to wide table
    records = []
    if not item_df.empty:
        item_df['cat'] = item_df['res_itm_name'].apply(get_category)
        filtered = item_df.dropna(subset=['cat', 'res_value']).copy()

        for (app_no, pat_id), group in filtered.groupby(['app_no', 'pat_id']):
            row = {'app_no': str(app_no).strip(), 'pat_id': str(pat_id).strip()}

            data_pool = {k: [] for k in FIELD_MAPPING.keys()}
            unit_pool = {k: [] for k in FIELD_MAPPING.keys()}
            for _, item in group.iterrows():
                data_pool[item['cat']].append(str(item['res_value']))
                unit_pool[item['cat']].append(item.get('res_unit', ''))

            def converted_first(cat, indicator, target_unit):
                values = data_pool.get(cat, [])
                units = unit_pool.get(cat, [])
                result = convert_measurement(
                    values[0] if values else None,
                    units[0] if units else '',
                    indicator=indicator,
                    target_unit=target_unit
                )
                return (result.value if result.ok else np.nan,
                        result.rule if result.ok else result.error)

            row['he4'], row['he4_unit_note'] = converted_first('HE4', 'he4', 'pmol/L')
            row['ca125'], row['ca125_unit_note'] = converted_first('CA125', 'ca125', 'U/mL')
            row['fsh'], row['fsh_unit_note'] = converted_first('FSH', 'fsh', 'IU/L')

            status_list = data_pool.get('menopause_status', [])
            row['status_text'] = status_list[0] if status_list else ""

            records.append(row)

    wide_items = pd.DataFrame(records)

    # Fallback: initialize an empty table with standard headers when item extraction is empty
    if wide_items.empty:
        wide_items = pd.DataFrame(columns=['app_no', 'pat_id', 'he4', 'ca125', 'fsh', 'status_text'])

    # 2. Merge the Base table (changed to how='left')
    base_df['app_no'] = base_df['app_no'].astype(str).str.strip()
    base_df['pat_id'] = base_df['pat_id'].astype(str).str.strip()

    final = pd.merge(base_df, wide_items, on=['app_no', 'pat_id'], how='left')

    # 3. Result generation
    results = []
    STATIC_GAUGE_REF = "0-13.1|13.1-27.7|27.7-100"

    for _, r in final.iterrows():
        age = clean_numeric(r.get('app_pat_age'))
        he4 = r.get('he4')
        ca125 = r.get('ca125')
        fsh = r.get('fsh')
        status_text = r.get('status_text', "")

        # Check whether key indicators are missing
        if pd.isna(age) or pd.isna(he4) or pd.isna(ca125):
            risk_code = "-"
            pp_display = "-"
            risk = "-"
            calc_info = {}
        else:
            pp, risk_calc, calc_info = calculate_roma_score(age, he4, ca125, fsh, status_text)

            # calculate_roma_score returns None when a value <= 0 makes the log undefined
            if pp is None:
                risk_code = "-"
                pp_display = "-"
                risk = "-"
            else:
                risk = risk_calc
                if risk == "low risk":
                    risk_code = "C56.2.1"
                else:
                    risk_code = "C56.2.3"
                pp_display = f"{pp:.2f}"

        # Format raw indicators separately: force "missing" for empty values
        age_out = int(age) if pd.notna(age) else "missing"
        he4_out = f"{he4:.2f}" if pd.notna(he4) else "missing"
        ca125_out = f"{ca125:.2f}" if pd.notna(ca125) else "missing"
        fsh_out = f"{fsh:.2f}" if pd.notna(fsh) else "missing"
        status_out = status_text if status_text != "" and pd.notna(status_text) else "missing"

        results.append({
            'pat_id': r.get('pat_id'),
            'app_no': r.get('app_no'),
            'name': r.get('pat_name', '-'),

            # --- All raw indicators used in the computation, output as separate columns ---
            'age': age_out,
            'HE4(pmol/L)': he4_out,
            'CA125(U/mL)': ca125_out,
            'FSH(IU/L)': fsh_out,
            'HE4_unit_handling': r.get('he4_unit_note', 'missing res_unit'),
            'CA125_unit_handling': r.get('ca125_unit_note', 'missing res_unit'),
            'FSH_unit_handling': r.get('fsh_unit_note', 'missing res_unit'),
            'menopausal_status_text': status_out,
            'menopausal_status_for_model': calc_info.get('menopause_group', '-'),
            'menopause_determination_basis': calc_info.get('menopause_rule', '-'),
            # ----------------------------------------------

            'code': risk_code,
            'gauge_reference': STATIC_GAUGE_REF,
            'model_name': 'Ovarian Cancer Mass Screening - ROMA',
            'risk_probability': pp_display,
            'risk_level': risk
        })

    if results:
        res_df = pd.DataFrame(results)

        # Fixed a small typo in sorting: use the 'app_no' generated in the dict
        target_cols = ['pat_id', 'app_no', 'name', 'code', 'gauge_reference', 'risk_probability', 'risk_level']

        target_cols = [c for c in target_cols if c in res_df.columns]
        other_cols = [c for c in res_df.columns if c not in target_cols]

        res_df = res_df[target_cols + other_cols]

        res_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"--- Processing complete: {len(res_df)} records (including missing indicators) -> {output_path} ---")


if __name__ == "__main__":
    process_data(
        "base.csv",
        "item_result.csv",
        "roma_final_results_v4.csv"
    )
