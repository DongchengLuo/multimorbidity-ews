import pandas as pd
import numpy as np
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model_unit_utils import align_d_dimer_pair

# ==========================================
# 1. Field mapping (per IMPROVEDD metrics)
# ==========================================
FIELD_MAPPING = {
    'age': ['app_pat_age', 'age'],  
    'gender': ['pat_sex', 'sex', 'gender'],
    'vte_history': ['prior_vte', 'thrombosis_history', 'previous_vte', 'prior_DVT'],  
    'thrombophilia': ['known_thrombophilia', 'thrombophilia', 'coagulopathy_history'],  
    'paralysis': ['lower_limb_paralysis', 'paraplegia'],  
    'cancer': ['active_cancer', 'active_malignancy', 'current_cancer'],  
    'immobilized': ['immobilization_ge_7d', 'prolonged_bed_rest', 'immobilized_7d'],  
    'icu_ccu': ['icu_ccu_admission', 'ICU', 'icu_stay'],  
    'd_dimer': ['plasma_D_dimer_quantitative'],  
    'd_dimer_uln': ['D_dimer_upper_limit', 'd_dimer_uln', 'dd_limit']    
}

# ==========================================
# 2. Helper utility functions
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

def get_first_numeric(val_list):
    if not val_list: return np.nan
    return clean_numeric(val_list[0])

def eval_status(vals):
    """Questionnaire items must be explicit yes/no; missing, ambiguous or conflicting returns None."""
    if not vals:
        return None
    states = []
    for val in vals:
        s = normalize_str(val)
        negative = s in {'0', 'no', 'absent', 'negative', 'false'} or any(
            token in s for token in ('denies', 'not seen', 'none', 'not_admitted', 'not_immobilized')
        )
        positive = s in {'1', 'yes', 'present', 'positive', 'true'}
        if negative:
            states.append(False)
        elif positive:
            states.append(True)
        else:
            return None
    return states[0] if all(state == states[0] for state in states) else None


def format_status(val):
    if val is None or pd.isna(val):
        return 'missing'
    return 'present' if bool(val) else 'absent'


def improvedd_age_points(age):
    """IMPROVEDD originally scores 1 point for age >= 60."""
    return 1 if pd.notna(age) and age >= 60 else 0

# ==========================================
# 3. Main data-processing routine
# ==========================================
def process_improvedd_data(base_path, item_path, output_path):
    print("--- Starting IMPROVEDD VTE risk model ---")

    def read_csv_safe(path):
        for enc in ['utf-8-sig', 'gb18030', 'gbk', 'utf-8']:
            try: return pd.read_csv(path, encoding=enc, low_memory=False)
            except: continue
        return pd.DataFrame()

    base_df = read_csv_safe(base_path)
    item_df = read_csv_safe(item_path)
    if base_df.empty: return

    base_df['app_no'] = base_df['app_no'].astype(str).str.strip()
    base_df['pat_id'] = base_df['pat_id'].astype(str).str.strip()

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
                unit_pool[item['cat']].append(
                    '' if pd.isna(item.get('res_unit', '')) else str(item.get('res_unit', ''))
                )

            d_values = data_pool.get('d_dimer', [])
            d_units = unit_pool.get('d_dimer', [])
            uln_values = data_pool.get('d_dimer_uln', [])
            uln_units = unit_pool.get('d_dimer_uln', [])
            row['d_dimer_raw'] = d_values[0] if d_values else np.nan
            row['d_dimer_unit'] = d_units[0] if d_units else ''
            row['uln_raw'] = uln_values[0] if uln_values else np.nan
            row['uln_unit'] = uln_units[0] if uln_units else ''
            row['vte'] = eval_status(data_pool.get('vte_history', []))
            row['thromb'] = eval_status(data_pool.get('thrombophilia', []))
            row['paralysis'] = eval_status(data_pool.get('paralysis', []))
            row['cancer'] = eval_status(data_pool.get('cancer', []))
            row['immob'] = eval_status(data_pool.get('immobilized', []))
            row['icu'] = eval_status(data_pool.get('icu_ccu', []))
            records.append(row)

    wide_items = pd.DataFrame(records)
    final = pd.merge(base_df, wide_items, on=['app_no', 'pat_id'], how='left') if not wide_items.empty else base_df.copy()

    results = []
    for _, r in final.iterrows():
        age = clean_numeric(r.get('app_pat_age'))
        
        vte_val = r.get('vte')
        thromb_val = r.get('thromb')
        para_val = r.get('paralysis')
        can_val = r.get('cancer')
        imm_val = r.get('immob')
        icu_val = r.get('icu')

        d_dimer, uln, unit_detail = align_d_dimer_pair(
            r.get('d_dimer_raw'), r.get('d_dimer_unit', ''),
            r.get('uln_raw'), r.get('uln_unit', ''),
        )

        res_row = {
            'pat_id': r.get('pat_id'),
            'app_no': r.get('app_no'),
            'pat_name': r.get('pat_name', 'unknown'),
            'age': int(age) if not pd.isna(age) else 'missing',
            'D_dimer_raw_value': r.get('d_dimer_raw', 'missing'),
            'D_dimer_raw_unit': r.get('d_dimer_unit', ''),
            'D_dimer_ULN_raw_value': r.get('uln_raw', 'missing'),
            'D_dimer_ULN_raw_unit': r.get('uln_unit', ''),
            'D_dimer_aligned_value': d_dimer if d_dimer is not None else 'missing',
            'D_dimer_ULN_aligned_value': uln if uln is not None else 'missing',
            'unit handling': unit_detail,
            'prior_vte': format_status(vte_val),
            'known_thrombophilia': format_status(thromb_val),
            'lower_limb_paralysis': format_status(para_val),
            'active_cancer': format_status(can_val),
            'immobilization_ge_7d': format_status(imm_val),
            'icu_ccu_admission': format_status(icu_val)
        }

        def present(status):
            return status is True or status == 1

        # Project compatibility mode: unscored questionnaire items count as 0 and no longer block computation.
        score = 0
        score += 3 if present(vte_val) else 0
        score += 2 if present(thromb_val) else 0
        score += 2 if present(para_val) else 0
        score += 2 if present(can_val) else 0
        score += improvedd_age_points(age)
        score += 1 if present(icu_val) else 0
        score += 1 if present(imm_val) else 0
        if d_dimer is not None and uln is not None and d_dimer >= 2 * uln:
            score += 2

        # Gibson 2017 defines only 0-1 as low risk and >=2 as at-risk.
        if score <= 1:
            level, code = "low risk", "I82.2.1"
        else:
            level, code = "high risk", "I82.2.3"

        res_row.update({
            'predicted_score': score,
            'risk_level': level,
            'code': code,
            'calculation_status': 'computed',
            'not_computed_reason': '',
        })

        # Fixed information columns
        res_row.update({
            'model_name': 'IMPROVEDD VTE Risk Score',
            'prediction_horizon_dimension': 'next 77 days',
            'calculation_basis': 'IMPROVE Factors + D-dimer (>=2x ULN)',
            'predicted_disease': 'Venous Thromboembolism (VTE)'
        })
        results.append(res_row)

    if results:
        res_df = pd.DataFrame(results)
        # Adjust the final display order
        final_cols = [
            'pat_id', 'app_no', 'pat_name', 'age',
            'D_dimer_raw_value', 'D_dimer_raw_unit',
            'D_dimer_ULN_raw_value', 'D_dimer_ULN_raw_unit',
            'D_dimer_aligned_value', 'D_dimer_ULN_aligned_value', 'unit handling',
            'prior_vte', 'known_thrombophilia', 'lower_limb_paralysis', 'active_cancer', 'immobilization_ge_7d', 'icu_ccu_admission',
            'predicted_score', 'risk_level', 'code', 'calculation_status', 'not_computed_reason',
            'prediction_horizon_dimension', 'predicted_disease'
        ]
        res_df = res_df.reindex(columns=final_cols)
        res_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"Assessment complete! Results exported to {output_path}")

if __name__ == "__main__":
    process_improvedd_data("base.csv", "item_result.csv", "improvedd_vte_77d_results.csv")
