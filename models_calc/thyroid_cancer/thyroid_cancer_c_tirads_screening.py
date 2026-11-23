import pandas as pd
import numpy as np
import re
import os
import warnings

# Ignore DtypeWarning
warnings.filterwarnings('ignore', category=pd.errors.DtypeWarning)

# ==========================================
# 1. Field mapping (matches the LLM extraction script output)
# ==========================================
LLM_ITEM_MAPPING = {
    'orientation': ['vertical_position'],
    'composition': ['solid'],
    'echogenicity': ['markedly_hypoechoic'],
    'margin': ['irregular_margin'],
    'microcalc': ['microcalcification'],
    'comet_tail': ['comet_tail_sign']
}

# Routine information mapping (from item_result.csv)
BASE_INFO_MAPPING = {
    'family_history': ['family_history', 'thyroid_cancer_family_history']
}


# ==========================================
# 2. Basic utility functions
# ==========================================
def read_csv_safe(path):
    if not os.path.exists(path): return pd.DataFrame()
    dtype_spec = {'app_no': str, 'pat_id': str}
    for enc in ['utf-8-sig', 'gbk', 'utf-8']:
        try:
            df = pd.read_csv(path, encoding=enc, dtype=dtype_spec)
            df.columns = df.columns.str.strip()
            if 'app_no' in df.columns: df['app_no'] = df['app_no'].str.strip()
            if 'pat_id' in df.columns: df['pat_id'] = df['pat_id'].str.strip()
            return df
        except:
            continue
    return pd.DataFrame()


def normalize_str(s):
    if pd.isna(s): return ""
    return re.sub(r'[\s]', '', str(s)).lower()


def get_category(name, mapping_dict):
    norm_name = normalize_str(name)
    for key, patterns in mapping_dict.items():
        for p in patterns:
            if normalize_str(p) == norm_name: return key
    return None


def is_positive(val):
    """Determine whether the LLM-extracted value is positive (present)"""
    s = str(val).strip()
    return s == "present"


# ==========================================
# 3. C-TIRADS scoring core logic
# ==========================================
def calculate_thyroid_risk(params):
    # [Applicability review]
    # Trigger the not-applicable logic when the core nodule indicator says "no thyroid nodule"
    nodule_fields = [str(params.get('orientation')), str(params.get('composition'))]
    if any("no_thyroid_nodule" in x for x in nodule_fields):
        return 0, "C-TR1 (no nodule, 0% malignancy)", "C73.1.1"

    # Check whether any scoring indicator exists (the LLM must at least return something)
    required_keys = ['orientation', 'composition', 'echogenicity', 'margin', 'microcalc', 'comet_tail']
    if all(params.get(k) is None for k in required_keys):
        # Fix 2: return all 3 values to prevent unpack errors
        return None, "core_indicator_missing", "-"

    # Scoring logic
    score = 0
    # Malignant feature +1
    if is_positive(params.get('orientation')): score += 1
    if is_positive(params.get('composition')): score += 1
    if is_positive(params.get('echogenicity')): score += 1
    if is_positive(params.get('margin')): score += 1
    has_microcalc = is_positive(params.get('microcalc'))
    if has_microcalc: score += 1

    # The original records microcalcification first when it co-occurs with comet-tail artifact, without double counting.
    if is_positive(params.get('comet_tail')) and not has_microcalc: score -= 1

    if score == -1:
        level, code = "C-TR2 (0% malignancy)", "C73.1.1"
    elif score == 0:
        level, code = "C-TR3 (<2% malignancy)", "C73.1.1"
    elif score == 1:
        level, code = "C-TR4A (2%-10% malignancy)", "C73.1.2"
    elif score == 2:
        level, code = "C-TR4B (10%-50% malignancy)", "C73.1.2"
    elif score in (3, 4):
        level, code = "C-TR4C (50%-90% malignancy)", "C73.1.3"
    else:
        level, code = "C-TR5 (>90% malignancy)", "C73.1.3"

    return score, level, code


# ==========================================
# 4. Main processing routine
# ==========================================
def process_thyroid_data(base_path, item_path, output_path):
    print("running thyroid C-TIRADS assessment...")

    base_df = read_csv_safe(base_path)
    item_df = read_csv_safe(item_path)

    if base_df.empty:
        print("Error: cannot read the base data file.")
        return

    # 1. Pivot LLM data (select LLM-extracted fields from item_result.csv)
    llm_records = []
    if not item_df.empty:
        llm_df = item_df.copy()
        llm_df['cat'] = llm_df['res_itm_name'].apply(lambda x: get_category(x, LLM_ITEM_MAPPING))
        llm_filtered = llm_df.dropna(subset=['cat'])
        for (app_no, pat_id), group in llm_filtered.groupby(['app_no', 'pat_id']):
            d = group.set_index('cat')['res_value'].to_dict()
            llm_records.append({
                'app_no': app_no, 'pat_id': pat_id,
                **{f"{k}_raw": v for k, v in d.items()}
            })

    # 2. Pivot routine indicators (family history)
    item_records = []
    if not item_df.empty:
        item_df['cat'] = item_df['res_itm_name'].apply(lambda x: get_category(x, BASE_INFO_MAPPING))
        for (app_no, pat_id), group in item_df.dropna(subset=['cat']).groupby(['app_no', 'pat_id']):
            d = group.set_index('cat')['res_value'].to_dict()
            item_records.append({'app_no': app_no, 'pat_id': pat_id, 'fam_hist_raw': d.get('family_history')})

    # 3. Two left joins keeping the full base table
    # Fix 3: guard against empty lists; build an empty table with correct headers even without data or family history, preventing merge errors
    df_llm = pd.DataFrame(llm_records) if llm_records else pd.DataFrame(columns=['app_no', 'pat_id'])
    df_item = pd.DataFrame(item_records) if item_records else pd.DataFrame(columns=['app_no', 'pat_id'])

    final = pd.merge(base_df, df_llm, on=['app_no', 'pat_id'], how='left')
    final = pd.merge(final, df_item, on=['app_no', 'pat_id'], how='left')

    results = []
    for _, r in final.iterrows():
        # Prepare scoring parameters
        params = {
            'orientation': r.get('orientation_raw'),
            'composition': r.get('composition_raw'),
            'echogenicity': r.get('echogenicity_raw'),
            'margin': r.get('margin_raw'),
            'microcalc': r.get('microcalc_raw'),
            'comet_tail': r.get('comet_tail_raw')
        }

        # Fetch family-history determination (display only)
        fam_raw = r.get('fam_hist_raw')
        fam_status = "present" if fam_raw in ['1', 'present', 'yes'] else ("absent" if pd.notna(fam_raw) else "missing (defaulted to none)")

        score_res, level, code = calculate_thyroid_risk(params)

        row_res = {
            'pat_id': r.get('pat_id'),
            'app_no': r.get('app_no'),
            'pat_name': r.get('pat_name') or "unknown",
            'model_name': 'Thyroid Cancer C-TIRADS 2020',
            'prediction_horizon': 'current'
        }

        if score_res == "NOT_APPLICABLE":
            row_res.update({
                'predicted_score': '-', 'risk_level': '-', 'code': '-', 'gauge_reference': '-',
                'assessment_note': level,
                'vertical_position': 'no_thyroid_nodule', 'composition': 'no_thyroid_nodule',
                'echogenicity': 'no_thyroid_nodule', 'margin': 'no_thyroid_nodule',
                'microcalcification': 'no_thyroid_nodule', 'comet_tail': 'no_thyroid_nodule'
            })
        else:
            row_res.update({
                'predicted_score': score_res if score_res is not None else '-',
                'risk_level': level,
                'code': code,
                'gauge_reference': "-1|0|1|2|3-4|5",
                'assessment_note': "data_complete" if score_res is not None else "indicator_missing",
                'vertical_position': r.get('orientation_raw') if pd.notna(r.get('orientation_raw')) else "missing",
                'composition': r.get('composition_raw') if pd.notna(r.get('composition_raw')) else "missing",
                'echogenicity': r.get('echogenicity_raw') if pd.notna(r.get('echogenicity_raw')) else "missing",
                'margin': r.get('margin_raw') if pd.notna(r.get('margin_raw')) else "missing",
                'microcalcification': r.get('microcalc_raw') if pd.notna(r.get('microcalc_raw')) else "missing",
                'comet_tail': r.get('comet_tail_raw') if pd.notna(r.get('comet_tail_raw')) else "missing"
            })

        row_res['family_history'] = fam_status
        results.append(row_res)

    # Export
    df_output = pd.DataFrame(results)
    ordered_cols = [
        'pat_id', 'app_no', 'pat_name', 'model_name', 'code', 'gauge_reference',
        'predicted_score', 'risk_level', 'prediction_horizon', 'assessment_note',
        'vertical_position', 'composition', 'echogenicity', 'margin', 'microcalcification', 'comet_tail', 'family_history'
    ]
    df_output[ordered_cols].to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f" processing complete; results saved to: {output_path}")


if __name__ == "__main__":
    process_thyroid_data("base.csv", "item_result.csv", "thyroid_TIRADS_results.csv")
