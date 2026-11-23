import pandas as pd
import numpy as np
import re
import os

# ==========================================
# 1. Field mapping
# ==========================================
BASE_INFO_MAPPING = {
    'family_history': ['lung_cancer_family_history', 'familial_lung_cancer', 'family_history'],
    'age': ['app_pat_age', 'age'],
    'gender': ['pat_sex', 'sex', 'gender']
}

LLM_ITEM_MAPPING = {
    'emphysema': ['emphysema'],
    'nodule_size': ['nodule_size'],
    'nodule_type': ['nodule_type'],
    'nodule_location': ['nodule_location'],
    'nodule_count': ['nodule_count'],
    'spiculation': ['spiculation']
}


# ==========================================
# 2. Utility functions
# ==========================================
def normalize_str(s):
    if pd.isna(s): return ""
    return re.sub(r'[\(\)\（\）\[\]\【\】\{\}\s]', '', str(s)).lower()


def get_category(name, mapping_dict):
    norm_name = normalize_str(name)
    for key, patterns in mapping_dict.items():
        for p in patterns:
            if normalize_str(p) in norm_name: return key
    return None


def clean_numeric(val):
    if pd.isna(val) or str(val).strip() == "" or "absent" in str(val) or "not_examined" in str(val):
        return np.nan
    try:
        match = re.search(r"[-+]?\d*\.\d+|\d+", str(val))
        return float(match.group()) if match else np.nan
    except:
        return np.nan


def clean_boolean_strict(val_list, keywords_true):
    if not val_list: return None
    for val in val_list:
        if pd.isna(val) or str(val).strip() == "": continue
        s = str(val).strip().lower()
        if any(kw in s for kw in keywords_true): return 1
        if any(kw in s for kw in ['absent', 'denies', 'not seen', '0', 'no_nodule']): return 0
    return None


# ==========================================
# 3. Brock model core logic
# ==========================================
def calculate_brock_probability(params):
    # [Eligibility check]
    nodule_fields = [str(params.get('nodule_size')), str(params.get('nodule_type'))]
    if any("no_nodule" in x for x in nodule_fields):
        return "NOT_APPLICABLE", "no lung nodule found; model not applicable"

    # Core-indicator missing check
    required_keys = ['age', 'gender', 'emphysema', 'nodule_size', 'nodule_type', 'nodule_location', 'nodule_count',
                     'spiculation']
    if any(pd.isna(params.get(k)) for k in required_keys):
        return None, "core_indicator_missing"

    # Family-history logic: default to 0 when missing
    fam_hist_val = params.get('family_history') if pd.notna(params.get('family_history')) else 0

    if "perifissural" in str(params['nodule_type']):
        return 0.0, "perifissural nodule (low risk)"

    try:
        # Brock model coefficients
        intercept, b_age, b_sex, b_fam, b_emp, b_size, b_gg, b_ps, b_loc, b_cnt, b_spi = \
            -6.6144, 0.0287, 0.6011, 0.2961, 0.2953, -5.5537, -0.1276, 0.3770, 0.6009, -0.0824, 0.9309

        logit = intercept
        logit += b_age * (params['age'] - 62)
        logit += b_sex * (1 if any(kw in str(params['gender']) for kw in ['female', 'f', '2']) else 0)
        logit += b_fam * fam_hist_val
        logit += b_emp * params['emphysema']
        logit += b_size * ((float(params['nodule_size']) ** -0.5) - (4 ** -0.5))

        n_type = str(params['nodule_type'])
        if "ground_glass" in n_type:
            logit += b_gg
        elif "part_solid" in n_type:
            logit += b_ps

        logit += b_loc * params['nodule_location']
        logit += b_cnt * (float(params['nodule_count']) - 4)
        logit += b_spi * params['spiculation']

        return 1 / (1 + np.exp(-logit)), round(logit, 4)
    except:
        return None, "computation_anomaly"


# ==========================================
# 4. Main processing routine
# ==========================================
def process_brock_data(base_path, item_path, output_path):
    print("--- Starting Brock assessment (all fields retained) ---")

    def read_csv_safe(path):
        if not os.path.exists(path): return pd.DataFrame()
        for enc in ['utf-8-sig', 'gbk', 'utf-8']:
            try:
                return pd.read_csv(path, encoding=enc)
            except:
                continue
        return pd.DataFrame()

    base_df = read_csv_safe(base_path)
    item_df = read_csv_safe(item_path)

    for df in [base_df, item_df]:
        if not df.empty:
            df['app_no'] = df['app_no'].astype(str).str.strip()
            df['pat_id'] = df['pat_id'].astype(str).str.strip()

    # 1. Pivot LLM data (select LLM-extracted fields from item_result.csv)
    llm_records = []
    if not item_df.empty:
        llm_df = item_df.copy()
        llm_df['cat'] = llm_df['res_itm_name'].apply(lambda x: get_category(x, LLM_ITEM_MAPPING))
        for (app_no, pat_id), group in llm_df.dropna(subset=['cat']).groupby(['app_no', 'pat_id']):
            d = group.set_index('cat')['res_value'].to_dict()
            llm_records.append({
                'app_no': app_no, 'pat_id': pat_id,
                'emp_raw': d.get('emphysema'), 'size_raw': d.get('nodule_size'),
                'type_raw': d.get('nodule_type'), 'loc_raw': d.get('nodule_location'),
                'cnt_raw': d.get('nodule_count'), 'spi_raw': d.get('spiculation')
            })

    # 2. Pivot exam indicators to wide table (focus on family history)
    item_records = []
    if not item_df.empty:
        item_df['cat'] = item_df['res_itm_name'].apply(lambda x: get_category(x, BASE_INFO_MAPPING))
        for (app_no, pat_id), group in item_df.dropna(subset=['cat']).groupby(['app_no', 'pat_id']):
            d = group.set_index('cat')['res_value'].to_dict()
            item_records.append({'app_no': app_no, 'pat_id': pat_id, 'fam_hist_raw': d.get('family_history')})

    # ==========================
    # Core merge logic: keep the entire Base table
    # ==========================
    # --- Add safe conversion logic ---
    df_llm = pd.DataFrame(llm_records)
    if 'app_no' not in df_llm.columns: df_llm['app_no'] = pd.Series(dtype='str')
    if 'pat_id' not in df_llm.columns: df_llm['pat_id'] = pd.Series(dtype='str')

    df_item = pd.DataFrame(item_records)
    if 'app_no' not in df_item.columns: df_item['app_no'] = pd.Series(dtype='str')
    if 'pat_id' not in df_item.columns: df_item['pat_id'] = pd.Series(dtype='str')

    # First left join: merge nodule indicators
    final = pd.merge(base_df, df_llm, on=['app_no', 'pat_id'], how='left')
    # Second left join: merge family-history indicators
    final = pd.merge(final, df_item, on=['app_no', 'pat_id'], how='left')

    GAUGE_REF = "0%-5%|5%-10%|10%-100%"
    results = []

    for _, r in final.iterrows():
        age = clean_numeric(r.get('app_pat_age'))
        gender = r.get('pat_sex')
        fam_hist_val = clean_boolean_strict([r.get('fam_hist_raw')], ['present', 'yes', '1'])

        params = {
            'age': age, 'gender': gender, 'family_history': fam_hist_val,
            'emphysema': clean_boolean_strict([r.get('emp_raw')], ['present']),
            'nodule_size': clean_numeric(r.get('size_raw')) if r.get('size_raw') != "no_nodule" else "no_nodule",
            'nodule_type': r.get('type_raw'),
            'nodule_location': clean_boolean_strict([r.get('loc_raw')], ['upper_lobe']),
            'nodule_count': clean_numeric(r.get('cnt_raw')) if r.get('cnt_raw') != "no_nodule" else "no_nodule",
            'spiculation': clean_boolean_strict([r.get('spi_raw')], ['present'])
        }

        prob, extra = calculate_brock_probability(params)

        # Wrap each result row
        row_res = {
            'pat_id': r.get('pat_id'), 'app_no': r.get('app_no'), 'pat_name': r.get('pat_name'),
            'model_name': 'Brock Lung Nodule Risk Model'
        }

        if prob == "NOT_APPLICABLE":
            row_res.update({
                'code': '-', 'gauge_reference': '-',
                'risk_probability': "-", 'risk_level': "-", 'assessment_note': extra, 'logit_intermediate': "-",
                'emphysema': r.get('emp_raw') if pd.notna(r.get('emp_raw')) else "absent",
                'nodule_size (mm)': "no_nodule", 'nodule_type': "no_nodule",
                'nodule_upper_lobe': "no_nodule", 'nodule_count': "no_nodule", 'spiculation': "no_nodule"
            })
        else:
            p_val = f"{prob * 100:.2f}%" if isinstance(prob, float) else "-"
            risk, code = "-", "-"
            if isinstance(prob, float):
                if prob < 0.05:
                    risk, code = "low risk", "C34.3.1"
                elif prob < 0.10:
                    risk, code = "medium risk", "C34.3.2"
                else:
                    risk, code = "high risk", "C34.3.3"

            row_res.update({
                'code': code, 'gauge_reference': GAUGE_REF if isinstance(prob, float) else "-",
                'risk_probability': p_val, 'risk_level': risk, 'assessment_note': extra,
                'logit_intermediate': extra if prob is not None else "-",
                'emphysema': r.get('emp_raw'), 'nodule_size (mm)': r.get('size_raw'),
                'nodule_type': r.get('type_raw'), 'nodule_upper_lobe': r.get('loc_raw'),
                'nodule_count': r.get('cnt_raw'), 'spiculation': r.get('spi_raw')
            })

        row_res.update({
            'sex': gender if pd.notna(gender) else "unknown",
            'age': int(age) if pd.notna(age) else "unknown",
            'lung_cancer_family_history': "present" if fam_hist_val == 1 else ("absent" if fam_hist_val == 0 else "missing (defaulted to none)")
        })
        results.append(row_res)

    # Export all fields
    df_final = pd.DataFrame(results)
    ordered_cols = [
        'pat_id', 'app_no', 'pat_name', 'model_name', 'code', 'gauge_reference',
        'risk_probability', 'risk_level', 'logit_intermediate', 'assessment_note',
        'sex', 'age', 'lung_cancer_family_history', 'emphysema', 'nodule_size (mm)',
        'nodule_type', 'nodule_upper_lobe', 'nodule_count', 'spiculation'
    ]
    df_final[ordered_cols].to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"--- Assessment complete, generated: {output_path} ---")


if __name__ == "__main__":
    process_brock_data("base.csv", "item_result.csv", "brock_risk_output.csv")
