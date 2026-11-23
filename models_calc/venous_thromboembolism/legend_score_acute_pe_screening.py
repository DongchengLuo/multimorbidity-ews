import pandas as pd
import numpy as np
import re

# ==========================================
# 1. Field mapping
# ==========================================
FIELD_MAPPING = {
    'age': ['app_pat_age', 'age'],
    'gender': ['pat_sex', 'sex'],
    'active_cancer': ['active_cancer', 'malignancy_history', 'tumor_history'],
    'd_dimer': ['d_dimer', 'D-Dimer', 'D_dimer', 'D-D'],
    'dvt_signs': ['DVT_symptoms', 'leg_swelling', 'leg_pain', 'DVT_signs', 'dvt_signs'],
    'vte_history': ['prior_VTE_history', 'prior_thrombosis_history', 'prior_pulmonary_embolism', 'prior_DVT', 'vte_history'],
    'recent_event': ['recent_surgery', 'recent_trauma', 'recent_immobilization', 'surgery_history', 'immobilization_history', 'recent_event']
}


# ==========================================
# 2. Basic cleaning logic
# ==========================================
def normalize_str(s):
    if pd.isna(s): return ""
    return re.sub(r'[\(\)\（\）\[\]\【\】\{\}\s]', '', str(s)).lower()


def get_category(name):
    norm_name = normalize_str(name)
    for key, patterns in FIELD_MAPPING.items():
        for p in patterns:
            # Strict exact matching; fuzzy mapping forbidden
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
    # Take the first valid value in the list; never average
    if not val_list: 
        return np.nan
    return clean_numeric(val_list[0])


def is_positive(val):
    if pd.isna(val): return False
    s = str(val).strip()
    return any(kw in s for kw in ['yes', 'present', '1', 'positive'])


def eval_status(vals):
    # An empty list means no record at all -> missing (None)
    if not vals: return None  
    return any(is_positive(v) for v in vals)


# ==========================================
# 3. Legend score core logic (binary)
# ==========================================
def calculate_legend_score(cancer_pos, d_dimer_val, dvt_pos, vte_pos, recent_pos):
    """
    Compute the Legend score (original version)
    Weights: cancer (1), D-dimer >= 1000 (3), others (2)
    """
    s_cancer = 1 if cancer_pos else 0
    s_ddimer = 3 if (not pd.isna(d_dimer_val) and d_dimer_val >= 1000) else 0
    s_dvt = 2 if dvt_pos else 0
    s_vte = 2 if vte_pos else 0
    s_recent = 2 if recent_pos else 0

    total = s_cancer + s_ddimer + s_dvt + s_vte + s_recent
    return total


# ==========================================
# 4. Main data-processing routine
# ==========================================
def process_data(base_path, item_path, output_path):
    print("--- Starting Legend score model (binary) ---")

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
        print("error: cannot read base data file or it is empty")
        return

    base_df['app_no'] = base_df['app_no'].astype(str).str.strip()
    base_df['pat_id'] = base_df['pat_id'].astype(str).str.strip()

    # 1. Aggregate indicators
    records = []
    if not item_df.empty:
        item_df['cat'] = item_df['res_itm_name'].apply(get_category)
        filtered = item_df.dropna(subset=['cat', 'res_value']).copy()

        for (app_no, pat_id), group in filtered.groupby(['app_no', 'pat_id']):
            row = {'app_no': str(app_no).strip(), 'pat_id_merge': str(pat_id).strip()}
            data_pool = {k: [] for k in FIELD_MAPPING.keys()}
            for _, item in group.iterrows():
                data_pool[item['cat']].append(str(item['res_value']))

            row['d_dimer_val'] = get_first_numeric(data_pool.get('d_dimer', []))
            row['has_cancer'] = eval_status(data_pool.get('active_cancer', []))
            row['has_dvt'] = eval_status(data_pool.get('dvt_signs', []))
            row['has_vte'] = eval_status(data_pool.get('vte_history', []))
            row['has_recent_event'] = eval_status(data_pool.get('recent_event', []))
            records.append(row)

    wide_items = pd.DataFrame(records)
    
    # Ensure patients without lab records are not skipped
    if wide_items.empty:
        final = base_df.copy()
    else:
        final = pd.merge(base_df, wide_items, left_on=['app_no', 'pat_id'], right_on=['app_no', 'pat_id_merge'], how='left')

    # 2. Generate results (binary classification logic)
    results = []
    for _, r in final.iterrows():
        age = clean_numeric(r.get('app_pat_age'))
        d_dimer = r.get('d_dimer_val')
        cancer = r.get('has_cancer')
        dvt = r.get('has_dvt')
        vte = r.get('has_vte')
        recent = r.get('has_recent_event')

        # Check whether core indicators are missing
        missing_items = []
        if pd.isna(age): missing_items.append("age")
        if pd.isna(d_dimer): missing_items.append("d_dimer")
        if cancer is None: missing_items.append("active_cancer")
        if dvt is None: missing_items.append("DVT_symptoms")
        if vte is None: missing_items.append("prior_VTE_history")
        if recent is None: missing_items.append("recent_event (surgery/trauma/immobilization)")

        if missing_items:
            results.append({
                'pat_id': r.get('pat_id'),
                'app_no': r.get('app_no'),
                'pat_name': r.get('pat_name', 'unknown'),
                'sex': r.get('pat_sex', 'missing') if pd.notna(r.get('pat_sex')) else 'missing',
                'age': int(age) if not pd.isna(age) else 'missing',
                'D_dimer_value': round(d_dimer, 2) if not pd.isna(d_dimer) else 'missing',
                'predicted_score': '-',
                'risk_determination': '-',
                'code': '-',
                'gauge_reference': "0-1|2-10",
                'recommendation': f"[Assessment refused] missing indicators: {', '.join(missing_items)}",
                'model_name': 'Legend Score (acute PE prediction)'
            })
            continue

        # Normal computation flow
        score = calculate_legend_score(cancer, d_dimer, dvt, vte, recent)

        # Binary threshold: >= 2 means PE-likely
        if score < 2:
            risk_label = "low risk"
            risk_code = "I82.3.1"
            advice = "low clinical probability; follow up with symptoms"
        else:
            risk_label = "high risk"
            risk_code = "I82.3.3"
            advice = "recommend immediate CTPA imaging"

        results.append({
            'pat_id': r.get('pat_id'),
            'app_no': r.get('app_no'),
            'pat_name': r.get('pat_name', 'unknown'),
            'sex': r.get('pat_sex', ''),
            'age': int(age),
            'D_dimer_value': round(d_dimer, 2),
            'predicted_score': score,
            'risk_determination': risk_label,
            'code': risk_code,
            'gauge_reference': "0-1|2-10",
            'recommendation': advice,
            'model_name': 'Legend Score (acute PE prediction)'
        })

    # 3. Export
    if results:
        res_df = pd.DataFrame(results)
        res_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"--- Assessment complete: {len(res_df)} results to {output_path} ---")
    else:
        print("--- Assessment complete, but no results (check data completeness) ---")


if __name__ == "__main__":
    process_data(
        "base.csv",
        "item_result.csv",
        "legend_pe_screening_result.csv"
    )
