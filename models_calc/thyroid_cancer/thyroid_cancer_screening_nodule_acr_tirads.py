import pandas as pd
import numpy as np
import re
import os
import sys
import warnings

# Suppress unnecessary dtype warnings
warnings.filterwarnings('ignore', category=pd.errors.DtypeWarning)

# ==========================================
# 1. Field mapping (exact match)
# ==========================================
FIELD_MAPPING = {
    'age': ['app_pat_age', 'age'],
    'gender': ['pat_sex', 'sex'],
    'composition': ['composition', 'nodule_composition', 'cystic_solid', 'structure'],
    'echogenicity': ['echogenicity', 'echogenicity_level', 'echogenicity_intensity'],
    'shape': ['shape', 'morphology', 'taller_than_wide', 'margin_taller_than_wide'],
    'margin': ['margin', 'boundary', 'capsule'],
    'echogenic_foci': ['echogenic_foci', 'calcification', 'punctate_echogenic_foci', 'strong_echo', 'calcification_foci'],
    'nodule_size': ['nodule_size', 'max_diameter', 'diameter', 'size'],
    'nodule_pos': ['site', 'location', 'examined_site']
}

# ==========================================
# 2. Utility functions
# ==========================================
def read_csv_safe(path, **kwargs):
    encodings = ['utf-8-sig', 'gbk', 'gb18030', 'utf-8']
    for enc in encodings:
        try:
            df = pd.read_csv(path, encoding=enc, low_memory=False, **kwargs)
            return df
        except:
            continue
    return pd.DataFrame()

def normalize_str(s):
    if pd.isna(s): return ""
    return re.sub(r'[\(\)\（\）\[\]\【\】\{\}\s]', '', str(s)).lower()

def get_category(name):
    norm_name = normalize_str(name)
    for key, patterns in FIELD_MAPPING.items():
        for p in patterns:
            # Change: switch to exact match
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
# 3. ACR TI-RADS core scoring logic
# ==========================================
def calculate_ti_rads_score(comp, echo, shape, margin, foci_str):
    # Full scoring is impossible when any core category description is empty or NaN
    import pandas as pd
    values = [comp, echo, shape, margin]
    if any(pd.isna(v) for v in values) or not all(values):
        return None, "-", "-", "-"
    
    # Ensure all values are strings
    comp = str(comp)
    echo = str(echo)
    shape = str(shape)
    margin = str(margin)
    foci_str = str(foci_str) if not pd.isna(foci_str) else ""

    score = 0
    
    # 1. Composition
    if "spongiform" in comp: return 0, "TR1", "C73.2.1", ""
    
    c_score = 0
    if "mixed" in comp or "cystic_solid" in comp: c_score = 1
    elif "solid" in comp: c_score = 2
    score += c_score

    # 2. Echogenicity
    e_score = 1  # isoechoic/hyperechoic defaults to 1
    if "markedly_hypoechoic" in echo: e_score = 3
    elif "hypoechoic" in echo: e_score = 2
    elif "anechoic" in echo: e_score = 0
    score += e_score

    # 3. Shape
    s_score = 3 if any(x in shape for x in ["vertical", "longitudinal", "taller_than_wide", "taller"]) else 0
    score += s_score

    # 4. Margin
    m_score = 0
    if "extrathyroidal_extension" in margin: m_score = 3
    elif any(x in margin for x in ["lobulated", "irregular"]): m_score = 2
    score += m_score

    # 5. Echogenic foci
    f_score = 0
    if any(x in foci_str for x in ["punctate", "microcalcification"]): f_score += 3
    if any(x in foci_str for x in ["margin", "rim"]): f_score += 2
    if "coarse" in foci_str: f_score += 1
    score += f_score

    # Risk-tiering logic
    if score == 0:
        level = "TR1"
        code = "C73.2.1"
    elif score == 2:
        level = "TR2"
        code = "C73.2.1"
    elif score == 3:
        level = "TR3"
        code = "C73.2.1"
    elif 4 <= score <= 6:
        level = "TR4"
        code = "C73.2.2"
    elif score >= 7:
        level = "TR5"
        code = "C73.2.3"
    else:
        level = "cannot grade per source (feature combination yields 1 point)"
        code = "-"

    return score, level, code, ""

# ==========================================
# 4. Main data-processing routine
# ==========================================
def process_ti_rads_data(base_path, item_path, output_path):
    print(f" running thyroid ACR-TIRADS screening...")

    base_df = read_csv_safe(base_path)
    item_df = read_csv_safe(item_path)

    if base_df.empty:
        print("Error: cannot read the base data file.")
        return

    # Normalize ID types
    base_df['app_no'] = base_df['app_no'].astype(str).str.strip()
    base_df['pat_id'] = base_df['pat_id'].astype(str).str.strip()

    # 1. Pivot exam indicators to a wide table
    records = []
    if not item_df.empty:
        item_df['cat'] = item_df['res_itm_name'].apply(get_category)
        filtered = item_df.dropna(subset=['cat', 'res_value']).copy()
        
        filtered['app_no'] = filtered['app_no'].astype(str).str.strip()
        filtered['pat_id'] = filtered['pat_id'].astype(str).str.strip()

        for (app_no, pat_id), group in filtered.groupby(['app_no', 'pat_id']):
            row = {'app_no': app_no, 'pat_id': pat_id}
            data_pool = {k: [] for k in FIELD_MAPPING.keys()}
            for _, item in group.iterrows():
                data_pool[item['cat']].append(str(item['res_value']))

            row['raw_comp'] = " ".join(data_pool.get('composition', []))
            row['raw_echo'] = " ".join(data_pool.get('echogenicity', []))
            row['raw_shape'] = " ".join(data_pool.get('shape', []))
            row['raw_margin'] = " ".join(data_pool.get('margin', []))
            row['raw_foci'] = " ".join(data_pool.get('echogenic_foci', []))
            row['raw_pos'] = " ".join(data_pool.get('nodule_pos', []))
            
            size_vals = [clean_numeric(v) for v in data_pool.get('nodule_size', [])]
            valid_sizes = [v for v in size_vals if pd.notna(v)]
            row['max_size'] = max(valid_sizes) if valid_sizes else np.nan
            records.append(row)

    wide_items = pd.DataFrame(records) if records else pd.DataFrame(columns=['app_no', 'pat_id'])

    # 2. Merge basic information (left join)
    final = pd.merge(base_df, wide_items, on=['app_no', 'pat_id'], how='left')

    results = []
    for _, r in final.iterrows():
        comp = r.get('raw_comp', "")
        echo = r.get('raw_echo', "")
        shape = r.get('raw_shape', "")
        margin = r.get('raw_margin', "")
        foci = r.get('raw_foci', "")
        max_size = r.get('max_size', np.nan)

        # Score computation
        score, level, code, _ = calculate_ti_rads_score(comp, echo, shape, margin, foci)

        results.append({
            'pat_id': r['pat_id'],
            'pat_name': r.get('pat_name') or r.get('name') or "unknown",
            'app_no': r['app_no'],
            'prediction_horizon': 'current',
            # --- Raw descriptive indicators ---
            'site': r.get('raw_pos') if pd.notna(r.get('raw_pos')) and r.get('raw_pos') != "" else "missing",
            'composition': comp if comp != "" else "missing",
            'echogenicity': echo if echo != "" else "missing",
            'morphology': shape if shape != "" else "missing",
            'margin': margin if margin != "" else "missing",
            'calcification_status': foci if foci != "" else "missing",
            'size (cm)': f"{max_size:.2f}" if pd.notna(max_size) else "missing",
            # --- Model prediction results ---
            'predicted_score': score if score is not None else "-",
            'risk_level': level,
            'code': code,
            'gauge_reference': "0|2|3|4-6|7-17",
            'model_name': 'Thyroid Cancer - ACR-TIRADS'
        })

    # 3. Output results
    if results:
        pd.DataFrame(results).to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f" processing complete! Results saved to: {output_path}, {len(results)} records.")
    else:
        print(" no assessment results generated.")

if __name__ == "__main__":
    process_ti_rads_data("base.csv", "item_result.csv", "thyroid_ACR_results.csv")
