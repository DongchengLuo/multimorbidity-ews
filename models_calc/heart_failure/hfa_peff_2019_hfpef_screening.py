import pandas as pd
import numpy as np
import re

# ==========================================
# 1. Field mapping and scoring rules
# ==========================================
FIELD_MAPPING = {
    # --- Baseline context (for tiered scoring) ---
    'app_pat_age': ['app_pat_age', 'age'],
    'pat_sex': ['pat_sex', 'sex'],
    'rhythm_status': ['rhythm', 'atrial_fibrillation', 'rhythm_status'], 
    
    # --- [Change 1] Add basic physiological indicators for BMI computation ---
    'height': ['height'],
    'weight': ['weight'],
    'bmi': ['BMI', 'bmi', 'body_mass_index'],

    # --- Functional domain ---
    'septal_e_prime': ['septal_e_prime', "septal e'"],
    'lateral_e_prime': ['lateral_e_prime', "lateral e'"],
    'avg_E_e_prime': ['avg_E_e_prime', "E/e' ratio"],  
    'tr_velocity': ['tr_velocity', 'tricuspid_regurgitation_velocity', 'TRV'],  
    'gls': ['gls', 'left_ventricular_global_longitudinal_strain', 'GLS'],  

    # --- Morphological domain ---
    'lavi': ['lavi', 'left_atrial_volume_index', 'LAVI'],  
    'lvmi': ['lvmi', 'left_ventricular_mass_index', 'LVMI'],  
    'rwt': ['rwt', 'relative_wall_thickness', 'RWT'],  
    'wall_thickness': ['wall_thickness', 'left_ventricular_posterior_wall_thickness'],  

    # --- Biomarkers ---
    'nt_pro_bnp': ['nt_pro_bnp', 'NT-proBNP', 'NT_proBNP'],
    'bnp': ['bnp', 'BNP']  
}

# ==========================================
# 2. Core cleaning and computation logic (pure computation)
# ==========================================
def clean_numeric(val):
    if pd.isna(val) or str(val).strip() == "": return np.nan
    match = re.search(r"[-+]?\d*\.\d+|\d+", str(val))
    return float(match.group()) if match else np.nan

def parse_rhythm(val):
    s = str(val).lower()
    return 'AF' if any(kw in s for kw in ['atrial_fibrillation', 'atrial_flutter', 'af', 'atrial']) else 'SR'

def calculate_hfa_peff_score(age, gender, rhythm, d):
    is_af = (rhythm == 'AF')
    
    # --- Functional domain ---
    f_maj, f_min = False, False
    if pd.notna(age):
        s_e, l_e = d.get('septal_e_prime', 99), d.get('lateral_e_prime', 99)
        if age < 75:
            if s_e < 7 or l_e < 10: f_maj = True
        else:
            if s_e < 5 or l_e < 7: f_maj = True
            
    ee = d.get('avg_E_e_prime', 0)
    if ee >= 15:
        f_maj = True
    elif 9 <= ee <= 14:
        f_min = True
        
    if d.get('tr_velocity', 0) > 2.8: f_maj = True
    if pd.notna(d.get('gls')) and abs(d.get('gls')) < 16: f_min = True
    s_func = 2 if f_maj else (1 if f_min else 0)

    # --- Morphological domain ---
    m_maj, m_min = False, False
    lavi = d.get('lavi', np.nan)
    if pd.notna(lavi):
        if is_af:
            if lavi > 40: m_maj = True
            elif 34 <= lavi <= 40: m_min = True
        else:
            if lavi > 34: m_maj = True
            elif 29 <= lavi <= 34: m_min = True
            
    lvmi, rwt = d.get('lvmi', np.nan), d.get('rwt', np.nan)
    if pd.notna(lvmi) and pd.notna(rwt):
        limit = 149 if gender == 'M' else 122
        if lvmi >= limit and rwt > 0.42:
            m_maj = True
        elif lvmi >= (115 if gender == 'M' else 95) or rwt > 0.42:
            m_min = True
            
    if d.get('wall_thickness', 0) >= 12: m_min = True
    s_morph = 2 if m_maj else (1 if m_min else 0)

    # --- Biomarkers ---
    b_maj, b_min = False, False
    nt, bnp = d.get('nt_pro_bnp', 0), d.get('bnp', 0)
    if is_af:
        if nt > 660 or bnp > 240: b_maj = True
        elif 365 <= nt <= 660 or 105 <= bnp <= 240: b_min = True
    else:
        if nt > 220 or bnp > 80: b_maj = True
        elif 125 <= nt <= 220 or 35 <= bnp <= 80: b_min = True
        
    s_bio = 2 if b_maj else (1 if b_min else 0)

    return s_func + s_morph + s_bio

# ==========================================
# 3. Batch-processing routine
# ==========================================
def process_data(base_path="base.csv", item_path="item_result.csv", output_path="hfa_peff_result.csv"):
    print("--- Starting HFA-PEFF batch assessment ---")
    
    def read_csv_safe(path):
        for enc in ['gbk', 'utf-8-sig', 'utf-8']:
            try:
                return pd.read_csv(path, encoding=enc, low_memory=False)
            except:
                continue
        return pd.DataFrame()

    base_df = read_csv_safe(base_path)
    item_df = read_csv_safe(item_path)
    
    if base_df.empty or item_df.empty:
        print("Error: cannot read the file, or it is empty.")
        return

    # Field conversion and aggregation
    def map_category(x):
        for k, v in FIELD_MAPPING.items():
            if any(p in str(x) for p in v):
                return k
        return None

    item_df['cat'] = item_df['res_itm_name'].apply(map_category)
    records = []
    
    for (app_no, pat_id), group in item_df.dropna(subset=['cat']).groupby(['app_no', 'pat_id']):
        d = {'app_no': str(app_no), 'pat_id': str(pat_id)}
        for cat in FIELD_MAPPING.keys():
            vals = group[group['cat'] == cat]['res_value'].apply(clean_numeric).dropna()
            d[cat] = vals.mean() if not vals.empty else np.nan
        
        # Detect rhythm text
        r_text = "".join(group[group['cat'] == 'rhythm_status']['res_value'].astype(str))
        d['r_status'] = parse_rhythm(r_text)
        
        # --- [Change 2] If the source has no BMI, compute it from captured height and weight ---
        h = d.get('height', np.nan)
        w = d.get('weight', np.nan)
        if pd.isna(d.get('bmi')) and pd.notna(h) and pd.notna(w) and h > 0:
            # Support cm and m conversion
            h_m = h / 100.0 if h > 3 else h 
            d['bmi'] = w / (h_m ** 2)

        records.append(d)

    wide_items = pd.DataFrame(records)
    if wide_items.empty:
        print("No valid indicator data extracted.")
        return
        
    base_df['pat_id'] = base_df['pat_id'].astype(str).str.strip()
    final_df = pd.merge(wide_items, base_df, on=['pat_id'], how='inner')

    results = []
    for _, r in final_df.iterrows():
        # Extract basic information
        age = r.get('app_pat_age', np.nan)
        gender_str = str(r.get('pat_sex', 'unknown'))
        gender_code = 'M' if 'male' in gender_str or gender_str == '1' else 'F'
        rhythm = r.get('r_status', 'SR')
        
        # Extract all raw indicators used in the computation
        septal_e = r.get('septal_e_prime', np.nan)
        lateral_e = r.get('lateral_e_prime', np.nan)
        avg_ee = r.get('avg_E_e_prime', np.nan)
        trv = r.get('tr_velocity', np.nan)
        gls = r.get('gls', np.nan)
        lavi = r.get('lavi', np.nan)
        lvmi = r.get('lvmi', np.nan)
        rwt = r.get('rwt', np.nan)
        wall_th = r.get('wall_thickness', np.nan)
        nt = r.get('nt_pro_bnp', np.nan)
        bnp = r.get('bnp', np.nan)

        score = calculate_hfa_peff_score(age, gender_code, rhythm, r)

        # Risk-to-code matching
        if score >= 5:
            res, code = "Definite", "I50.2.3"
        elif 2 <= score <= 4:
            res, code = "Intermediate", "I50.2.2"
        else:
            res, code = "Unlikely", "I50.2.1"

        results.append({
            'pat_id': r.get('pat_id'),
            'app_no': r.get('app_no'),
            'pat_name': r.get('pat_name', 'unknown'),
            'sex': r.get('pat_sex', 'unknown'),
            'age': int(age) if pd.notna(age) else '-',
            
            # --- [Change 3] Output BMI into the CSV results ---
            'BMI': round(r.get('bmi'), 2) if pd.notna(r.get('bmi')) else '-',
            
            # --- All raw indicators used in the computation, fully expanded ---
            'rhythm_status': 'atrial_fibrillation (AF)' if rhythm == 'AF' else 'sinus_rhythm (SR)',
            'septal_e': round(septal_e, 2) if pd.notna(septal_e) else '-',
            'lv_free_wall_e': round(lateral_e, 2) if pd.notna(lateral_e) else '-',
            'mean_E_e': round(avg_ee, 2) if pd.notna(avg_ee) else '-',
            'tricuspid_regurgitation_velocity (TRV)': round(trv, 2) if pd.notna(trv) else '-',
            'left_ventricular_global_longitudinal_strain (GLS)': round(gls, 2) if pd.notna(gls) else '-',
            'left_atrial_volume_index (LAVI)': round(lavi, 2) if pd.notna(lavi) else '-',
            'left_ventricular_mass_index (LVMI)': round(lvmi, 2) if pd.notna(lvmi) else '-',
            'relative_wall_thickness (RWT)': round(rwt, 2) if pd.notna(rwt) else '-',
            'wall_thickness': round(wall_th, 2) if pd.notna(wall_th) else '-',
            'NT-proBNP': round(nt, 2) if pd.notna(nt) else '-',
            'BNP': round(bnp, 2) if pd.notna(bnp) else '-',
            # -----------------------------------
            
            'model_name': 'HFA-PEFF HFpEF Screening',
            'predicted_score': score,
            'prediction_result': res,
            'code': code,
            'gauge_reference': '0-1|2-4|5-6'
        })

    if results:
        res_df = pd.DataFrame(results)
        
        # --- [Change 3] Ensure BMI is placed early when reordering columns ---
        main_cols = ['pat_id', 'app_no', 'pat_name', 'sex', 'age', 'BMI', 'rhythm_status',
                     'septal_e', 'lv_free_wall_e', 'mean_E_e', 'tricuspid_regurgitation_velocity (TRV)', 'left_ventricular_global_longitudinal_strain (GLS)',
                     'left_atrial_volume_index (LAVI)', 'left_ventricular_mass_index (LVMI)', 'relative_wall_thickness (RWT)', 'wall_thickness',
                     'NT-proBNP', 'BNP',
                     'model_name', 'predicted_score', 'prediction_result', 'code', 'gauge_reference']
        
        other_cols = [c for c in res_df.columns if c not in main_cols]
        res_df = res_df[main_cols + other_cols]
        
        res_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"--- Assessment complete: {len(res_df)} subjects ---")
        print(f"results exported to {output_path}")
    else:
        print("--- Assessment complete, no results ---")

if __name__ == "__main__":
    process_data()
