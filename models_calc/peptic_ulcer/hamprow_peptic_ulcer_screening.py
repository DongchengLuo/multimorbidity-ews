import pandas as pd
import numpy as np
import re

# ==========================================
# 1. Field mapping (HAMPROW peptic ulcer risk assessment)
# ==========================================
# Key is the "standard field" name; value is the "matched raw field" or a generic synonym
FIELD_MAPPING = {
    # --- Basic information ---
    'gender': ['pat_sex', 'sex'],  # required: male=1, female=0
    'age': ['app_pat_age', 'age'],  # display only

    # --- Laboratory indicators ---
    'ALB': ['albumin', 'albumin (ALB)', 'serum_albumin'],  # unit: g/L
    'pg_i': ['pepsinogen_I', 'PGI', 'PG I', 'PG1'],  # unit: ug/L
    'hp_positive': ['h_pylori_serology', 'h_pylori_serology_monitoring'],  # parse result: positive=1, negative=0
    # --- Supplementary indicators ---
    'WBC': ['WBC (LEU)', 'WBC'],  # unit: 10^9/L
    'RBC': ['rbc_count', 'RBC'],  # unit: 10^12/L
    'GLB': ['globulin', 'GLB', 'globulin (GLB)'],  # unit: g/L
}

# ==========================================
# 2. Configuration parameters
# ==========================================
# Risk threshold (probability > 30% is high risk, based on the ~146-point nomogram value)
RISK_THRESHOLD = 0.30
GAUGE_REF_STR = "0%-30%|30%-100%"  # low | high (binary)

# Risk code
CODE_LOW = "K27.1.1"
# CODE_MID = "K27.1.2" # binary classification does not use medium risk
CODE_HIGH = "K27.1.3"
CODE_UNABLE = "K27.1.9"  # cannot compute


# ==========================================
# 3. Basic cleaning and matching logic
# ==========================================
def normalize_str(s):
    if pd.isna(s): return ""
    return re.sub(r'[\(\)\（\）\[\]\【\】\{\}\s]', '', str(s)).lower()


def get_category(name):
    """Match the standard label from the field name"""
    norm_name = normalize_str(name)
    for key, patterns in FIELD_MAPPING.items():
        for p in patterns:
            if normalize_str(p) == norm_name: 
                return key  # strict exact match
    return None


def clean_numeric(val):
    """Extract numeric value"""
    if pd.isna(val) or str(val).strip() == "": return np.nan
    try:
        match = re.search(r"[-+]?\d*\.\d+|\d+", str(val))
        return float(match.group()) if match else np.nan
    except:
        return np.nan


def parse_gender(val):
    """Parse sex: male -> 1, female -> 0, missing -> None"""
    if pd.isna(val) or str(val).strip() == "":
        return None  
    s = str(val).strip()
    if 'male' in s or 'Male' in s or s == '1':
        return 1
    if 'female' in s or 'Female' in s or s == '2':
        return 0
    return None  


def parse_hp_status(val):
    """Parse HP antibody status: positive -> 1, negative -> 0, dirty data -> missing"""
    s = str(val).strip().upper()
    if any(x in s for x in ['positive', 'POS', 'POSITIVE', '+', '1']):
        return 1
    if any(x in s for x in ['negative', 'NEG', 'NEGATIVE', '-', '0']):
        return 0
    return np.nan # ambiguous dirty data is treated as missing


# ==========================================
# 4. HAMPROW core scoring logic
# ==========================================
def calculate_hamprow_risk(is_male, wbc, rbc, glb, alb, pgi, hp_positive):
    """
    Compute the HAMPROW risk probability
    Parameters must use the matching units: WBC (10^9/L), RBC (10^12/L), GLB (g/L), ALB (g/L), PGI (ug/L)
    Returns: (probability, debug_info, is_calculable)
    """
    # Strict missing-value check: any missing indicator blocks computation
    inputs = {'Male': is_male, 'WBC': wbc, 'RBC': rbc, 'GLB': glb, 'ALB': alb, 'PGI': pgi, 'HP': hp_positive}
    if any(pd.isna(v) for v in inputs.values()):
        return None, None, False  

    # Define coefficients
    intercept = 3.92103
    coef_male = 0.85315
    coef_wbc = 0.14551
    coef_rbc = -0.72291
    coef_glb = -0.07839
    coef_alb = -0.06550
    coef_pgi = 0.00788
    coef_hp = 0.91716

    # Compute log-odds
    log_odds = (
            intercept +
            (coef_male * is_male) +
            (coef_wbc * wbc) +
            (coef_rbc * rbc) +
            (coef_glb * glb) +
            (coef_alb * alb) +
            (coef_pgi * pgi) +
            (coef_hp * hp_positive)
    )

    # Compute probability P
    probability = 1 / (1 + np.exp(-log_odds))

    debug_info = {
        'Log_Odds': round(log_odds, 4),
        'Probability': round(probability, 4)
    }

    return probability, debug_info, True


# ==========================================
# 5. Main data-processing routine
# ==========================================
def process_data(base_path, item_path, output_path):
    print("--- Starting HAMPROW peptic ulcer risk model ---")

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
        print("Error: data file could not be read or is empty")
        return

    # 1. Wide-table processing (Item Result) - minimal read logic
    item_df['cat'] = item_df['res_itm_name'].apply(get_category)
    filtered = item_df.dropna(subset=['cat', 'res_value']).copy()

    records = []
    # Ensure matching fields are strings
    filtered['app_no'] = filtered['app_no'].astype(str).str.strip()
    filtered['pat_id'] = filtered['pat_id'].astype(str).str.strip()

    # Aggregate by exam number (app_no) and patient ID (pat_id)
    for (app_no, pat_id), group in filtered.groupby(['app_no', 'pat_id']):
        row = {'app_no': app_no, 'pat_id': pat_id}

        # Iterate this exam's detail items, strictly taking only the first record
        for _, item in group.iterrows():
            cat = item['cat']
            val = item['res_value']
            
            if cat not in row:
                if cat == 'hp_positive':
                    row[cat] = parse_hp_status(val)
                else:
                    row[cat] = clean_numeric(val)

        records.append(row)

    wide_items = pd.DataFrame(records)

    # 2. Merge basic information (Base table)
    base_df['app_no'] = base_df['app_no'].astype(str).str.strip()
    base_df['pat_id'] = base_df['pat_id'].astype(str).str.strip()

    if wide_items.empty:
        print("no valid exam indicator data extracted")
        return

    final = pd.merge(wide_items, base_df, on=['app_no', 'pat_id'], how='inner')

    # 3. Scoring and result generation
    results = []
    for _, r in final.iterrows():
        # Fetch sex (prefer the Base table)
        gender_raw = r.get('pat_sex')
        is_male = parse_gender(gender_raw)  

        # Fetch other indicators (get returns np.nan by default when missing)
        wbc = r.get('WBC', np.nan)
        rbc = r.get('RBC', np.nan)
        glb = r.get('GLB', np.nan)
        alb = r.get('ALB', np.nan)
        pgi = r.get('pg_i', np.nan)
        hp = r.get('hp_positive', np.nan)

        # Run computation
        prob, details, is_calculable = calculate_hamprow_risk(
            is_male if is_male is not None else np.nan, wbc, rbc, glb, alb, pgi, hp
        )

        # Binary risk tiering
        if is_calculable:
            if prob >= RISK_THRESHOLD:
                risk_label = "high risk"
                risk_code = CODE_HIGH
            else:
                risk_label = "low risk"
                risk_code = CODE_LOW
            prob_pct_str = f"{prob * 100:.2f}%"
        else:
            risk_label = "-"
            risk_code = "-"
            prob_pct_str = "-"

        # Build output rows
        results.append({
            'app_no': r.get('app_no'),  
            'pat_id': r.get('pat_id'),  
            'pat_name': r.get('pat_name', 'unknown'),
            'code': risk_code,
            'gauge_reference': GAUGE_REF_STR,
            'model_name': 'HAMPROW Peptic Ulcer Risk Assessment',
            'sex': 'male' if is_male == 1 else ('female' if is_male == 0 else 'missing'),  
            'age': r.get('app_pat_age', '-'),
            'WBC_count': f"{wbc:.2f}" if not pd.isna(wbc) else "missing",
            'rbc_count': f"{rbc:.2f}" if not pd.isna(rbc) else "missing",
            'globulin': f"{glb:.1f}" if not pd.isna(glb) else "missing",
            'ALB': f"{alb:.1f}" if not pd.isna(alb) else "missing",
            'PGI': f"{pgi:.1f}" if not pd.isna(pgi) else "missing",
            'HP': ('positive' if hp == 1 else 'negative') if not pd.isna(hp) else "missing",
            'risk_probability': prob_pct_str,
            'risk_level': risk_label,
            'prediction_horizon':'current'
        })

    if results:
        res_df = pd.DataFrame(results)
        # Output directly; reordering code removed
        res_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"--- Assessment complete: {len(res_df)} results generated, exported to {output_path} ---")
    else:
        print("--- Assessment complete, but no results ---")


if __name__ == "__main__":
    process_data(
        "base.csv",
        "item_result.csv",
        "hamprow_ulcer_risk_K27_result.csv"
    )
