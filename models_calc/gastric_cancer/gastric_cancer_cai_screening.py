import pandas as pd
import numpy as np
import re
import json

# ==========================================
# 1. Field mapping (see the precise field-matching reference table)
# ==========================================
FIELD_MAPPING = {
    # --- Basic information ---
    'age': ['app_pat_age', 'age'],
    'gender': ['pat_sex', 'sex'],

    # --- Serological indicators ---
    # Standard field: pgr (pepsinogen ratio)
    'pgr': ['pepsinogen_I_II', 'pepsinogen_ratio', 'PGI/PGII', 'PG I/II', 'PGR'],

    # Standard field: g17 (gastrin-17)
    'g17': ['gastrin_17', 'G-17', 'Gastrin-17'],

    # Standard field: hp_positive (H. pylori)
    'hp_positive': ['h_pylori_serology', 'h_pylori_serology_monitoring', 'h_pylori_antibody', 'Hp IgG', 'Hp_antibody',
                    'helicobacter_pylori'],

    # --- Auxiliary indicators (added for ratio computation) ---
    'pg_i': ['pepsinogen_I', 'PG I', 'PGI', 'pepsinogen_1'],
    'pg_ii': ['pepsinogen_II', 'PG II', 'PGII', 'pepsinogen_2'],

    # --- Diet and lifestyle habits (added manually) ---
    'pickled_food': ['salt_preserved_food', 'frequent_pickled_vegetables', 'pickled_vegetables', 'pickled_food', 'preserved_food'],
    'fried_food': ['fried_food', 'frequent_fried_food']
}


# ==========================================
# 2. Basic cleaning and matching logic
# ==========================================
def normalize_str(s):
    if pd.isna(s): return ""
    return re.sub(r'[\(\)\（\）\[\]\【\】\{\}\s]', '', str(s)).lower()


def get_category(name):
    """Exact-match the standard label from the field name"""
    norm_name = normalize_str(name)
    for key, patterns in FIELD_MAPPING.items():
        for p in patterns:
            if normalize_str(p) == norm_name: return key  # [change] exact match: use == instead of in
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
    """Parse sex: Male/Female"""
    s = str(val).strip()
    if 'male' in s or s.lower() == 'male' or s == '1':
        return 'Male'
    if 'female' in s or 'female' in s or s == '0':
        return 'Female'
    return None


def parse_hp_status(val):
    """Parse H. pylori infection status"""
    if pd.isna(val): return None
    s = str(val).lower()
    if 'positive' in s or 'pos' in s or '+' in s: return True
    if 'negative' in s or 'neg' in s or '-' in s: return False
    num = clean_numeric(val)
    if not pd.isna(num): return num >= 34
    return None


def parse_diet_habit(val):
    """Parse dietary habits"""
    if pd.isna(val): return None
    s = str(val).lower()
    if any(k in s for k in ['often', 'regular', 'yes', 'daily', 'always', '>=3']): return True
    if any(k in s for k in ['occasionally', 'occasional', 'no', 'never', 'few', '<3']): return False
    return None


# ==========================================
# 3. Core scoring logic (Cai Q et al. Gut 2019)
# ==========================================
def calculate_gastric_cancer_risk(age, sex, pgr, g17, hp_pos, pickled_reg, fried_reg):
    """
    Returns: (score, risk_level, code, debug_info)
    """
    # Strict mode: do not compute when key indicators are missing
    if any(pd.isna(x) for x in [age, pgr, g17]): return None, None, None, None
    if sex is None or hp_pos is None or pickled_reg is None or fried_reg is None: return None, None, None, None

    score = 0
    debug_info = {}

    # 1. Age (40-80 target, <40 assumed 0)
    age_s = 0
    if 50 <= age <= 59:
        age_s = 4
    elif 60 <= age <= 69:
        age_s = 6
    elif age > 69:
        age_s = 9
    score += age_s
    debug_info['age_score'] = age_s

    # 2. Sex (Male=4)
    sex_s = 4 if sex == 'Male' else 0
    score += sex_s
    debug_info['sex_score'] = sex_s

    # 3. PG I/II Ratio (<3.89 = 3)
    pg_s = 3 if pgr < 3.89 else 0
    score += pg_s
    debug_info['pgr_score'] = pg_s

    # 4. G-17 (1.50-5.70=3, >5.70=4)
    g17_s = 0
    if 1.50 <= g17 <= 5.70:
        g17_s = 3
    elif g17 > 5.70:
        g17_s = 4
    score += g17_s
    debug_info['g17_score'] = g17_s

    # 5. Hp Infection (Pos=1)
    hp_s = 1 if hp_pos else 0
    score += hp_s
    debug_info['hp_score'] = hp_s

    # 6. Pickled Food (Regular=2)
    pickle_s = 2 if pickled_reg else 0
    score += pickle_s
    debug_info['pickle_score'] = pickle_s

    # 7. Fried Food (Regular=2)
    fried_s = 2 if fried_reg else 0
    score += fried_s
    debug_info['fried_score'] = fried_s

    # Risk tiering and codes
    if score <= 11:
        risk_level = "low risk"
        code = "C16.2.1"
    elif 12 <= score <= 16:
        risk_level = "medium risk"
        code = "C16.2.2"
    else:
        risk_level = "high risk"
        code = "C16.2.3"

    return score, risk_level, code, debug_info


# ==========================================
# 4. Main data-processing routine
# ==========================================
def process_data(base_path, item_path, output_path):
    print("--- Starting China high-risk gastric cancer risk model (Cai Q et al. 2019) ---")

    def read_csv_safe(path):
        for enc in ['gbk', 'utf-8-sig', 'utf-8']:
            try:
                return pd.read_csv(path, encoding=enc, low_memory=False)
            except:
                continue
        return pd.DataFrame()

    base_df = read_csv_safe(base_path)
    item_df = read_csv_safe(item_path)

    # --- [Change] Compatibility: prevent non-string column names from breaking .str.strip() ---
    base_df.columns = [str(c).strip() for c in base_df.columns]
    item_df.columns = [str(c).strip() for c in item_df.columns]
    # -----------------------------------------------------------------------

    if base_df.empty or item_df.empty:
        print("Data file could not be read or is empty")
        return

    # 1. Pivot to wide table
    item_df['cat'] = item_df['res_itm_name'].apply(get_category)
    filtered = item_df.dropna(subset=['cat', 'res_value']).copy()

    records = []
    # Assume the file join keys are app_no and pat_id
    group_cols = ['app_no', 'pat_id']
    # Check whether the column exists
    if not all(col in filtered.columns for col in group_cols):
        print(f"error: detail table missing columns {group_cols}")
        return

    for (app_no, pat_id), group in filtered.groupby(group_cols):
        # Coerce to string to avoid matching errors
        app_no_str = str(app_no).strip()
        pat_id_str = str(pat_id).strip()

        row = {'app_no': app_no_str, 'pat_id': pat_id_str}  # rename app_no -> app_no on output
        data_pool = {k: [] for k in FIELD_MAPPING.keys()}

        for _, item in group.iterrows():
            data_pool[item['cat']].append(str(item['res_value']))

        # Extract indicators
        # PGR - [change] take the first valid value instead of averaging
        pgr_values = [clean_numeric(v) for v in data_pool.get('pgr', [])]
        pgr = next((v for v in pgr_values if not pd.isna(v)), np.nan)
        if pd.isna(pgr):
            p1_values = [clean_numeric(v) for v in data_pool.get('pg_i', [])]
            p2_values = [clean_numeric(v) for v in data_pool.get('pg_ii', [])]
            p1 = next((v for v in p1_values if not pd.isna(v)), np.nan)
            p2 = next((v for v in p2_values if not pd.isna(v)), np.nan)
            if not pd.isna(p1) and not pd.isna(p2) and p2 != 0:
                pgr = p1 / p2
        row['pgr'] = pgr

        # G17 - [change] take the first valid value instead of averaging
        g17_values = [clean_numeric(v) for v in data_pool.get('g17', [])]
        row['g17'] = next((v for v in g17_values if not pd.isna(v)), np.nan)

        hp_raw = next((v for v in data_pool.get('hp_positive', []) if v), None)
        row['hp_pos'] = parse_hp_status(hp_raw)
        row['hp_raw'] = hp_raw

        pickled_raw = next((v for v in data_pool.get('pickled_food', []) if v), None)
        row['pickled_reg'] = parse_diet_habit(pickled_raw)

        fried_raw = next((v for v in data_pool.get('fried_food', []) if v), None)
        row['fried_reg'] = parse_diet_habit(fried_raw)

        records.append(row)

    wide_items = pd.DataFrame(records)

    # 2. Merge the Base table
    # Ensure the Base table also has the join key
    if 'app_no' in base_df.columns and 'pat_id' in base_df.columns:
        base_df['app_no'] = base_df['app_no'].astype(str).str.strip()  # build app_no to retain after merging
        base_df['pat_id'] = base_df['pat_id'].astype(str).str.strip()

        # Merge on app_no and pat_id (wide_items already has both)
        if wide_items.empty:
            final = base_df
        else:
            final = pd.merge(base_df, wide_items, on=['app_no', 'pat_id'], how='left')
    else:
        print("error: Base table missing app_no or pat_id")
        return

    # 3. Score computation
    results = []
    model_name = "Gastric Cancer Risk Prediction (Cai Q et al. 2019)"
    gauge_ref = "0-11|12-16|17-25"  # dashboard reference

    for _, r in final.iterrows():
        # Fetch basic information
        age = clean_numeric(r.get('app_pat_age') if 'app_pat_age' in r else r.get('age'))
        gender_raw = r.get('pat_sex') if 'pat_sex' in r else r.get('gender')
        sex = parse_gender(gender_raw)

        # Fetch wide-table data
        pgr = r.get('pgr', np.nan)
        g17 = r.get('g17', np.nan)
        hp_pos = r.get('hp_pos', None)
        pickled_reg = r.get('pickled_reg', None)
        fried_reg = r.get('fried_reg', None)

        # Core computation
        score, risk_level, code, details = calculate_gastric_cancer_risk(
            age, sex, pgr, g17, hp_pos, pickled_reg, fried_reg
        )

        res_row = {
            'app_no': r.get('app_no'),
            'pat_id': r.get('pat_id'),
            'pat_name': r.get('pat_name', 'unknown'),  # rename name -> pat_name
            'model_name': model_name,
            'prediction_horizon': 'current',
            'sex': gender_raw,
            'age': age,
            'code': code,  # added code
            'gauge_reference': gauge_ref,  # added dashboard reference
            'PG_I_II_ratio': f"{pgr:.2f}" if not pd.isna(pgr) else None,
            'G17': f"{g17:.2f}" if not pd.isna(g17) else None,
            'Hp_infection_status': "positive" if hp_pos is True else ("negative" if hp_pos is False else r.get('hp_raw')),
            'frequent_salt_preserved_food': "yes" if pickled_reg is True else ("no" if pickled_reg is False else None),
            'frequent_fried_food': "yes" if fried_reg is True else ("no" if fried_reg is False else None),
            'predicted_score': score,
            'risk_level': risk_level,
        }

        if score is None:
            res_row['note'] = "data missing; cannot compute"
        else:
            res_row['note'] = "computed"
            if risk_level in ['medium risk', 'high risk']:
                res_row['recommendation'] = "recommend gastroscopy"
            else:
                res_row['recommendation'] = "regular follow-up"

        results.append(res_row)

    if results:
        res_df = pd.DataFrame(results)
        
        # [Change]: export directly; remove hard-coded column filtering and reordering
        res_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"--- Assessment complete: {len(res_df)} results generated, exported to {output_path} ---")
    else:
        print("--- Assessment complete, but no results ---")


if __name__ == "__main__":
    # Replace with the actual file path
    process_data(
        "base.csv",
        "item_result.csv",
        "gastric_Cai_screening_result.csv"
    )
