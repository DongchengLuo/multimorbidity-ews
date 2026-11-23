import pandas as pd
import numpy as np
import re

# ==========================================
# 1. Field mapping (based on the precise field-matching reference table)
# ==========================================
# Key: named after the "standard field" column in the Excel sheet
# Value: from the "matched raw field" column of the Excel sheet plus added generic synonyms
FIELD_MAPPING = {
    # --- Basic indicators ---
    # age is age and participates in the core computation
    'age': ['app_pat_age', 'age', 'patient_age', 'nianling'],

    # weight is body weight and participates in the core computation
    'weight': ['weight', 'body_weight', 'wgt', 'weight (kg)', 'tizhong'],

    # --- Screening indicators ---
    # gender is sex, used to filter for women
    'gender': ['pat_sex', 'sex', 'gender', 'xingbie'],

    # menopause_status is menopausal status, used to filter the included population
    'menopause_status': ['menopausal', 'menopause_flag', 'menstrual_status', 'menopause_status', 'menopause', 'is_menopause']
}


# ==========================================
# 2. Basic cleaning and matching logic
# ==========================================
def normalize_str(s):
    """String normalization: strip symbols, lowercase"""
    if pd.isna(s): return ""
    return re.sub(r'[\(\)\（\）\[\]\【\】\{\}\s\-\_]', '', str(s)).lower()


def get_category(name):
    """Match the standard label from the field name (key)"""
    norm_name = normalize_str(name)
    if not norm_name: return None

    # "(body mass index) BMI" contains "weight" but is a derived BMI, not raw weight.
    # Exclude before fuzzy matching so selecting the BMI model does not pollute OSTA's weight value.
    if "bmi" in norm_name or "bmi" in norm_name:
        return None

    # Prefer exact matching
    for key, patterns in FIELD_MAPPING.items():
        for p in patterns:
            if normalize_str(p) == norm_name: return key

    # Then fuzzy matching
    for key, patterns in FIELD_MAPPING.items():
        for p in patterns:
            if normalize_str(p) in norm_name: return key
    return None


def clean_numeric(val):
    """Extract numeric value"""
    if pd.isna(val) or str(val).strip() == "": return np.nan
    try:
        match = re.search(r"[-+]?\d*\.\d+|\d+", str(val))
        return float(match.group()) if match else np.nan
    except:
        return np.nan


def clean_gender(val):
    """Clean sex"""
    if pd.isna(val): return ""
    return str(val).strip().lower()


def parse_menopause(val):
    """
    Parse menopausal status
    Returns: 1 (postmenopausal), 0 (premenopausal), None (unknown)
    """
    if pd.isna(val): return None
    s = str(val).strip().lower()

    yes_keywords = ['yes', 'y', '1', 'true', 'postmenopausal', 'menopause']
    no_keywords = ['no', 'n', '0', 'false', 'premenopausal', 'normal', 'regular']

    for w in yes_keywords:
        if w in s: return 1
    for w in no_keywords:
        if w in s: return 0
    return None


# ==========================================
# 3. OSTA model core scoring logic
# ==========================================
# Constant definitions
FEMALE_LABELS = ['female', 'f', 'woman', '2']
MIN_VALID_AGE = 45  # without a menopause field, age >= 45 is treated as postmenopausal (study sample range)


def check_inclusion_criteria(age, gender_str, menopause_val):
    """
    Check inclusion criteria
    1. Must be female
    2. Must be postmenopausal or age >= 45 (as a proxy)
    """
    # 1. Sex filtering
    is_female = False
    if gender_str:
        for label in FEMALE_LABELS:
            if label in gender_str:
                is_female = True;
                break
    if not is_female: return False, "excluded: non-female"

    # 2. Menopause / age filtering
    # Check the menopause field first
    if menopause_val == 1:
        return True, "eligible: postmenopausal"
    if menopause_val == 0:
        return False, "excluded: premenopausal"

    # When the field is missing, infer from age
    if not pd.isna(age):
        if age >= MIN_VALID_AGE:
            return True, "eligible: inferred from age"
        else:
            return False, "excluded: too young with no menopause record"

    return False, "excluded: insufficient data"


def calculate_osta(age, weight):
    """
    Compute the OSTA index
    Formula: (weight[kg] - age[years]) * 0.2 -> truncate to integer
    Returns: (score, risk_level, code, suggestion, details)
    """
    if pd.isna(age) or pd.isna(weight) or weight <= 0:
        return None, None, None, None, "missing age or weight"

    # Core computation
    raw_val = (weight - age) * 0.2
    score = int(raw_val)  # truncate to integer

    details = f"Raw={raw_val:.2f}"

    # Risk classification
    # > -1       : low risk
    # -4 to -1   : medium risk
    # < -4       : high risk
    if score > -1:
        risk = "low risk"
        code = "M80.2.1"
        advice = "no BMD test needed for now"
    elif score >= -4:
        risk = "medium risk"
        code = "M80.2.2"
        advice = "BMD measurement advised"
    else:
        risk = "high risk"
        code = "M80.2.3"
        advice = "examination and treatment strongly advised"

    return score, risk, code, advice, details


# ==========================================
# 4. Main data-processing routine
# ==========================================
def process_data(base_path, item_path, output_path):
    print("--- Starting OSTA osteoporosis screening model ---")

    def read_csv_safe(path):
        for enc in ['gbk', 'utf-8-sig', 'utf-8']:
            try:
                return pd.read_csv(path, encoding=enc, low_memory=False)
            except:
                continue
        return pd.DataFrame()

    base_df = read_csv_safe(base_path)
    item_df = read_csv_safe(item_path)

    if base_df.empty and item_df.empty:
        print(f"data file unreadable or empty")
        return

    # --- A. Field standardization and preprocessing ---
    # Rename to app_no to match the output requirement
    for df in [base_df, item_df]:
        if not df.empty:
            cols = [c.lower() for c in df.columns]
            if 'app_no' in cols and 'app_no' not in cols:
                real_col = df.columns[cols.index('app_no')]
                df.rename(columns={real_col: 'app_no'}, inplace=True)

    # Ensure the join key exists
    join_keys = ['app_no', 'pat_id']
    valid_keys = [k for k in join_keys if k in base_df.columns]

    if not valid_keys:
        print("error: Base table missing app_no or pat_id join key")
        return

    for k in valid_keys:
        base_df[k] = base_df[k].astype(str).str.strip()

    # --- B. Pivot Item data to wide table ---
    # Consolidate item-table data into a dict keyed by (app_no, pat_id)
    item_dict_map = {}
    if not item_df.empty:
        # Handle Item-table keys the same way
        item_valid_keys = [k for k in join_keys if k in item_df.columns]
        if item_valid_keys:
            for k in item_valid_keys:
                item_df[k] = item_df[k].astype(str).str.strip()

            item_df['cat'] = item_df['res_itm_name'].apply(get_category)
            filtered = item_df.dropna(subset=['cat', 'res_value'])

            # Build a dict grouped by key
            for ids, group in filtered.groupby(item_valid_keys):
                # Build the key: a single key is the value itself, multiple keys form a tuple
                if len(item_valid_keys) == 1:
                    key = ids
                else:
                    key = tuple(ids)

                data_pool = {k: [] for k in FIELD_MAPPING.keys()}
                for _, item in group.iterrows():
                    data_pool[item['cat']].append(item['res_value'])
                item_dict_map[key] = data_pool

    # --- C. Iterate over the Base table for computation ---
    results = []
    for _, row in base_df.iterrows():
        # Build the query key
        if len(valid_keys) == 1:
            key = row[valid_keys[0]]
        else:
            key = tuple(row[k] for k in valid_keys)

        # Fetch this person's exam indicators (empty list when absent from the Item table)
        data_pool = item_dict_map.get(key, {k: [] for k in FIELD_MAPPING.keys()})

        # --- 1. Extract and merge variables (Base first, Item supplements) ---

        # Age
        age = clean_numeric(row.get('app_pat_age'))
        if pd.isna(age):  # try other Base fields
            for col in base_df.columns:
                if get_category(col) == 'age':
                    v = clean_numeric(row[col])
                    if not pd.isna(v): age = v; break
        if pd.isna(age) and data_pool['age']:  # try the Item table
            nums = [clean_numeric(x) for x in data_pool['age']]
            if nums: age = np.nanmean(nums)

        # Body weight
        weight = np.nan
        # Look up Base first
        for col in base_df.columns:
            if get_category(col) == 'weight':
                v = clean_numeric(row[col])
                if not pd.isna(v): weight = v; break
        # Then look up Item
        if pd.isna(weight) and data_pool['weight']:
            nums = [clean_numeric(x) for x in data_pool['weight']]
            if nums: weight = np.nanmean(nums)

        # Sex
        gender = clean_gender(row.get('pat_sex', ''))
        if not gender:  # try other Base fields
            for col in base_df.columns:
                if get_category(col) == 'gender':
                    v = clean_gender(row[col])
                    if v: gender = v; break
        if not gender and data_pool['gender']:  # try the Item table
            gender = clean_gender(data_pool['gender'][0])

        # Menopausal status
        menopause_val = None
        # Look up Base first
        for col in base_df.columns:
            if get_category(col) == 'menopause_status':
                menopause_val = parse_menopause(row[col]);
                break
        # Then look up Item
        if menopause_val is None and data_pool['menopause_status']:
            menopause_val = parse_menopause(data_pool['menopause_status'][0])

        # --- 2. Logical evaluation ---
        # Inclusion-criteria check
        is_valid, reason = check_inclusion_criteria(age, gender, menopause_val)

        if is_valid:
            # Compute OSTA
            score, risk_level, code, suggestion, details = calculate_osta(age, weight)

            # Output only when computation succeeds
            if score is not None:
                # Menopause description text
                m_desc = ""
                if menopause_val == 1:
                    m_desc = "postmenopausal"
                elif menopause_val == 0:
                    m_desc = "premenopausal"
                elif not pd.isna(age) and age >= MIN_VALID_AGE:
                    m_desc = "menopause_inferred_from_age"

                entry = {
                    'pat_id': row.get('pat_id', ''),
                    'app_no': row.get('app_no', ''),
                    'pat_name': row.get('pat_name', row.get('name', 'unknown')),
                    'age': int(age),
                    'weight': round(weight, 1),
                    'predicted_score': score,
                    'risk_level': risk_level,
                    'code': code,
                    'gauge_reference': ">-1|-4~-1|<-4",  # low|medium|high
                    'suggestion': suggestion,
                    'menopause_desc': m_desc
                }
                results.append(entry)

    # --- D. Export results ---
    if results:
        res_df = pd.DataFrame(results)

        # Order columns in the usual sequence
        cols_order = ['pat_id', 'app_no', 'pat_name', 'age', 'weight', 'menopause_desc',
                      'predicted_score', 'risk_level', 'code', 'gauge_reference', 'suggestion']
        # Keep only existing columns
        final_cols = [c for c in cols_order if c in res_df.columns]
        res_df = res_df[final_cols]

        res_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"--- Assessment complete: {len(res_df)} valid results, exported to {output_path} ---")
    else:
        print("--- no valid results (possibly no eligible data) ---")


if __name__ == "__main__":
    process_data(
        "base.csv",
        "item_result.csv",
        "osta_final_result_M80_.csv"
    )
