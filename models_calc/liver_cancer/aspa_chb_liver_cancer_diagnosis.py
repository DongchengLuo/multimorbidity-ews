import pandas as pd
import numpy as np
import re

# ==========================================
# 1. Field-mapping configuration
# ==========================================
FIELD_MAPPING = {
    'age': ['app_pat_age', 'age'],
    'gender': ['pat_sex', 'sex', 'gender'],
    'AFP': ['afp', 'AFP', 'AFP_quantitative'],
    'pivka': ['PIVKA-II', 'pivka']
}

MODEL_NAME = "Liver Cancer ASAP Model"


# ==========================================
# 2. Basic utility functions
# ==========================================
def normalize_str(s):
    if pd.isna(s): return ""
    return re.sub(r'[\(\)\（\）\[\]\【\】\{\}\s]', '', str(s)).lower()


def get_category(item_name):
    norm_name = normalize_str(item_name)
    for key, patterns in FIELD_MAPPING.items():
        for p in patterns:
            # Strict exact matching
            if normalize_str(p) == norm_name:
                return key
    return None


def clean_numeric(val):
    if pd.isna(val) or str(val).strip() == "": return np.nan
    try:
        match = re.search(r"[-+]?\d*\.\d+|\d+", str(val))
        return float(match.group()) if match else np.nan
    except:
        return np.nan


def parse_sex(val):
    """Project DB pat_sex convention: 1=male, 2=female; inside models female=1, male=0."""
    if pd.isna(val):
        return np.nan
    s = str(val).strip().lower()
    if not s:
        return np.nan
    if s in ['female', 'woman', 'f', '2', '2.0']:
        return 1
    if s in ['male', 'man', 'm', '1', '1.0']:
        return 0
    # 0 has conflicting meanings across systems; refuse to guess.
    return np.nan


def convert_unit(val, unit, category):
    """Unit conversion: normalize values to the units the model formula requires"""
    if pd.isna(val): return np.nan
    u = normalize_str(str(unit))

    if category == 'AFP':
        # Target unit: ng/mL (medically equivalent to ug/L, ug/L)
        if u == 'mg/l':
            return val * 1000
        elif u == 'mg/ml':
            return val * 1000000
        elif u == 'ng/l':
            return val / 1000
        elif u in ['iu/ml', 'miu/ml']:
            return val * 1.21  # common clinical conversion: 1 IU/mL ~ 1.21 ng/mL

    elif category == 'pivka':
        # Target unit: mAU/mL
        if u == 'au/ml':
            return val * 1000
        elif u == 'mau/l':
            return val / 1000

    # Return the original value when the unit is already the target unit (e.g. ng/ml, mau/ml) or unrecognized
    return val


# ==========================================
# 3. ASAP model core algorithm
# ==========================================
def calculate_asap_score(age, sex, afp, pivka):
    # Strict gate: return immediately when any item is missing
    if any(pd.isna(x) for x in [age, sex, afp, pivka]):
        return "-", "-", "-", "-", "core indicators (age/sex/AFP/PIVKA-II) missing; cannot assess"

    try:
        afp_val = 0.001 if float(afp) <= 0 else float(afp)
        pivka_val = 0.001 if float(pivka) <= 0 else float(pivka)

        intercept = -7.57711770
        coef_age = 0.04666357
        coef_sex = -0.57611693
        coef_ln_afp = 0.42243533
        coef_ln_pivka = 1.10518910

        logit = (intercept +
                 (coef_age * age) +
                 (coef_sex * sex) +
                 (coef_ln_afp * np.log(afp_val)) +
                 (coef_ln_pivka * np.log(pivka_val)))

        prob = 1 / (1 + np.exp(-logit))

        gauge = "0-30%|30%-70%|70%-100%"

        if prob < 0.30:
            level, code = "low risk", "C22.2.1"
        elif 0.30 <= prob < 0.70:
            level, code = "medium risk", "C22.2.2"
        else:
            level, code = "high risk", "C22.2.3"

        return prob, level, code, gauge, "data complete, computable"

    except:
        return "-", "-", "-", "-", "numeric anomaly during computation"


# ==========================================
# 4. Main data-processing routine
# ==========================================
def process_data(base_path, item_path, output_path):
    print(f"--- Starting {MODEL_NAME} screening model (strict unit conversion) ---")

    def read_csv_safe(path):
        for enc in ['utf-8-sig', 'gbk', 'utf-8']:
            try:
                return pd.read_csv(path, encoding=enc, low_memory=False)
            except:
                continue
        return pd.DataFrame()

    base_df = read_csv_safe(base_path)
    item_df = read_csv_safe(item_path)

    if base_df.empty:
        print("error: input file empty or unreadable.")
        return

    # Preserve original ID format (trim spaces, do not strip zeros casually)
    base_df['app_no'] = base_df['app_no'].astype(str).str.strip()
    base_df['pat_id'] = base_df['pat_id'].astype(str).str.strip()

    # 1. Process the Item table (read res_value and res_unit)
    records = []
    if not item_df.empty:
        item_df['app_no'] = item_df['app_no'].astype(str).str.strip()
        item_df['pat_id'] = item_df['pat_id'].astype(str).str.strip()

        # Fault tolerance: guard against a missing res_unit column
        if 'res_unit' not in item_df.columns:
            item_df['res_unit'] = ""

        item_df['cat'] = item_df['res_itm_name'].astype(str).apply(get_category)
        filtered_items = item_df.dropna(subset=['cat']).copy()

        def get_first_valid(val_list, category=None):
            for item_dict in val_list:
                v = item_dict['val']
                u = item_dict['unit']
                if pd.isna(v) or str(v).strip() == "": continue

                num = clean_numeric(v)
                if pd.notna(num):
                    if category:
                        return convert_unit(num, u, category)
                    return num
            return np.nan

        for (app_no, pat_id), group in filtered_items.groupby(['app_no', 'pat_id']):
            row = {'app_no': app_no, 'pat_id': pat_id}
            data_pool = {k: [] for k in FIELD_MAPPING.keys()}

            for _, item in group.iterrows():
                data_pool[item['cat']].append({
                    'val': item['res_value'],
                    'unit': item['res_unit']
                })

            # Carry the category for unit conversion
            row['AFP'] = get_first_valid(data_pool.get('AFP', []), category='AFP')
            row['pivka'] = get_first_valid(data_pool.get('pivka', []), category='pivka')

            records.append(row)

    wide_items = pd.DataFrame(records) if records else pd.DataFrame(columns=['app_no', 'pat_id'])

    # 2. Merge: left join, keep all
    merged_df = pd.merge(base_df, wide_items, on=['app_no', 'pat_id'], how='left')

    def fmt_in(v, is_num=False, is_sex=False):
        if pd.isna(v) or v is None or str(v).strip() == "": return "missing"
        if is_sex:
            if v == 1: return "female"
            if v == 0: return "male"
            return str(v)
        return f"{v:.2f}" if is_num else str(v)

    # 3. Score computation
    results = []
    for _, row in merged_df.iterrows():
        age = clean_numeric(row.get('app_pat_age'))

        raw_sex = row.get('pat_sex')
        sex_code = parse_sex(raw_sex)

        afp = row.get('AFP', np.nan)
        pivka = row.get('pivka', np.nan)

        # Model computation
        prob, level, risk_code, gauge, remark = calculate_asap_score(age, sex_code, afp, pivka)

        # Dict insertion order defines the final column order
        res_item = {
            'pat_id': row.get('pat_id'),
            'app_no': row.get('app_no'),
            'pat_name': row.get('pat_name', 'unknown'),
            'model_name': MODEL_NAME,
            'code': risk_code,
            'gauge_reference': gauge,
            'risk_probability': f"{prob * 100:.2f}%" if prob != "-" else "-",
            'risk_level': level,
            'prediction_horizon': 'current' if prob != "-" else "-",
            'calculation_status': 'computed' if prob != "-" else 'input_incomplete',
            'not_computed_reason': '' if prob != "-" else remark,
            'assessment_note': remark,

            # --- Raw indicator section ---
            'sex': fmt_in(sex_code, is_sex=True) if pd.notna(sex_code) else (raw_sex if pd.notna(raw_sex) else "missing"),
            'age': int(age) if pd.notna(age) else "missing",
            'AFP(ng/mL)': fmt_in(afp, True),
            'PIVKA-II(mAU/mL)': fmt_in(pivka, True)
        }

        results.append(res_item)

    # 4. Export results
    if results:
        res_df = pd.DataFrame(results)
        res_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"--- Assessment complete: {len(res_df)} records processed (with automatic unit conversion), exported to {output_path} ---")
    else:
        print("no results generated.")


if __name__ == "__main__":
    process_data("base.csv", "item_result.csv", "asap_screening_result_perfect.csv")
