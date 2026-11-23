import pandas as pd
import numpy as np
import re
import math

# ==========================================
# 1. Field mapping (see the precise field-matching reference table)
# ==========================================
FIELD_MAPPING = {
    'age': ['app_pat_age', 'age'],
    'gender': ['pat_sex', 'sex'],
    'has_diabetes': ['prior_diabetes', 'type2_diabetes', 'diabetes_history', 'past_history_diabetes'],
    'ast': ['AST'],
    'alt': ['ALT'],
    'plt': ['PLT', 'platelet_count', 'B-PLT'],
    'lsm': ['liver_stiffness', 'liver_stiffness_noninvasive', 'e_value', 'liver_elastography']
}


# ==========================================
# 2. Basic cleaning and matching logic
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
    """Generic numeric extraction: take the first number in the string"""
    if pd.isna(val) or str(val).strip() == "": return np.nan
    try:
        match = re.search(r"[-+]?\d*\.\d+|\d+", str(val))
        return float(match.group()) if match else np.nan
    except:
        return np.nan


def clean_lsm(val):
    """
    Dedicated cleaning of complex liver-stiffness (LSM) strings.
    Handle cases such as "CAP:171dB/m, liver stiffness:5.7kPa".
    """
    if pd.isna(val) or str(val).strip() == "":
        return np.nan

    s = str(val)

    try:
        # Priority 1: look for explicit keywords "liver stiffness", "E value" or "LSM" and take the following number
        # [^\d]* skips non-digit characters between the keyword and the number (e.g. "value:" or ":")
        match = re.search(r'(?:liver stiffness|E value|LSM)[^\d]*(\d+(?:\.\d+)?)', s, re.IGNORECASE)
        if match:
            return float(match.group(1))

        # Priority 2: if "liver stiffness" is not stated but a kPa unit appears, take the number before kPa
        match_kpa = re.search(r'(\d+(?:\.\d+)?)\s*kpa', s, re.IGNORECASE)
        if match_kpa:
            return float(match_kpa.group(1))

        # Priority 3 (fallback): if only a bare number is present (e.g. "5.7"), fall back to basic extraction
        return clean_numeric(s)
    except:
        return np.nan


def parse_gender_to_num(val):
    s = str(val).lower()
    if 'male' in s or 'm' in s or '1' == s:
        return 1
    return 0


def parse_yes_no(val):
    s = str(val).lower()
    if any(x in s for x in ['yes', 'present', 'y', '1']):
        return 1
    return 0


# ==========================================
# 3. Agile 3+ core scoring logic
# ==========================================
def calculate_agile3_plus(age, sex_num, diabetes_num, lsm, ast, alt, plt):
    """
    Agile 3+ (advanced fibrosis F>=3)
    """
    debug_info = {}
    if any(pd.isna(x) or x <= 0 for x in [lsm, ast, alt, plt, age]):
        return None, "-", "-", "-", {}

    try:
        aar = ast / alt
        logit = -3.92368 + \
                (2.29714 * math.log(lsm)) - \
                (0.00902 * plt) - \
                (0.98633 * (1.0 / aar)) + \
                (1.08636 * diabetes_num) - \
                (0.38581 * sex_num) + \
                (0.03018 * age)

        score = math.exp(logit) / (1 + math.exp(logit))
        debug_info['AAR'] = round(aar, 2)

        if score < 0.451:
            risk_level = "low risk"
            suggestion = "regular follow-up"
            code = "K74.2.1"
        elif score > 0.679:
            risk_level = "high risk"
            suggestion = "hepatology specialist evaluation advised"
            code = "K74.2.3"
        else:
            risk_level = "medium risk"
            suggestion = "assess with other clinical indications"
            code = "K74.2.2"

        return score, risk_level, code, suggestion, debug_info

    except Exception as e:
        return None, "-", "-", "-", {}


# ==========================================
# 4. Main data-processing routine
# ==========================================
def process_data(base_path, item_path, summary_path, output_path):
    print("--- Starting Agile 3+ (advanced fibrosis) model ---")

    def read_csv_safe(path):
        for enc in ['gbk', 'utf-8-sig', 'utf-8']:
            try:
                return pd.read_csv(path, encoding=enc, low_memory=False)
            except:
                continue
        return pd.DataFrame()

    base_df = read_csv_safe(base_path)
    item_df = read_csv_safe(item_path)
    summary_df = read_csv_safe(summary_path)

    if base_df.empty:
        print("Error: base data file could not be read or is empty")
        return

    # 1. Pivot to wide table
    records = []
    concat_list = []

    # Handle the old table: extract item_result data
    if not item_df.empty:
        temp_item = item_df[['app_no', 'pat_id', 'res_itm_name', 'res_value']].copy()
        temp_item.rename(columns={'res_itm_name': 'name', 'res_value': 'value'}, inplace=True)
        concat_list.append(temp_item)

    # Handle the new table: extract item_summary data (sum_summary column)
    if not summary_df.empty:
        temp_summary = summary_df[['app_no', 'pat_id', 'det_com_name', 'sum_summary']].copy()
        temp_summary.rename(columns={'det_com_name': 'name', 'sum_summary': 'value'}, inplace=True)
        concat_list.append(temp_summary)

    # Fuse old and new table data for unified recognition
    if concat_list:
        combined_df = pd.concat(concat_list, ignore_index=True)
        combined_df['cat'] = combined_df['name'].apply(get_category)

        filtered = combined_df.dropna(subset=['cat', 'value']).copy()

        for (app_no, pat_id), group in filtered.groupby(['app_no', 'pat_id']):
            row = {'app_no': str(app_no).strip(), 'pat_id': str(pat_id).strip()}

            data_pool = {}
            for _, item in group.iterrows():
                # Seamlessly merge indicators from both tables into one dict
                data_pool[item['cat']] = str(item['value'])

            # Core cleaning logic
            row['lsm'] = clean_lsm(data_pool.get('lsm', np.nan))
            row['ast'] = clean_numeric(data_pool.get('ast', np.nan))
            row['alt'] = clean_numeric(data_pool.get('alt', np.nan))
            row['plt'] = clean_numeric(data_pool.get('plt', np.nan))

            dia_raw = data_pool.get('has_diabetes', "")
            row['diabetes_num'] = parse_yes_no(dia_raw)

            records.append(row)

    if records:
        wide_items = pd.DataFrame(records)
    else:
        wide_items = pd.DataFrame(columns=['app_no', 'pat_id', 'lsm', 'ast', 'alt', 'plt', 'diabetes_num'])

    # 2. Merge basic information
    base_df['app_no'] = base_df['app_no'].astype(str).str.strip()
    base_df['pat_id'] = base_df['pat_id'].astype(str).str.strip()

    if 'app_no' not in base_df.columns:
        base_df['app_no'] = ""

    final = pd.merge(base_df, wide_items, on=['app_no', 'pat_id'], how='left')

    # 3. Scoring and result generation
    results = []
    GAUGE_REF = "0-0.451|0.451-0.679|0.679-1.0"

    for _, r in final.iterrows():
        age = clean_numeric(r.get('app_pat_age'))
        sex_raw = r.get('pat_sex', '')
        sex_num = parse_gender_to_num(sex_raw)

        lsm = r.get('lsm')
        ast = r.get('ast')
        alt = r.get('alt')
        plt = r.get('plt')

        diabetes_num = r.get('diabetes_num')
        diabetes_num = int(diabetes_num) if pd.notna(diabetes_num) else 0

        score, risk_level, code, suggestion, info = calculate_agile3_plus(age, sex_num, diabetes_num, lsm, ast, alt,
                                                                          plt)

        results.append({
            'app_no': r.get('app_no'),
            'pat_id': r.get('pat_id'),
            'app_no': r.get('app_no'),
            'pat_name': r.get('pat_name', 'unknown'),
            'app_pat_age': int(age) if pd.notna(age) else "-",
            'pat_sex': sex_raw if pd.notna(sex_raw) and str(sex_raw).strip() != "" else "-",
            'prior_diabetes': diabetes_num,
            'AST': round(ast, 2) if pd.notna(ast) else "-",
            'ALT': round(alt, 2) if pd.notna(alt) else "-",
            'PLT': round(plt, 2) if pd.notna(plt) else "-",
            'liver_stiffness': round(lsm, 2) if pd.notna(lsm) else "-",
            'AAR': info.get('AAR', "-"),
            'predicted_score': round(score, 4) if score is not None else "-",
            'risk_level': risk_level,
            'code': code,
            'gauge_reference': GAUGE_REF,
            'suggestion': suggestion,
            'model_name': 'Advanced Fibrosis Agile 3+'
        })

    # 4. Export results
    if results:
        res_df = pd.DataFrame(results)
        res_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"--- Assessment complete: {len(res_df)} results generated, exported to {output_path} ---")
    else:
        print("--- Assessment complete, but no valid results were generated ---")


if __name__ == "__main__":
    process_data(
        base_path="base.csv",
        item_path="item_result.csv",
        summary_path="item_summary.csv",
        output_path="agile3_result.csv"
    )
