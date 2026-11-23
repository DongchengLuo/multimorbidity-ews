import pandas as pd
import numpy as np
import re

# ==========================================
# 1. Field mapping (see the precise field-matching reference table)
# ==========================================
FIELD_MAPPING = {
    'age': ['app_pat_age', 'age', 'date_of_birth'],
    'gender': ['pat_sex', 'sex', 'gender'],
    'LSM': ['LSM', 'liver_stiffness', 'e_value', 'kpa'],
    'plt': ['plt', 'platelet', 'platelet_count', 'PLT'],
    'AFP': ['afp', 'AFP', 'AFP_quantitative'],
    'ALB': ['albumin', 'ALB'],
    'cirrhosis': ['cirrhosis', 'cirrhosis_history', 'has_cirrhosis'],
    'antiviral': ['antiviral', 'antiviral_therapy', 'on_antiviral', 'past_history']
}

# Risk-criteria constants (for logic and display)
# Note: added the 'r' prefix to fix a SyntaxWarning
RISK_STANDARDS = [
    {"level": "low risk", "code": "C34.4.1", "range": r"Total Score \le 87.2", "prob": "~1.0%",
     "suggest": "routine follow-up (annual)"},
    {"level": "medium risk", "code": "C34.4.2", "range": "87.2 < Total Score < 121.1", "prob": "~5.7%",
     "suggest": "close monitoring (every 6 months)"},
    {"level": "high risk", "code": "C34.4.3", "range": r"Total Score \ge 121.1", "prob": "~17.8%",
     "suggest": "recommend intensified screening (CT/MRI)"}
]


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
            # Change: use strict exact matching; drop fuzzy matching
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
    # Change: take the first value in the list instead of averaging
    if not val_list:
        return np.nan
    return clean_numeric(val_list[0])


def parse_boolean(val_list):
    if not val_list: return None
    # Use the first record for the decision
    combined = normalize_str(val_list[0])
    positive_kws = ['1', 'yes', 'y', 'true', 'present', 'positive', 'taking', 'in_use']
    return any(kw in combined for kw in positive_kws)


def parse_gender(val_list):
    if not val_list or pd.isna(val_list[0]) or str(val_list[0]).strip() == "":
        return None
    s = normalize_str(val_list[0])
    if s in ['1', 'male', 'm']: return 'Male'
    if s in ['0', 'female', 'f']: return 'Female'
    return None


# ==========================================
# 3. HCC-LSM core scoring logic
# ==========================================
def calculate_hcc_lsm_score(gender, age, cirrhosis, antiviral, plt, AFP, LSM, ALB):
    score = 0
    details = {}

    # 1. Sex (Male=30, Female=0)
    s_gender = 30 if gender == 'Male' else 0
    score += s_gender

    # 2. Cirrhosis (Yes=28, No=0)
    s_cirr = 28 if cirrhosis else 0
    score += s_cirr

    # 3. Age (>=50=18, <50=0)
    s_age = 18 if age >= 50 else 0
    score += s_age

    # 4. Antiviral therapy (Yes=15, No=0)
    s_anti = 15 if antiviral else 0
    score += s_anti

    # 5. Platelets (<150=12, >=150=0)
    s_plt = 12 if plt < 150 else 0
    score += s_plt

    # 6. Alpha-fetoprotein (>=20=3, <20=0)
    s_afp = 3 if AFP >= 20 else 0
    score += s_afp

    # 7. Liver stiffness (LSM) weight
    s_lsm = 0
    if LSM < 7.3:
        s_lsm = 0
    elif 7.3 <= LSM < 9.7:
        s_lsm = 14
    elif 9.7 <= LSM < 12.4:
        s_lsm = 32
    elif 12.4 <= LSM <= 17.5:
        s_lsm = 8
    elif LSM > 17.5:
        s_lsm = 38
    score += s_lsm

    # 8. Albumin formula: (60 - ALB) * 2.3
    s_alb = (60 - ALB) * 2.3
    if s_alb < 0: s_alb = 0
    score += s_alb

    return round(score, 2), details


# ==========================================
# 4. Main data-processing routine
# ==========================================
def process_data(base_path, item_path, output_path):
    print("--- Starting HCC-LSM liver cancer risk model ---")

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
        print("Error: base data file could not be read or is empty")
        return

    base_df['app_no'] = base_df['app_no'].astype(str).str.strip()
    base_df['pat_id'] = base_df['pat_id'].astype(str).str.strip()

    # 1. Pivot to wide table
    records = []
    if not item_df.empty:
        item_df['cat'] = item_df['res_itm_name'].apply(get_category)
        filtered = item_df.dropna(subset=['cat', 'res_value']).copy()

        for (app_no, pat_id), group in filtered.groupby(['app_no', 'pat_id']):
            row = {'app_no': str(app_no).strip(), 'pat_id_merge': str(pat_id).strip()}
            data_pool = {k: [] for k in FIELD_MAPPING.keys()}
            for _, item in group.iterrows():
                data_pool[item['cat']].append(str(item['res_value']))

            # Change: call the new function to take only the first usable record
            row['lsm_val'] = get_first_numeric(data_pool.get('LSM', []))
            row['plt_val'] = get_first_numeric(data_pool.get('plt', []))
            row['afp_val'] = get_first_numeric(data_pool.get('AFP', []))
            row['alb_val'] = get_first_numeric(data_pool.get('ALB', []))
            row['cirrhosis_bool'] = parse_boolean(data_pool.get('cirrhosis', []))
            row['antiviral_bool'] = parse_boolean(data_pool.get('antiviral', []))
            records.append(row)

    wide_items = pd.DataFrame(records)

    # Change: left join with base_df as the driving table so patients without records are not skipped
    if wide_items.empty:
        final = base_df.copy()
    else:
        final = pd.merge(base_df, wide_items, left_on=['app_no', 'pat_id'], right_on=['app_no', 'pat_id_merge'],
                         how='left')

    # Helper for formatted output
    def format_val(val, is_int=False):
        if pd.isna(val) or val is None or str(val).strip() == '':
            return 'missing (defaulted to no)'
        return int(val) if is_int else round(val, 2)

    def format_bool(val):
        if val is None or pd.isna(val):
            return 'missing'
        return 'yes' if val else 'no'

    # 2. Scoring and tiering
    results = []
    for _, r in final.iterrows():
        age = clean_numeric(r.get('app_pat_age'))
        gender = parse_gender([str(r.get('pat_sex', ''))])
        lsm = r.get('lsm_val')
        plt = r.get('plt_val')
        afp = r.get('afp_val')
        alb = r.get('alb_val')
        cirr_bool = r.get('cirrhosis_bool')
        anti_bool = r.get('antiviral_bool')

        # Check whether required core computation indicators are missing
        is_missing = any(pd.isna(v) for v in [age, lsm, plt, afp, alb]) or gender is None

        if not is_missing:
            # Even when history is missing, treat it as False (no points) for scoring stability
            calc_cirr = cirr_bool if cirr_bool is not None else False
            calc_anti = anti_bool if anti_bool is not None else False

            score, _ = calculate_hcc_lsm_score(gender, age, calc_cirr, calc_anti, plt, afp, lsm, alb)

            # Risk logic determination
            if score <= 87.2:
                std = RISK_STANDARDS[0]
            elif score < 121.1:
                std = RISK_STANDARDS[1]
            else:
                std = RISK_STANDARDS[2]

            out_score = score
            out_level = std['level']
            out_code = std['code']
            out_prob = std['prob']
            out_suggest = std['suggest']
        else:
            # Output '-' in result columns when indicators are missing, but still emit the person's information
            out_score = '-'
            out_level = '-'
            out_code = '-'
            out_prob = '-'
            out_suggest = '-'

        results.append({
            'model_name': 'Liver Cancer HCC-LSM',
            'pat_id': r.get('pat_id'),
            'app_no': r.get('app_no'),
            'name': r.get('pat_name', 'unknown') if pd.notna(r.get('pat_name')) else 'missing',
            'sex': gender if gender else 'missing',
            'age': format_val(age, is_int=True),
            'HCC-LSM_total_score': out_score,
            'risk_level': out_level,
            'code': out_code,
            'estimated_5y_cumulative_cancer_rate': out_prob,
            'recommendation': out_suggest,
            'LSM': format_val(lsm),
            'PLT': format_val(plt),
            'AFP': format_val(afp),
            'ALB': format_val(alb),
            'cirrhosis_history': format_bool(cirr_bool),
            'antiviral_therapy_history': format_bool(anti_bool)
        })

    # 3. Result export and standard output
    if results:
        res_df = pd.DataFrame(results)
        res_df.to_csv(output_path, index=False, encoding='utf-8-sig')

        print("\n" + "=" * 50)
        print("assessment task complete!")
        print(f"total records: {len(res_df)}")
        print(f"result path: {output_path}")
        print("-" * 50)
        print("HCC-LSM risk-tier criteria:")
        print(f"{'risk_level':<10} | {'Code':<10} | {'score_range':<25} | {'5y_cancer_probability'}")
        for s in RISK_STANDARDS:
            clean_range = s['range'].replace('\\le', '<=').replace('\\ge', '>=')
            print(f"{s['level']:<10} | {s['code']:<10} | {clean_range:<25} | {s['prob']}")
        print("=" * 50 + "\n")
    else:
        print("--- Assessment complete, but no results (check data completeness) ---")


if __name__ == "__main__":
    process_data(
        "base.csv",
        "item_result.csv",
        "hcc_lsm_final.csv"
    )
