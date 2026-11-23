import pandas as pd
import numpy as np
import re

# ==========================================
# 1. Field mapping (for the CKB colorectal cancer model)
# ==========================================
FIELD_MAPPING = {
    'age': ['app_pat_age', 'age'],
    'gender': ['pat_sex', 'sex'],
    'bmi': ['(body mass index) BMI', 'bmi', 'BMI'],
    'height': ['height'],
    'weight': ['weight'],
    'daily_cigs': ['smoking_history', 'daily_cigarettes', 'cigarettes_per_day', 'daily_smoking'],
    'smoke_years': ['smoking_history', 'smoking_duration', 'smoking_years', 'years_smoked'],
    'alcohol_freq': ['drinking_frequency', 'drinking_habit', 'alcohol_status'], # drinking frequency > 3 days/week
    'red_meat_freq': ['red_meat_intake', 'meat_diet_habit', 'dietary_preference'], # red meat intake > 5 days/week
    'diabetes': ['diabetes_history', 'prior_diabetes', 'type2_diabetes'],
    'gallstones': ['gallstone', 'gallbladder_stone', 'cholelithiasis'],
    # Change: relax to any-cancer history in first-degree relatives
    'family_hist': ['family_history', 'tumor_family_history', 'cancer_family_history', 'first_degree_cancer', 'first_degree_cancer_history']
}


# ==========================================
# 2. Basic cleaning and matching utilities
# ==========================================
def normalize_str(s):
    if pd.isna(s): return ""
    return re.sub(r'[\(\)\（\）\[\]\【\】\{\}\s]', '', str(s)).lower()


def get_category(name):
    """Exact-match the standard label"""
    norm_name = normalize_str(name)
    for key, patterns in FIELD_MAPPING.items():
        for p in patterns:
            if normalize_str(p) == norm_name: return key
    return None


def clean_numeric(val):
    if pd.isna(val) or str(val).strip() == "": return np.nan
    try:
        match = re.search(r"[-+]?\d*\.\d+|\d+", str(val))
        return float(match.group()) if match else np.nan
    except:
        return np.nan


def parse_boolean(val, true_keywords=None):
    if true_keywords is None:
        true_keywords = ['yes', 'present', 'positive', 'true', '1']
    s = str(val).lower()
    if pd.isna(val) or s == 'nan' or s == '': return None
    if any(k in s for k in true_keywords):
        return True
    return False


def parse_freq(val, threshold):
    """
    Frequency parsing: prefer numbers (days/week) in the text compared with cut-offs; fall back to fuzzy matching when absent
    """
    if pd.isna(val) or str(val).strip() == "" or str(val).lower() == 'nan': return None
    s = str(val).lower()

    # Try to extract a number from the text (e.g. "3 times/week" -> 3)
    match = re.search(r"(\d+)", s)
    if match:
        days = int(match.group(1))
        return days >= threshold

    # Fall back to keyword-based fuzzy detection when no explicit number is given
    positive_kw = ['frequent', 'often', 'always', 'daily', 'high', 'yes']
    if any(k in s for k in positive_kw):
        return True
    return False


# ==========================================
# 3. CKB colorectal cancer (CRC) core scoring logic
# ==========================================
def calculate_ckb_crc_score(data_row):
    score = 0
    missing_items = []

    age = data_row.get('age')
    if pd.isna(age):
        missing_items.append("age")
    else:
        # Change: add age inclusion bounds and baseline points
        if age < 40 or age > 75:
            return None, "-", "-", f"age {age} outside the model range (40-75)", "-"
        elif 40 <= age < 45:
            score += 0
        elif 45 <= age < 50:
            score += 5
        elif 50 <= age < 55:
            score += 9
        elif 55 <= age < 60:
            score += 12
        elif 60 <= age < 65:
            score += 16
        elif 65 <= age < 70:
            score += 19
        elif 70 <= age <= 75:
            score += 22

    gender = str(data_row.get('gender', '')).strip()
    if not gender or gender == 'nan':
        missing_items.append("sex")
    else:
        if 'male' in gender or 'Man' in gender or 'Male' in gender:
            score += 2

    bmi = data_row.get('bmi')
    if pd.isna(bmi):
        missing_items.append("BMI")
    else:
        if bmi >= 28: score += 3

    cigs = data_row.get('daily_cigs')
    years = data_row.get('smoke_years')
    if pd.isna(cigs) or pd.isna(years):
        missing_items.append("smoking_data")
    else:
        if (cigs / 20.0) * years >= 30: score += 3

    alcohol = data_row.get('alcohol_freq')
    if alcohol is None:
        missing_items.append("alcohol_status")
    else:
        if alcohol: score += 2

    meat = data_row.get('red_meat_freq')
    if meat is None:
        missing_items.append("red_meat_intake")
    else:
        if meat: score += 1

    diabetes = data_row.get('diabetes')
    if diabetes is None:
        missing_items.append("diabetes_history")
    else:
        if diabetes: score += 2

    gallstones = data_row.get('gallstones')
    if gallstones is None:
        missing_items.append("gallstone_history")
    else:
        if gallstones: score += 2

    fam = data_row.get('family_hist')
    if fam is None:
        missing_items.append("family_history")
    else:
        if fam: score += 1

    # If anything is missing, return None or a placeholder for that result item
    if missing_items:
        advice = f"cannot complete assessment; missing indicators: {', '.join(missing_items)}"
        return None, "-", "-", advice, "-"

    # Change: replace the probability with the source's 10-year cumulative incidence value
    if score < 6:
        risk_level, prob, advice, risk_code = "low risk", "0.24%", "low risk; follow routine check-ups.", "18.4.1"
    elif 6 <= score <= 19:
        risk_level, prob, advice, risk_code = "medium risk", "0.58%", "recommend FIT screening every 1-2 years.", "18.4.2"
    else:
        risk_level, prob, advice, risk_code = "high risk", "1.59%", "strongly recommend scheduling a full colonoscopy.", "18.4.13"

    return score, risk_level, prob, advice, risk_code


# ==========================================
# 4. Main data-processing routine
# ==========================================
def process_data(base_path, item_path, output_path):
    print("--- Starting CKB colorectal cancer risk model ---")

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

    # 1. Pivot to wide table
    records = []
    if not item_df.empty:
        item_df['cat'] = item_df['res_itm_name'].apply(get_category)
        filtered = item_df.dropna(subset=['cat', 'res_value']).copy()
        grouped = filtered.groupby(['app_no', 'pat_id'])

        for (app_no, pat_id), group in grouped:
            row_raw = {'app_no': str(app_no).strip(), 'pat_id': str(pat_id).strip()}
            data_pool = {k: [] for k in FIELD_MAPPING.keys()}
            for _, item in group.iterrows():
                data_pool[item['cat']].append(str(item['res_value']))

            def get_first_valid(lst, parser=clean_numeric):
                for v in lst:
                    res = parser(v)
                    if res is not None and not (isinstance(res, float) and pd.isna(res)):
                        return res
                return np.nan if parser == clean_numeric else None

            h = get_first_valid(data_pool.get('height', []))
            w = get_first_valid(data_pool.get('weight', []))
            row_raw['height'] = h
            row_raw['weight'] = w

            bmi_val = get_first_valid(data_pool.get('bmi', []))
            if pd.isna(bmi_val) and pd.notna(h) and pd.notna(w) and h > 0:
                h_calc = h / 100.0 if h > 3 else h
                bmi_val = round(w / (h_calc * h_calc), 2)
            row_raw['bmi'] = bmi_val

            row_raw['daily_cigs'] = get_first_valid(data_pool.get('daily_cigs', []))
            row_raw['smoke_years'] = get_first_valid(data_pool.get('smoke_years', []))

            # Change: apply the thresholded frequency parser
            row_raw['alcohol_freq'] = get_first_valid(data_pool.get('alcohol_freq', []), lambda v: parse_freq(v, 3))
            row_raw['red_meat_freq'] = get_first_valid(data_pool.get('red_meat_freq', []), lambda v: parse_freq(v, 5))

            row_raw['diabetes'] = get_first_valid(data_pool.get('diabetes', []), parse_boolean)
            row_raw['gallstones'] = get_first_valid(data_pool.get('gallstones', []), parse_boolean)
            row_raw['family_hist'] = get_first_valid(data_pool.get('family_hist', []), parse_boolean)
            records.append(row_raw)

    wide_items = pd.DataFrame(records) if records else pd.DataFrame(columns=['app_no', 'pat_id'])

    # 2. Merge basic information (left join)
    base_df['app_no'] = base_df['app_no'].astype(str).str.strip()
    base_df['pat_id'] = base_df['pat_id'].astype(str).str.strip()

    overlap_cols = [c for c in wide_items.columns if c in base_df.columns and c not in ['app_no', 'pat_id']]
    if overlap_cols:
        base_df = base_df.drop(columns=overlap_cols)

    final_df = pd.merge(base_df, wide_items, on=['app_no', 'pat_id'], how='left')

    # 3. Scoring loop
    results = []
    print("computing risk...")
    for _, r in final_df.iterrows():
        clean_row = {
            'age': clean_numeric(r.get('app_pat_age')),
            'gender': str(r.get('pat_sex', '')),
            'bmi': r.get('bmi'),
            'daily_cigs': r.get('daily_cigs'),
            'smoke_years': r.get('smoke_years'),
            'alcohol_freq': r.get('alcohol_freq'),
            'red_meat_freq': r.get('red_meat_freq'),
            'diabetes': r.get('diabetes'),
            'gallstones': r.get('gallstones'),
            'family_hist': r.get('family_hist')
        }

        score, level, prob, advice, r_code = calculate_ckb_crc_score(clean_row)

        def fmt_input(v):
            return "missing" if (pd.isna(v) or v is None) else v

        def fmt_bool(v):
            if v is None: return "missing"
            return "yes" if v else "no"

        results.append({
            'app_no': r['app_no'],
            'pat_id': r['pat_id'],
            'model_name': 'Colorectal Cancer CORES',
            'prediction_horizon': 'next 10 years',
            'name': r.get('pat_name', 'unknown'),
            'sex': clean_row['gender'] if clean_row['gender'] not in ['nan', ''] else 'missing',
            'age': int(clean_row['age']) if not pd.isna(clean_row['age']) else 'missing',
            # --- Result items (missing shown as "-") ---
            'CKB_total_score': score if score is not None else "-",
            'risk_level': level,
            'risk_probability': prob,
            'code': r_code,
            'gauge_reference': '0-5|6-19|20-40',
            'recommendation': advice,
            # --- Input items (missing shown as "missing") ---
            'height': fmt_input(r.get('height')),
            'weight': fmt_input(r.get('weight')),
            'BMI': fmt_input(clean_row['bmi']),
            'daily_cigarettes': fmt_input(clean_row['daily_cigs']),
            'smoking_years': fmt_input(clean_row['smoke_years']),
            'frequent_drinking': fmt_bool(clean_row['alcohol_freq']),
            'frequent_red_meat': fmt_bool(clean_row['red_meat_freq']),
            'diabetes_history': fmt_bool(clean_row['diabetes']),
            'gallstone_history': fmt_bool(clean_row['gallstones']),
            'family_history': fmt_bool(clean_row['family_hist'])
        })

    if results:
        res_df = pd.DataFrame(results)
        res_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"Assessment complete! Results saved to: {output_path}")


if __name__ == "__main__":
    process_data("base.csv", "item_result.csv", "ckb_crc_risk_assessment_result.csv")