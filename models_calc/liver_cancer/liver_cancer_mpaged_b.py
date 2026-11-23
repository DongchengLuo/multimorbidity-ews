import pandas as pd
import numpy as np
import re
import os

# ==========================================
# 1. Field mapping
# ==========================================
FIELD_MAPPING = {
    'age': ['app_pat_age', 'age'],
    'gender': ['pat_sex', 'sex', 'gender'],
    'plt': ['platelet_count', 'PLT', 'platelet', 'plt'],
    'alb': ['albumin', 'ALB'],
    'history': ['past_history', 'past_medical_history', 'disease_history', 'history']
}


# ==========================================
# 2. Basic cleaning and matching logic
# ==========================================
def normalize_str(s):
    if pd.isna(s): return ""
    # Keep fuzzy handling for symbols: strip brackets/spaces and lowercase
    return re.sub(r'[\(\)\（\）\[\]\【\】\{\}\s]', '', str(s)).lower()


def get_category(name):
    norm_name = normalize_str(name)
    for key, patterns in FIELD_MAPPING.items():
        for p in patterns:
            # [Core change]: use '==' instead of 'in' for exact text matching
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


def is_chb_patient(history_text):
    if pd.isna(history_text): return False
    text = str(history_text).strip()
    keywords = ['hepatitis_b', 'chronic_hepatitis_b', 'CHB', 'HBV', 'HBsAg+/HBeAb+/HBcAb+', 'HBsAg+/HBeAg+/HBcAb+']
    negation_words = ['denies', 'absent', 'no disease', 'exclude', 'negative']
    if any(k in text for k in keywords):
        segments = re.split(r'[，。；,; \n]', text)
        for seg in segments:
            if any(k in seg for k in keywords):
                if not any(nw in seg for nw in negation_words):
                    return True
    return False


# ==========================================
# 3. mPAGE-B core scoring logic
# ==========================================
def calculate_mpageb_score(age, gender, plt, alb):
    if any(pd.isna([age, gender, plt, alb])):
        return None, None

    score = 0
    debug_info = {}

    # 1. Age scoring
    if age < 30:
        s_age = 0
    elif 30 <= age <= 39:
        s_age = 3
    elif 40 <= age <= 49:
        s_age = 5
    elif 50 <= age <= 59:
        s_age = 7
    elif 60 <= age <= 69:
        s_age = 9
    else:
        s_age = 11
    score += s_age
    debug_info['age_points'] = s_age

    # 2. Sex score (1=male, 0=female)
    s_gen = 2 if gender == 1 else 0
    score += s_gen
    debug_info['gender_points'] = s_gen

    # 3. Platelet score
    if plt >= 250:
        s_plt = 0
    elif 200 <= plt < 250:
        s_plt = 2
    elif 150 <= plt < 200:
        s_plt = 3
    elif 100 <= plt < 150:
        s_plt = 4
    else:
        s_plt = 5
    score += s_plt
    debug_info['plt_points'] = s_plt

    # 4. Albumin score
    alb_val = alb if alb < 10 else alb / 10.0
    if alb_val >= 4.0:
        s_alb = 0
    elif 3.5 <= alb_val < 4.0:
        s_alb = 1
    elif 3.0 <= alb_val < 3.5:
        s_alb = 2
    else:
        s_alb = 3
    score += s_alb
    debug_info['alb_points'] = s_alb

    return score, debug_info


# ==========================================
# 4. Main data-processing routine
# ==========================================
def process_data(base_path, item_path, output_path):
    print("--- Starting mPAGE-B liver cancer risk model ---")

    def read_file_safe(path):
        if not os.path.exists(path): return pd.DataFrame()
        if path.endswith('.csv'):
            for enc in ['gbk', 'utf-8-sig', 'utf-8']:
                try:
                    return pd.read_csv(path, encoding=enc, low_memory=False)
                except:
                    continue
        else:
            try:
                return pd.read_excel(path)
            except:
                pass
        return pd.DataFrame()

    base_df = read_file_safe(base_path)
    item_df = read_file_safe(item_path)

    if base_df.empty:
        print("Error: base data file could not be read or is empty.")
        return

    base_df['app_no'] = base_df['app_no'].astype(str).str.strip()
    base_df['pat_id'] = base_df['pat_id'].astype(str).str.strip()

    records = []
    if not item_df.empty:
        item_df['cat'] = item_df['res_itm_name'].apply(get_category)
        filtered = item_df.dropna(subset=['cat']).copy()

        for (app_no, pat_id), group in filtered.groupby(['app_no', 'pat_id']):
            row_data = {'app_no': str(app_no).strip(), 'pat_id': str(pat_id).strip(), 'is_chb': False}
            data_pool = {k: [] for k in FIELD_MAPPING.keys()}

            for _, item in group.iterrows():
                val = item['res_value']
                cat = item['cat']
                if pd.notna(val):
                    data_pool[cat].append(val)

            history_str = " ".join([str(x) for x in data_pool.get('history', [])])
            if is_chb_patient(history_str):
                row_data['is_chb'] = True

            plt_data = data_pool.get('plt', [])
            alb_data = data_pool.get('alb', [])

            row_data['plt_val'] = clean_numeric(plt_data[0]) if plt_data else np.nan
            row_data['alb_val'] = clean_numeric(alb_data[0]) if alb_data else np.nan

            records.append(row_data)

    if records:
        wide_items = pd.DataFrame(records)
    else:
        wide_items = pd.DataFrame(columns=['app_no', 'pat_id', 'plt_val', 'alb_val', 'is_chb'])

    final = pd.merge(base_df, wide_items, on=['app_no', 'pat_id'], how='left')

    results = []
    for _, r in final.iterrows():
        age = clean_numeric(r.get('app_pat_age'))
        raw_sex = str(r.get('pat_sex', ''))
        gender = 1 if ('male' in raw_sex or raw_sex == '1') else (0 if ('female' in raw_sex or raw_sex == '2') else np.nan)

        plt = r.get('plt_val')
        alb = r.get('alb_val')

        score, details = calculate_mpageb_score(age, gender, plt, alb)

        age_str = "missing" if pd.isna(age) else str(int(age))
        gender_str = "missing" if pd.isna(gender) else ('male' if gender == 1 else 'female')
        plt_str = "missing" if pd.isna(plt) else f"{plt:.1f}"
        alb_str = "missing" if pd.isna(alb) else f"{alb:.1f}"

        if score is None:
            score_str = "-"
            risk_level = "-"
            risk_code = "-"
            suggestion = "-"
        else:
            score_str = str(score)
            if score <= 8:
                risk_level = "low risk"
                risk_code = "C22.3.1"
                suggestion = "very low 5-year liver cancer risk; routine follow-up"
            elif 9 <= score <= 12:
                risk_level = "medium risk"
                risk_code = "C22.3.2"
                suggestion = "some liver cancer risk; ultrasound every 6 months"
            else:
                risk_level = "high risk"
                risk_code = "C22.3.3"
                suggestion = "high liver cancer risk; screen every 3-6 months"

        results.append({
            'model_name': 'mPAGE-B Asian HBV Liver Cancer Risk Score',
            'pat_id': r['pat_id'],
            'app_no': r['app_no'],
            'pat_name': r.get('pat_name', 'unknown'),
            'sex': gender_str,
            'age': age_str,
            'PLT': plt_str,
            'ALB': alb_str,
            'predicted_score': score_str,
            'risk_level': risk_level,
            'prediction_horizon': 'next 5 years',
            'code': risk_code,
            'recommendation': suggestion
        })

    if results:
        res_df = pd.DataFrame(results)
        res_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"--- Assessment complete: {len(res_df)} results generated, exported to {output_path} ---")
    else:
        print("--- Assessment complete, no valid results ---")


if __name__ == "__main__":
    process_data(
        "base.csv",
        "item_result.csv",
        "mpage_b_risk.csv"
    )
