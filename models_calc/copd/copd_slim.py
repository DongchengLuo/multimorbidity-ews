import pandas as pd
import numpy as np
import math
import re
import warnings

# Ignore all warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. Field mapping and SLIM model parameters
# ==========================================
FIELD_MAPPING = {
    'gender': ['pat_sex'],
    'age': ['app_pat_age'],
    'smoke_history': ['smoking_history', 'personal_history_smoking', 'smoking_status'],  # primary long-text field
    'cpd': ['daily_cigarettes'],  # keep separate fields in case they are filled separately
    'smoke_years': ['smoking_years', 'smoking_duration'],
    'history': ['past_history', 'past_medical_history'],
    'fev_ratio': ['FEV1/FVC'],
    'bmi': ['BMI', 'bmi'],
    'chronic_bronchitis': ['chronic_bronchitis', 'sputum', 'chronic_cough'],
    'height': ['height'],
    'weight': ['weight']
}

SLIM_COEFS = {
    'intercept': -3.89,
    'beta_fev': 2.73,
    'beta_smoke': 1.22,
    'beta_bmi': 0.88,
    'beta_cb': 0.63,
    'threshold': 16.0
}


# ==========================================
# 2. Basic matching and cleaning logic
# ==========================================
def normalize_str(s):
    if pd.isna(s): return ""
    return re.sub(r'[\(\)\（\）\[\]\【\】\{\}]', '', str(s)).strip().lower()


def get_category(name):
    norm_name = normalize_str(name)
    for key, patterns in FIELD_MAPPING.items():
        for p in patterns:
            if normalize_str(p) in norm_name: return key
    return None


def clean_numeric(val):
    if pd.isna(val) or str(val).strip() == "": return np.nan
    try:
        match = re.search(r"[-+]?\d*\.\d+|\d+", str(val))
        return float(match.group()) if match else np.nan
    except:
        return np.nan


# --- Added: smart parsing of long smoking-history text ---
def parse_smoking_text(text):
    """Extract daily cigarette amount and smoking years from free text"""
    if pd.isna(text) or str(text).strip() == "":
        return np.nan, np.nan

    s = str(text).lower()

    # 1. If non-smoking is explicit and no number appears, return zeros for both
    if any(kw in s for kw in ['absent', 'denies', 'non_smoker', 'never_smoked']) and not re.search(r'\d+', s):
        return 0.0, 0.0

    cpd = np.nan
    years = np.nan

    # 2. Extract years (match: "20 years", "15.5 years", "smoked for 30 years")
    year_match = re.search(r'(\d+(?:\.\d+)?)\s*years?', s)
    if year_match:
        years = float(year_match.group(1))

    # 3. Extract daily cigarette amount
    # Prefer "cigarettes" or "sticks" (match: "20 cigarettes", "10 sticks")
    zhi_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:cigarettes|sticks)', s)
    if zhi_match:
        cpd = float(zhi_match.group(1))
    else:
        # If no "cigarettes", look for "packs" and multiply by 20 (match: "1 pack", "0.5 pack")
        bao_match = re.search(r'(\d+(?:\.\d+)?)\s*packs?', s)
        if bao_match:
            cpd = float(bao_match.group(1)) * 20.0

    return cpd, years


def is_already_copd(history_str):
    keywords = ['copd', 'chronic_obstructive_pulmonary_disease', 'emphysema']
    s = str(history_str)
    if any(kw in s for kw in keywords):
        negations = ['denies', 'absent', 'not found', 'normal']
        for kw in keywords:
            pos = s.find(kw)
            if pos != -1 and not any(neg in s[max(0, pos - 5):pos] for neg in negations):
                return True
    return False


# ==========================================
# 3. Core computation function
# ==========================================
def calculate_slim_risk_logic(fev_ratio, bmi, cpd, duration, has_cb):
    try:
        if pd.isna(cpd) or pd.isna(duration):
            pack_years = 0
        else:
            pack_years = (float(cpd) / 20.0) * float(duration)

        logit_l = SLIM_COEFS['intercept']

        if float(fev_ratio) < 0.75: logit_l += SLIM_COEFS['beta_fev']
        if pack_years >= 30: logit_l += SLIM_COEFS['beta_smoke']
        if float(bmi) <= 25: logit_l += SLIM_COEFS['beta_bmi']
        if int(has_cb) == 1: logit_l += SLIM_COEFS['beta_cb']

        prob = (math.exp(logit_l) / (1 + math.exp(logit_l))) * 100
        return prob, pack_years
    except:
        return np.nan, np.nan


# ==========================================
# 4. Main data-processing logic
# ==========================================
def process_data(base_path, item_path, output_path):
    print("--- Starting SLIM risk prediction (smart text parsing) ---")

    GAUGE_REF = "0-16%|16%-100%"

    def read_csv_safe(path):
        for enc in ['gbk', 'utf-8-sig', 'utf-8']:
            try:
                return pd.read_csv(path, encoding=enc, low_memory=False)
            except:
                continue
        return pd.DataFrame()

    base_df = read_csv_safe(base_path)
    item_df = read_csv_safe(item_path)
    if base_df.empty or item_df.empty: return

    item_df['cat'] = item_df['res_itm_name'].apply(get_category)
    filtered = item_df.dropna(subset=['cat', 'res_value']).copy()

    def get_first_valid(val_list):
        for v in val_list:
            num = clean_numeric(v)
            if not pd.isna(num): return num
        return np.nan

    records = []
    for (app_no, pat_id), group in filtered.groupby(['app_no', 'pat_id']):
        row = {'app_no': str(app_no).strip(), 'pat_id': str(pat_id).strip()}
        data_pool = {k: [] for k in FIELD_MAPPING.keys()}
        for _, item in group.iterrows():
            data_pool[item['cat']].append(str(item['res_value']))

        # ====================================================
        # === Core change: unified three-way smoking-data parsing ===
        # ====================================================
        # 1. Capture the long smoking-history text field
        smoke_history_str = " ".join(data_pool.get('smoke_history', []))

        # 2. Call smart regex parsing to extract cpd and years
        parsed_cpd, parsed_years = parse_smoking_text(smoke_history_str)

        # 3. Dual-track compatibility: prefer dedicated daily-amount/years fields; otherwise use parsed values
        direct_cpd = get_first_valid(data_pool.get('cpd', []))
        row['cpd'] = direct_cpd if not pd.isna(direct_cpd) else parsed_cpd

        direct_years = get_first_valid(data_pool.get('smoke_years', []))
        row['smoke_years'] = direct_years if not pd.isna(direct_years) else parsed_years

        # Keep the raw long text for later verification
        row['raw_smoking_text'] = smoke_history_str if smoke_history_str.strip() else "not_recorded"
        # ====================================================

        row['fev_ratio'] = get_first_valid(data_pool.get('fev_ratio', []))

        bmi_val = get_first_valid(data_pool.get('bmi', []))
        if pd.isna(bmi_val):
            height_val = get_first_valid(data_pool.get('height', []))
            weight_val = get_first_valid(data_pool.get('weight', []))
            if not pd.isna(height_val) and not pd.isna(weight_val) and height_val > 0:
                height_m = height_val / 100.0 if height_val > 3.0 else height_val
                bmi_val = weight_val / (height_m ** 2)
        row['bmi'] = bmi_val

        history_str = " ".join(set(data_pool.get('history', []) + data_pool.get('chronic_bronchitis', [])))
        row['has_cb'] = 1 if any(kw in history_str for kw in ['bronchitis', 'sputum', 'cough']) else 0
        row['history'] = history_str
        records.append(row)

    wide_items = pd.DataFrame(records)
    base_df['app_no'] = base_df['app_no'].astype(str).str.strip()
    base_df['pat_id'] = base_df['pat_id'].astype(str).str.strip()
    final = pd.merge(wide_items, base_df, on=['app_no', 'pat_id'], how='inner')

    results = []
    for _, r in final.iterrows():
        age = clean_numeric(r.get('app_pat_age'))

        if pd.isna(age) or pd.isna(r['fev_ratio']) or pd.isna(r['bmi']):
            calc_status = "missing"
        elif is_already_copd(r['history']) or age < 40:
            calc_status = "invalid"
        else:
            calc_status = "ok"

        # Pack-year output logic
        if pd.isna(r['cpd']) or pd.isna(r['smoke_years']):
            py_str = "missing (defaulted to not severe)"
        else:
            try:
                py = (float(r['cpd']) / 20.0) * float(r['smoke_years'])
                # If classified as a non-smoker (both zero), output "0.0 (non-smoker)" for clarity
                py_str = "0.0 (non-smoker)" if py == 0 else f"{py:.1f}"
            except:
                py_str = "missing (defaulted to not severe)"

        history_text = str(r.get('history', '')).strip()
        cb_str = "missing (defaulted to none)" if history_text == "" else ("yes" if r['has_cb'] == 1 else "no")

        if calc_status == "ok":
            prob, py_calc = calculate_slim_risk_logic(r['fev_ratio'], r['bmi'], r['cpd'], r['smoke_years'], r['has_cb'])
            if py_str not in ["missing (defaulted to not severe)", "0.0 (non-smoker)"] and not pd.isna(py_calc):
                py_str = f"{py_calc:.1f}"

            prob_str = f"{prob:.2f}%"
            if prob >= SLIM_COEFS['threshold']:
                risk_str, code_str, advice_str = "high risk", "J44.5.3", "recommend annual lung-function follow-up and smoking cessation"
            else:
                risk_str, code_str, advice_str = "low risk", "J44.5.1", "routine_follow_up"

        elif calc_status == "missing":
            prob_str, risk_str, advice_str, code_str = "missing", "-", "insufficient data; cannot assess", "-"
        else:
            prob_str, risk_str, advice_str, code_str = "-", "not applicable (already diagnosed or age out of range)", "-", "-"

        results.append({
            'pat_id': r.get('pat_id', ''),
            'app_no': r.get('app_no', ''),
            'name': r.get('pat_name', 'unknown'),
            'sex': r.get('pat_sex', 'unknown'),
            'age': int(age) if not pd.isna(age) else "missing",
            'BMI': f"{r['bmi']:.1f}" if not pd.isna(r['bmi']) else "missing",
            'FEV1/FVC': f"{r['fev_ratio']:.3f}" if not pd.isna(r['fev_ratio']) else "missing",

            # --- Full display of smoking records ---
            'raw_smoking_text': r.get('raw_smoking_text', 'not_recorded'),
            'cumulative_pack_years': py_str,
            # ---------------------------

            'chronic_bronchitis': cb_str,
            'model_name': 'copd-SLIM prediction model',
            'code': code_str,
            'gauge_reference': GAUGE_REF,
            'prediction_horizon': 'next 6 years',
            'risk_probability': prob_str,
            'risk_level': risk_str,
            'clinical_recommendation': advice_str
        })

    if results:
        res_df = pd.DataFrame(results)
        cols = ['pat_id', 'app_no', 'name', 'model_name', 'code', 'gauge_reference', 'prediction_horizon',
                'risk_probability', 'risk_level', 'clinical_recommendation', 'sex', 'age', 'BMI', 'FEV1/FVC',
                'raw_smoking_text', 'cumulative_pack_years', 'chronic_bronchitis']
        final_cols = [c for c in cols if c in res_df.columns]
        res_df[final_cols].to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"--- Assessment complete! Results exported ---")


if __name__ == "__main__":
    process_data("base.csv", "item_result.csv", "slim_risk_smart_smoking.csv")