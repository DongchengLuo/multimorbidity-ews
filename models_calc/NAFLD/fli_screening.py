import pandas as pd
import numpy as np
import re

# ==========================================
# 1. Field mapping and model parameters
# ==========================================
FIELD_MAPPING = {
    'age': ['app_pat_age', 'age'],
    'gender': ['pat_sex', 'sex'],
    'BMI': ['(body mass index) BMI', 'BMI', 'bmi'],
    'wc': ['waist', 'abdominal_circumference', 'Waist Circumference'],
    'TG': ['triglyceride', 'TG', 'Triglycerides'],
    'GGT': ['GGT'],
    'height': ['height', 'Height'],
    'weight': ['weight', 'Weight'],
    'history': ['past_history', 'past_medical_history']  # added: exclude already-diseased population
}

MODEL_NAME = "Fatty Liver Index (FLI)"


# ==========================================
# 2. Basic cleaning and matching logic
# ==========================================
def normalize_str(s):
    if pd.isna(s): return ""
    return re.sub(r'[\(\)\（\）\[\]\【\】\{\}\s\-\－]', '', str(s)).lower()


def get_category(name):
    norm_name = normalize_str(name)
    for key, patterns in FIELD_MAPPING.items():
        for p in patterns:
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


def is_already_fatty_liver(history_str):
    """Exclusion logic: check whether fatty liver is already in the past history"""
    if pd.isna(history_str): return False
    s = str(history_str)
    if 'fatty_liver' in s:
        negations = ['denies', 'absent', 'not found', 'exclude', 'normal', 'unknown']
        pos = s.find('fatty_liver')
        prefix = s[max(0, pos - 5):pos]
        if any(neg in prefix for neg in negations): return False
        return True
    return False


# ==========================================
# 3. FLI core scoring logic
# ==========================================
def calculate_fli_score(bmi, waist, tg, ggt):
    # Strict gate: return "-" when any item is missing
    if any(pd.isna(x) for x in [bmi, waist, tg, ggt]):
        return "-"

    # Check numeric validity (log inputs must be > 0)
    if tg <= 0 or ggt <= 0:
        return "-"

    try:
        beta_tg = 0.953
        beta_bmi = 0.139
        beta_ggt = 0.718
        beta_waist = 0.053
        constant = -15.745

        l_score = (beta_tg * np.log(tg)) + (beta_bmi * bmi) + (beta_ggt * np.log(ggt)) + (beta_waist * waist) + constant
        fli = (np.exp(l_score) / (1 + np.exp(l_score))) * 100
        return fli
    except:
        return "-"


# ==========================================
# 4. Main data-processing routine
# ==========================================
def process_data(base_path, item_path, output_path):
    print(f"--- Starting {MODEL_NAME} computation (transparent, strict) ---")

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

    # [Core fix]: strip leading zeros that break joins
    base_df['app_no'] = base_df['app_no'].astype(str).str.strip().str.lstrip('0')
    base_df['pat_id'] = base_df['pat_id'].astype(str).str.strip().str.lstrip('0')

    # 1. Wide-table extraction
    records = []
    if not item_df.empty:
        item_df['app_no'] = item_df['app_no'].astype(str).str.strip().str.lstrip('0')
        item_df['pat_id'] = item_df['pat_id'].astype(str).str.strip().str.lstrip('0')

        item_df['cat'] = item_df['res_itm_name'].apply(get_category)
        filtered = item_df.dropna(subset=['cat', 'res_value']).copy()

        for (app_no, pat_id), group in filtered.groupby(['app_no', 'pat_id']):
            row = {'app_no': app_no, 'pat_id': pat_id}
            data_pool = {k: [] for k in FIELD_MAPPING.keys()}

            for _, item in group.iterrows():
                data_pool[item['cat']].append(str(item['res_value']))

            # Take the first valid number
            def get_first_num(lst):
                nums = [clean_numeric(v) for v in lst if pd.notna(clean_numeric(v))]
                return nums[0] if nums else np.nan

            for k in ['BMI', 'wc', 'TG', 'GGT', 'height', 'weight']:
                row[k] = get_first_num(data_pool.get(k, []))

            # Extract past history
            row['history'] = " ".join(set(data_pool['history'])) if data_pool['history'] else ""
            records.append(row)

    wide_items = pd.DataFrame(records) if records else pd.DataFrame(columns=['app_no', 'pat_id'])

    # 2. Merge basic information (left join, keep everyone)
    final = pd.merge(base_df, wide_items, on=['app_no', 'pat_id'], how='left')

    # 3. Indicator post-processing (BMI fallback & TG unit conversion)
    def post_process_row(r):
        # BMI fallback
        bmi_final = r.get('BMI', np.nan)
        if pd.isna(bmi_final):
            h = r.get('height', np.nan)
            w = r.get('weight', np.nan)
            if pd.notna(h) and pd.notna(w) and h > 0 and w > 0:
                h_m = h / 100.0 if h > 3 else h
                bmi_final = w / (h_m * h_m)

        # TG unit conversion (mmol/L -> mg/dL)
        tg_final = r.get('TG', np.nan)
        if pd.notna(tg_final):
            if tg_final < 15:  # treat as mmol/L; multiply by 88.57
                tg_final = tg_final * 88.57

        return pd.Series([bmi_final, tg_final])

    final[['bmi_final', 'tg_final']] = final.apply(post_process_row, axis=1)

    # 4. Scoring and result generation
    results = []
    GAUGE_REF = "0-30|30-60|60-100"

    for _, r in final.iterrows():
        # A. Check whether the model applies
        exclusion_reason = ""
        if is_already_fatty_liver(r.get('history')):
            exclusion_reason = "excluded: existing fatty liver"
            score = "-"
        else:
            score = calculate_fli_score(
                r.get('bmi_final'), r.get('wc'), r.get('tg_final'), r.get('GGT')
            )

        # B. Risk assessment
        if score == "-":
            risk_level = "-"
            risk_code = "-"
            suggestion = "-"
            gauge = "-"
        else:
            gauge = GAUGE_REF
            if score < 30:
                risk_level, risk_code, suggestion = "low risk", "K76.3.1", "current risk is low; maintain a healthy lifestyle"
            elif score >= 60:
                risk_level, risk_code, suggestion = "high risk", "K76.3.3", "liver ultrasound and physician consultation advised"
            else:
                risk_level, risk_code, suggestion = "medium risk", "K76.3.2", "further clinical examination advised"

        # C. Output formatting
        def fmt_in(v, is_num=False):
            if pd.isna(v) or v == "": return "missing"
            return f"{v:.2f}" if is_num else str(v)

        age = clean_numeric(r.get('app_pat_age'))

        results.append({
            'model_name': MODEL_NAME,
            'pat_id': r.get('pat_id'),
            'app_no': r.get('app_no'),
            'pat_name': r.get('pat_name', 'unknown'),
            'sex': r.get('pat_sex', 'missing'),
            'age': int(age) if pd.notna(age) else "missing",

            # --- Raw indicator section (missing shown as "missing") ---
            'BMI': fmt_in(r['bmi_final'], True),
            'waist (cm)': fmt_in(r.get('wc'), True),
            'triglyceride (mg/dL)': fmt_in(r['tg_final'], True),
            'GGT(U/L)': fmt_in(r.get('GGT'), True),

            # --- Model predictions (missing or excluded shown as "-") ---
            'predicted_score': f"{score:.2f}" if score != "-" else "-",
            'prediction_horizon': 'current' if score != "-" else "-",
            'risk_level': risk_level,
            'code': risk_code,
            'gauge_reference': gauge,
            'recommendation': suggestion,
            'assessment_note': exclusion_reason if exclusion_reason else (
                "data complete, computable" if score != "-" else "indicators missing; cannot compute")
        })

    if results:
        res_df = pd.DataFrame(results)
        res_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"--- Assessment complete: {len(res_df)} results generated, exported to {output_path} ---")
    else:
        print("--- Assessment complete, but no valid results were generated ---")


if __name__ == "__main__":
    process_data(
        "base.csv",
        "item_result.csv",
        "fli_screening_result_perfect.csv"
    )