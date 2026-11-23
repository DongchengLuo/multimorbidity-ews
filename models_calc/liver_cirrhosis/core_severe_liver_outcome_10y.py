import pandas as pd
import numpy as np
import re
import math
import os

# ==========================================
# 1. Field mapping (for your data headers)
# ==========================================
FIELD_MAPPING = {
    'age': ['app_pat_age', 'age'],
    'gender': ['pat_sex', 'sex'],
    'AST': ['AST'],
    'ALT': ['ALT'],
    'GGT': ['GGT', 'ggt']
}


# ==========================================
# 2. Core algorithm: CORE risk score (strictly aligned with the BMJ 2025 formula)
# ==========================================
def calculate_core_score(age, sex_str, ast_ul, alt_ul, ggt_ul):
    if ast_ul > 600 or alt_ul > 600 or ggt_ul > 1020:
        return np.nan, {"error": "indicators outside the model range"}

    def to_ln_ukat(v):
        val_ukat = max(v / 60.0, 0.01)
        return math.log(val_ukat)

    ln_ast = to_ln_ukat(ast_ul)
    ln_alt = to_ln_ukat(alt_ul)
    ln_ggt = to_ln_ukat(ggt_ul)
    is_male = 1 if str(sex_str).strip() in ['male', 'Male', 'M', '1'] else 0

    def p3(val):
        return max(0, val) ** 3

    xb = -13.0001 + 0.1130476 * age
    xb -= 0.0001670826 * p3(age - 30.65)
    xb += 0.0003334586 * p3(age - 39.27)
    xb -= 0.0001490879 * p3(age - 46.68)
    xb -= 1.425762e-05 * p3(age - 56.16)
    xb -= 3.030396e-06 * p3(age - 70.52)

    xb -= 0.9665283 * ln_ggt
    xb += 2.042459 * p3(ln_ggt + 1.97)
    xb -= 6.212038 * p3(ln_ggt + 1.39)
    xb += 4.895392 * p3(ln_ggt + 0.94)
    xb -= 0.8183846 * p3(ln_ggt - 0.17)
    xb += 0.0925717 * p3(ln_ggt - 1.61)

    xb -= 0.6254278 * ln_ast
    xb += 0.9052493 * p3(ln_ast + 1.56)
    xb += 3.456491 * p3(ln_ast + 1.20)
    xb -= 7.848813 * p3(ln_ast + 0.99)
    xb += 4.072892 * p3(ln_ast + 0.46)
    xb -= 0.5858198 * p3(ln_ast - 0.59)

    xb -= 0.5258893 * ln_alt
    xb += 0.158157 * p3(ln_alt + 1.90)
    xb -= 0.5817669 * p3(ln_alt + 1.27)
    xb += 0.2363219 * p3(ln_alt + 0.84)
    xb += 0.4381633 * p3(ln_alt - 0.01)
    xb -= 0.2508752 * p3(ln_alt - 0.98)

    if is_male:
        m_xb = 0.6098087 - 0.04002686 * age
        m_xb += 0.0001498023 * p3(age - 30.65)
        m_xb -= 0.0002338722 * p3(age - 39.27)
        m_xb += 8.857314e-05 * p3(age - 46.68)
        m_xb += 0.0002400746 * p3(age - 56.16)
        m_xb -= 6.743156e-05 * p3(age - 70.52)

        m_xb -= 0.2622872 * ln_ggt
        m_xb += 0.0566088 * p3(ln_ggt + 1.97)
        m_xb += 0.1045051 * p3(ln_ggt + 1.39)
        m_xb -= 0.2941066 * p3(ln_ggt + 0.94)
        m_xb += 0.1626142 * p3(ln_ggt - 0.17)
        m_xb -= 0.02962157 * p3(ln_ggt - 1.61)

        m_xb -= 0.2470467 * ln_ast
        m_xb -= 1.211869 * p3(ln_ast + 1.56)
        m_xb += 5.617974 * p3(ln_ast + 1.20)
        m_xb -= 5.453922 * p3(ln_ast + 0.99)
        m_xb += 1.1022 * p3(ln_ast + 0.46)
        m_xb -= 0.06240334 * p3(ln_ast - 0.59)

        m_xb -= 0.03140536 * ln_alt
        m_xb -= 0.1253678 * p3(ln_alt + 1.90)
        m_xb += 0.6473214 * p3(ln_alt + 1.27)
        m_xb -= 0.6380499 * p3(ln_alt + 0.84)
        m_xb += 0.06804696 * p3(ln_alt - 0.01)
        m_xb += 0.04804941 * p3(ln_alt - 0.98)
        xb += m_xb

    age_ggt = -0.005854575 * p3(ln_ggt + 1.97)
    age_ggt += 0.01532457 * p3(ln_ggt + 1.39)
    age_ggt -= 0.01369761 * p3(ln_ggt + 0.94)
    age_ggt += 0.006903603 * p3(ln_ggt - 0.17)
    age_ggt -= 0.002675983 * p3(ln_ggt - 1.61)
    xb += age * age_ggt

    ggt_block = 0.0152652 * age - 4.708568e-05 * p3(age - 30.65) + 0.0001232485 * p3(age - 39.27) \
                - 0.0001101636 * p3(age - 46.68) + 4.541027e-05 * p3(age - 56.16) - 1.140946e-05 * p3(age - 70.52)
    ggt_block += -0.8635879 * ln_ast + 0.5615145 * p3(ln_ast + 1.56) + 0.947918 * p3(ln_ast + 1.20) \
                 - 2.820256 * p3(ln_ast + 0.99) + 1.476022 * p3(ln_ast + 0.46) - 0.1681977 * p3(ln_ast - 0.59)
    ggt_block += 0.01375014 * ln_alt - 0.1003385 * p3(ln_alt + 1.90) - 0.3823287 * p3(ln_alt + 1.27) \
                 + 0.9263131 * p3(ln_alt + 0.84) - 0.5563601 * p3(ln_alt - 0.01) + 0.1127141 * p3(ln_alt - 0.98)
    xb += ln_ggt * ggt_block

    ast_block = 0.2048662 * p3(ln_ggt + 1.97) + 0.344006 * p3(ln_ggt + 1.39) - 1.023491 * p3(ln_ggt + 0.94) \
                + 0.5872282 * p3(ln_ggt - 0.17) - 0.1126098 * p3(ln_ggt - 1.61)
    ast_block += -0.03154256 * ln_alt + 0.04201595 * p3(ln_alt + 1.90) + 0.03201952 * p3(ln_alt + 1.27) \
                 - 0.5571034 * p3(ln_alt + 0.84) + 0.8491616 * p3(ln_alt - 0.01) - 0.3660936 * p3(ln_alt - 0.98)
    xb += ln_ast * ast_block

    alt_block = -0.06489212 * p3(ln_ggt + 1.97) - 0.2472643 * p3(ln_ggt + 1.39) + 0.5990766 * p3(ln_ggt + 0.94) \
                - 0.3847223 * p3(ln_ggt - 0.17) + 0.0978021 * p3(ln_ggt - 1.61)
    alt_block += 0.07487616 * p3(ln_ast + 1.56) + 0.05706162 * p3(ln_ast + 1.20) - 0.9928076 * p3(ln_ast + 0.99) \
                 + 1.243481 * p3(ln_ast + 0.46) - 0.3826107 * p3(ln_ast - 0.59)
    xb += ln_alt * alt_block

    risk = 1.0 / (1.0 + math.exp(-xb))
    return risk, {"linear_predictor": xb}


# ==========================================
# 3. Helper cleaning functions
# ==========================================
def normalize_column_name(name):
    if pd.isna(name): return ""
    return re.sub(r'[\(\)\（\）\[\]\【\】\{\}\s\|\-]', '', str(name)).lower()


def clean_numeric_value(val):
    if pd.isna(val) or str(val).strip() == "": return np.nan
    try:
        match = re.search(r"[-+]?\d*\.\d+|\d+", str(val))
        return float(match.group()) if match else np.nan
    except:
        return np.nan


# ==========================================
# 4. Main data-processing routine
# ==========================================
def run_core_prediction(base_csv_path, item_csv_path, output_csv_path):
    print("--- Starting CORE liver risk assessment (full base iteration) ---")

    GAUGE_REF = "0-0.4|0.4-5.0|5.0-100"

    def read_csv_safe(path):
        for enc in ['gbk', 'utf-8-sig', 'utf-8']:
            try:
                return pd.read_csv(path, encoding=enc, low_memory=False)
            except:
                continue
        return pd.DataFrame()

    base_df = read_csv_safe(base_csv_path)
    item_df = read_csv_safe(item_csv_path)

    if base_df.empty:
        print("Error: base table could not be read or is empty.")
        return

    base_df['app_no'] = base_df['app_no'].astype(str).str.strip()
    base_df['pat_id'] = base_df['pat_id'].astype(str).str.strip()

    # Preprocess item data
    def map_category(name):
        norm = normalize_column_name(name)
        for cat, patterns in FIELD_MAPPING.items():
            if cat in ['age', 'gender']: continue
            for p in patterns:
                if normalize_column_name(p) in norm:
                    return cat
        return None

    item_indexed = {}  # {(app_no, pat_id): {cat: [values]}}
    if not item_df.empty:
        item_df['app_no'] = item_df['app_no'].astype(str).str.strip()
        item_df['pat_id'] = item_df['pat_id'].astype(str).str.strip()
        item_df['category'] = item_df['res_itm_name'].apply(map_category)
        valid_items = item_df.dropna(subset=['category']).copy()
        valid_items['numeric_val'] = valid_items['res_value'].apply(clean_numeric_value)

        for _, row in valid_items.iterrows():
            key = (str(row['app_no']).strip(), str(row['pat_id']).strip())
            if key not in item_indexed:
                item_indexed[key] = {}
            cat = row['category']
            if cat not in item_indexed[key]:
                item_indexed[key][cat] = []
            if not pd.isna(row['numeric_val']):
                item_indexed[key][cat].append(row['numeric_val'])

    # Iterate over each patient in base_df
    results = []
    for _, base_row in base_df.iterrows():
        app_no = str(base_row['app_no']).strip()
        pat_id = str(base_row['pat_id']).strip()

        age = clean_numeric_value(base_row.get('app_pat_age'))
        sex = str(base_row.get('pat_sex', ''))

        # Fetch indicators from item
        key = (app_no, pat_id)
        patient_data = item_indexed.get(key, {})
        ast = np.nanmean(patient_data.get('AST', [np.nan])) if patient_data.get('AST') else np.nan
        alt = np.nanmean(patient_data.get('ALT', [np.nan])) if patient_data.get('ALT') else np.nan
        ggt = np.nanmean(patient_data.get('GGT', [np.nan])) if patient_data.get('GGT') else np.nan

        # Attempt computation
        if pd.isna(age) or pd.isna(ast) or pd.isna(alt) or pd.isna(ggt):
            results.append({
                'pat_id': pat_id,
                'app_no': app_no,
                'pat_name': base_row.get('pat_name', 'unknown'),
                'sex': sex,
                'age': '-',
                'AST': '-',
                'ALT': '-',
                'GGT': '-',
                'risk_probability': '-',
                'risk_level': '-',
                'prediction_horizon': "next 10 years",
                'code': '-',
                'gauge_reference': GAUGE_REF,
                'recommendation': '-'
            })
            continue

        risk, debug = calculate_core_score(age, sex, ast, alt, ggt)
        if pd.isna(risk):
            results.append({
                'pat_id': pat_id,
                'app_no': app_no,
                'pat_name': base_row.get('pat_name', 'unknown'),
                'sex': sex,
                'age': age,
                'AST': ast,
                'ALT': alt,
                'GGT': ggt,
                'risk_probability': '-',
                'risk_level': '-',
                'prediction_horizon': "next 10 years",
                'code': '-',
                'gauge_reference': GAUGE_REF,
                'recommendation': '-'
            })
            continue

        risk_pct = risk * 100
        if risk_pct < 0.4:
            level, code, sug = "low risk", "K74.4.1", "current risk is low; maintain a healthy lifestyle"
        elif risk_pct >= 5.0:
            level, code, sug = "high risk", "K74.4.3", "specialist evaluation advised (e.g. liver stiffness scan)"
        else:
            level, code, sug = "medium risk", "K74.4.2", "regular follow-up; monitor liver function"

        results.append({
            'pat_id': pat_id,
            'app_no': app_no,
            'pat_name': base_row.get('pat_name', 'unknown'),
            'sex': sex,
            'age': age,
            'AST': ast,
            'ALT': alt,
            'GGT': ggt,
            'risk_probability': f"{risk_pct:.2f}%",
            'risk_level': level,
            'prediction_horizon': "next 10 years",
            'code': code,
            'gauge_reference': GAUGE_REF,
            'recommendation': sug
        })

    if results:
        res_df = pd.DataFrame(results)
        res_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
        valid_count = len(res_df[res_df['code'] != '-'])
        print(f"--- Assessment complete: {len(res_df)} records processed ---")
        print(f"--- Valid computed records: {valid_count} ---")
        print(f"--- Records not computable due to missing values: {len(res_df) - valid_count} ---")
        print(f"--- Results saved to: {output_csv_path} ---")


if __name__ == "__main__":
    run_core_prediction("base.csv", "item_result.csv", "core_liver_risk_result.csv")
