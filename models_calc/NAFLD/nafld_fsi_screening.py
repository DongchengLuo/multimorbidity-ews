import pandas as pd
import numpy as np
import re
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model_unit_utils import convert_measurement

# ==========================================
# 1. Field mapping (see the precise field-matching reference table)
# ==========================================
FIELD_MAPPING = {
    'age': ['app_pat_age', 'age'],
    'gender': ['pat_sex', 'sex'],
    'BMI': ['(body mass index) BMI', 'bmi', 'BMI'],
    'height': ['height'],
    'weight': ['weight'],
    'ALT': ['ALT'],
    'AST': ['AST'],
    'TG': ['triglyceride', 'TG'],
    'has_hypertension': ['past_history', 'past_medical_history'],
    'has_diabetes': ['past_history', 'past_medical_history']
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
            if normalize_str(p) in norm_name: return key
    return None


def clean_numeric(val):
    if pd.isna(val) or str(val).strip() == "": return np.nan
    try:
        match = re.search(r"[-+]?\d*\.\d+|\d+", str(val))
        return float(match.group()) if match else np.nan
    except:
        return np.nan


# ==========================================
# 3. FSI core scoring logic
# ==========================================
def calculate_fsi_score(age, is_female, bmi, tg, has_htn, has_dm, alt, ast):
    # Strict gate: refuse to compute when core biochemical or physical indicators are missing
    if any(pd.isna(x) for x in [age, bmi, tg, alt, ast]):
        return "-"

    # Avoid division by zero
    if ast == 0:
        ratio_score = 0
    else:
        ratio_score = 1.1 if (alt / ast) >= 1.33 else 0

    # Safe fallback for missing history (NaN or other values treated as False/0)
    htn_val = 1 if has_htn == 1 else 0
    dm_val = 1 if has_dm == 1 else 0

    linear_score = (
            -7.981 +
            (0.011 * age) -
            (0.146 * (1 if is_female else 0)) +
            (0.173 * bmi) +
            (0.007 * tg) +
            (0.593 * htn_val) +
            (0.789 * dm_val) +
            ratio_score
    )

    if linear_score >= 0:
        probability = 1 / (1 + np.exp(-linear_score))
    else:
        exp_score = np.exp(linear_score)
        probability = exp_score / (1 + exp_score)

    return probability * 100


# ==========================================
# 4. Main data-processing routine
# ==========================================
def process_data(base_path, item_path, output_path):
    model_name = "Framingham Steatosis Index (FSI)"
    print(f"--- Starting {model_name} assessment model (transparent output) ---")

    def read_csv_safe(path):
        if not os.path.exists(path): return pd.DataFrame()
        for enc in ['gbk', 'utf-8-sig', 'utf-8']:
            try:
                return pd.read_csv(path, encoding=enc, low_memory=False)
            except:
                continue
        return pd.DataFrame()

    base_df = read_csv_safe(base_path)
    item_df = read_csv_safe(item_path)

    if base_df.empty:
        print("base data file empty or unreadable!")
        return

    # Strip leading zeros that break joins
    base_df['app_no'] = base_df['app_no'].astype(str).str.strip().str.lstrip('0')
    base_df['pat_id'] = base_df['pat_id'].astype(str).str.strip().str.lstrip('0')

    # Pivot data to a wide table
    records = []
    if not item_df.empty:
        item_df['app_no'] = item_df['app_no'].astype(str).str.strip().str.lstrip('0')
        item_df['pat_id'] = item_df['pat_id'].astype(str).str.strip().str.lstrip('0')

        item_df['cat'] = item_df['res_itm_name'].apply(get_category)
        filtered = item_df.dropna(subset=['cat', 'res_value']).copy()

        # === Core change: upgraded history-detection logic ===
        def check_condition(text, keyword):
            if pd.isna(text) or str(text).strip() == "":
                return np.nan  # no record at all; return missing

            if keyword not in text:
                return np.nan  # other diseases recorded but not this one; return missing

            # When the disease is mentioned, check for a preceding negation (e.g. "denies hypertension")
            negations = ['absent', 'denies', 'not seen', 'exclude', 'no', 'not found', 'not_included']
            pattern = rf"({'|'.join(negations)})[^\u4e00-\u9fa5]{{0,3}}{keyword}"
            return 0 if re.search(pattern, text) else 1

        # ==================================

        def get_first_valid(val_list):
            for v in val_list:
                num = clean_numeric(v)
                if not pd.isna(num): return num
            return np.nan

        for (app_no, pat_id), group in filtered.groupby(['app_no', 'pat_id']):
            row = {'app_no': app_no, 'pat_id': pat_id}
            data_pool = {k: [] for k in FIELD_MAPPING.keys()}
            unit_pool = {k: [] for k in FIELD_MAPPING.keys()}

            for _, item in group.iterrows():
                data_pool[item['cat']].append(str(item['res_value']))
                unit_pool[item['cat']].append(item.get('res_unit', ''))

            def converted_first(cat, indicator, target_unit):
                values = data_pool.get(cat, [])
                units = unit_pool.get(cat, [])
                result = convert_measurement(
                    values[0] if values else None,
                    units[0] if units else '',
                    indicator=indicator,
                    target_unit=target_unit
                )
                return (result.value if result.ok else np.nan,
                        result.rule if result.ok else result.error)

            # FSI uses kg/m2; convert height/weight by res_unit only, no longer guessing cm or m from magnitude.
            row['bmi'], row['bmi_unit_note'] = converted_first('BMI', 'bmi', 'kg/m2')
            if pd.isna(row['bmi']):
                h, h_note = converted_first('height', 'height', 'm')
                w, w_note = converted_first('weight', 'weight', 'kg')
                if pd.notna(h) and pd.notna(w) and h > 0 and w > 0:
                    row['bmi'] = w / (h * h)
                    row['bmi_unit_note'] = f'computed from height [{h_note}], weight [{w_note}]'

            row['alt'], row['alt_unit_note'] = converted_first('ALT', 'liver_enzyme', 'U/L')
            row['ast'], row['ast_unit_note'] = converted_first('AST', 'liver_enzyme', 'U/L')
            tg_values = data_pool.get('TG', [])
            tg_units = unit_pool.get('TG', [])
            tg_result = convert_measurement(
                tg_values[0] if tg_values else None,
                tg_units[0] if tg_units else '',
                indicator='triglyceride',
                target_unit='mg/dL'
            )
            row['tg'] = tg_result.value if tg_result.ok else np.nan
            row['tg_unit_note'] = tg_result.rule if tg_result.ok else tg_result.error

            # Extract medical history
            hist_records = data_pool.get('has_hypertension', []) + data_pool.get('has_diabetes', [])
            hist_text = " ".join(hist_records) if hist_records else ""
            row['htn'] = check_condition(hist_text, "hypertension")
            row['dm'] = check_condition(hist_text, "diabetes")

            records.append(row)

    wide_items = pd.DataFrame(records) if records else pd.DataFrame(columns=['app_no', 'pat_id'])

    # Use a left join to keep all patients in the base table
    final = pd.merge(base_df, wide_items, on=['app_no', 'pat_id'], how='left')

    results = []
    GAUGE_REF = "-10-23|23-60"

    for _, r in final.iterrows():
        age = clean_numeric(r.get('app_pat_age'))
        sex_val = str(r.get('pat_sex', '')).strip()
        is_female = True if sex_val == '2' or 'female' in sex_val else False

        # Core scoring
        score = calculate_fsi_score(
            age, is_female, r.get('bmi'), r.get('tg'),
            r.get('htn'), r.get('dm'), r.get('alt'), r.get('ast')
        )

        # Result wrapping
        if score == "-":
            risk_level, risk_code, gauge = "-", "-", "-"
            remark = "core indicators missing or unit not convertible; cannot compute"
        else:
            gauge = GAUGE_REF
            remark = "data complete, computable"
            if score >= 23:
                risk_level, risk_code = "high risk", "K76.2.3"
            else:
                risk_level, risk_code = "low risk", "K76.2.1"

        # Format-output helper
        def fmt_in(v, is_num=False, is_bool=False):
            # If NaN (no record at all, or the disease is not mentioned), print "missing"
            if pd.isna(v) or v == "":
                return "missing (defaulted to none)" if is_bool else "missing"

            # If explicitly determined as present (1) or absent (0)
            if is_bool: return "present" if v == 1 else "absent"

            # If numeric
            return f"{v:.2f}" if is_num else str(v)

        results.append({
            'model_name': model_name,
            'pat_id': r.get('pat_id'),
            'app_no': r.get('app_no'),
            'pat_name': r.get('pat_name', 'unknown'),
            'sex': r.get('pat_sex', 'missing'),
            'age': int(age) if pd.notna(age) else "missing",

            # --- Raw indicator section ---
            'BMI': fmt_in(r.get('bmi'), True),
            'bmi_unit_handling': r.get('bmi_unit_note', 'missing res_unit'),
            'ALT': fmt_in(r.get('alt'), True),
            'alt_unit_handling': r.get('alt_unit_note', 'missing res_unit'),
            'AST': fmt_in(r.get('ast'), True),
            'ast_unit_handling': r.get('ast_unit_note', 'missing res_unit'),
            'TG(mg/dL)': fmt_in(r.get('tg'), True),
            'tg_unit_handling': r.get('tg_unit_note', 'missing res_unit'),
            'hypertension_history': fmt_in(r.get('htn'), False, True),  # output after strict evaluation
            'diabetes_history': fmt_in(r.get('dm'), False, True),  # output after strict evaluation

            # --- Model prediction results ---
            'predicted_score': f"{score:.2f}" if score != "-" else "-",
            'prediction_horizon': 'current' if score != "-" else "-",
            'risk_level': risk_level,
            'code': risk_code,
            'gauge_reference': gauge,
            'assessment_note': remark
        })

    if results:
        res_df = pd.DataFrame(results)

        # Enforce a column order for a clear table
        cols = ['pat_id', 'app_no', 'pat_name', 'model_name', 'code', 'gauge_reference',
                'predicted_score', 'risk_level', 'prediction_horizon', 'assessment_note',
                'sex', 'age', 'BMI', 'bmi_unit_handling', 'ALT', 'alt_unit_handling',
                'AST', 'ast_unit_handling', 'TG(mg/dL)', 'tg_unit_handling',
                'hypertension_history', 'diabetes_history']
        final_cols = [c for c in cols if c in res_df.columns]

        res_df[final_cols].to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"--- Assessment complete: {len(res_df)} results generated, exported to {output_path} ---")
    else:
        print("--- Assessment complete, no valid results ---")


if __name__ == "__main__":
    process_data("base.csv", "item_result.csv", "fsi_screening_result_perfect.csv")
