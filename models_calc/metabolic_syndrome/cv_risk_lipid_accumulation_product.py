import pandas as pd
import numpy as np
import re
from model_unit_utils import convert_measurement

# ==========================================
# 1. Field mapping (see the precise field-matching reference table)
# ==========================================
FIELD_MAPPING = {
    'gender': ['pat_sex', 'sex'],
    'wc': ['waist', 'wc', 'waist_circumference', 'abdominal_circumference'],
    'TG': ['triglyceride', 'TG']
}

MODEL_NAME = "Lipid Accumulation Product (LAP)"


# ==========================================
# 2. Basic cleaning utilities
# ==========================================
def normalize_str(s):
    if pd.isna(s): return ""
    return re.sub(r'[\(\)\（\）\[\]\【\】\{\}\s]', '', str(s)).lower()


def get_category(name):
    norm_name = normalize_str(name)
    for key, patterns in FIELD_MAPPING.items():
        for p in patterns:
            # Replace in with == for exact matching after normalization
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


# ==========================================
# 3. LAP core computation logic
# ==========================================
def calculate_lap_score(gender, wc_cm, tg_mmol):
    """Compute the continuous LAP indicator per the source formula; inputs must already be in cm and mmol/L."""
    if pd.isna(wc_cm) or pd.isna(tg_mmol) or pd.isna(gender) or str(gender).strip() == "" or str(gender).strip() == "missing":
        return "-", "-", "-", "-", "indicator_missing"

    sex = str(gender).strip().lower()
    if sex in ['male', '1']:
        k = 65
    elif sex in ['female', '2']:
        k = 58
    else:
        # Block computation when sex cannot be determined
        return "-", "-", "-", "-", "unknown sex; cannot match formula"

    # The original treats male waist <=65 cm as 66 cm, equivalent to a difference floor of 1;
    # The minimum waist in the female sample is 58.4 cm; keep only the continuous indicators defined in the source, with no self-set thresholds.
    difference = max(1.0, float(wc_cm) - k) if sex in ['male', '1'] else float(wc_cm) - k
    score = difference * float(tg_mmol)
    return round(score, 2), "continuous_LAP_indicator", "-", "-", ""


# ==========================================
# 4. Main data-processing routine
# ==========================================
def process_data(base_path, item_path, output_path):
    print(f"--- Starting {MODEL_NAME} assessment (transparent full version) ---")

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
        print("error: cannot read base file or it is empty.")
        return

    # Trim spaces only; keep the original leading zeros!
    base_df['app_no'] = base_df['app_no'].astype(str).str.strip()
    base_df['pat_id'] = base_df['pat_id'].astype(str).str.strip()

    # 1. Pivot to wide table
    records = []
    if not item_df.empty:
        item_df['app_no'] = item_df['app_no'].astype(str).str.strip()
        item_df['pat_id'] = item_df['pat_id'].astype(str).str.strip()

        item_df['cat'] = item_df['res_itm_name'].apply(get_category)
        filtered = item_df.dropna(subset=['cat', 'res_value']).copy()

        # Helper: extract the first valid value in a list
        def get_first_valid(val_list, is_num=True):
            for v in val_list:
                if is_num:
                    num = clean_numeric(v)
                    if not pd.isna(num): return num
                else:
                    if pd.notna(v) and str(v).strip() != "": return str(v).strip()
            return np.nan

        for (app_no, pat_id), group in filtered.groupby(['app_no', 'pat_id']):
            row = {'app_no': app_no, 'pat_id': pat_id}
            def first_item(category):
                matches = group[group['cat'] == category]
                return matches.iloc[0] if not matches.empty else None

            wc_item, tg_item, gender_item = first_item('wc'), first_item('TG'), first_item('gender')
            row['wc_raw'] = wc_item.get('res_value') if wc_item is not None else np.nan
            row['wc_unit'] = wc_item.get('res_unit', '') if wc_item is not None else ''
            row['tg_raw'] = tg_item.get('res_value') if tg_item is not None else np.nan
            row['tg_unit'] = tg_item.get('res_unit', '') if tg_item is not None else ''
            row['gender_item'] = gender_item.get('res_value') if gender_item is not None else np.nan

            wc_result = convert_measurement(row['wc_raw'], row['wc_unit'], indicator='height', target_unit='cm')
            tg_result = convert_measurement(row['tg_raw'], row['tg_unit'], indicator='triglyceride', target_unit='mmol/L')
            row['wc'] = wc_result.value if wc_result.ok else np.nan
            row['tg'] = tg_result.value if tg_result.ok else np.nan
            row['unit_note'] = f"waist [{wc_result.rule or wc_result.error}]; triglyceride [{tg_result.rule or tg_result.error}]"

            records.append(row)

    wide_items = pd.DataFrame(records) if records else pd.DataFrame(columns=['app_no', 'pat_id'])

    # 2. Merge basic information (left join, keep all)
    final = pd.merge(base_df, wide_items, on=['app_no', 'pat_id'], how='left')

    # 3. Compute scores and assemble results
    results = []

    # Format-output function
    def fmt_in(v, is_num=False):
        if pd.isna(v) or v == "": return "missing"
        return f"{v:.2f}" if is_num else str(v)

    for _, r in final.iterrows():
        # Prefer sex from the base table, then the item table
        gender = r.get('pat_sex') if pd.notna(r.get('pat_sex')) and str(r.get('pat_sex')).strip() != "" else r.get(
            'gender_item')

        score, risk_level, risk_code, gauge, conclusion = calculate_lap_score(
            gender=gender,
            wc_cm=r.get('wc'),
            tg_mmol=r.get('tg')
        )

        # Assessment notes
        if score == "-":
            remark = conclusion if "unknown_sex" in conclusion else "core indicators (sex/waist/triglyceride) missing; cannot assess"
            conclusion = "-"  # no conclusion when it cannot be computed
        else:
            remark = "data complete, computable"

        # Result assembly
        results.append({
            'pat_id': r.get('pat_id'),
            'app_no': r.get('app_no'),
            'pat_name': r.get('pat_name', 'unknown'),
            'model_name': MODEL_NAME,

            # --- Transparent output of raw feature inputs ---
            'sex': gender if pd.notna(gender) else "missing",
            'waist_raw_value': r.get('wc_raw', 'missing'),
            'waist_raw_unit': r.get('wc_unit', ''),
            'waist (cm)': fmt_in(r.get('wc'), True),
            'triglyceride_raw_value': r.get('tg_raw', 'missing'),
            'triglyceride_raw_unit': r.get('tg_unit', ''),
            'triglyceride (mmol/L)': fmt_in(r.get('tg'), True),
            'unit handling': r.get('unit_note', '-'),

            # --- Model output results ---
            'code': risk_code,
            'gauge_reference': gauge,
            'predicted_score': score,
            'risk_level': risk_level,
            'prediction_horizon': 'current' if score != "-" else "-",
            'assessment_note': remark
        })

    # 4. Export
    if results:
        res_df = pd.DataFrame(results)
        # Removed the trailing reordering; export in the original dict key order
        res_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"--- Assessment complete: {len(res_df)} results generated, exported to {output_path} ---")
    else:
        print("no valid assessment results generated.")


if __name__ == "__main__":
    process_data(
        base_path="base.csv",
        item_path="item_result.csv",
        output_path="lap_risk_assessment_result_perfect.csv"
    )
