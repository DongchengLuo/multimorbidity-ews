import pandas as pd
import numpy as np
import re

# ==========================================
# 1. Field mapping (ZJU Index indicators)
# ==========================================
FIELD_MAPPING = {
    'BMI': ['(body mass index) BMI', 'BMI', 'bmi'],
    'height': ['height'],
    'weight': ['weight'],
    'FBG': ['GLU', 'FBG'],
    'TG': ['triglyceride', 'TG'],
    'ALT': ['ALT'],
    'AST': ['AST'],
    'gender': ['pat_sex', 'sex']
}

GAUGE_REF_STR = "0-32|32-38|38-100"


# ==========================================
# 2. Basic cleaning and utility functions
# ==========================================
def normalize_str(s):
    if pd.isna(s): return ""
    return re.sub(r'[\(\)\（\）\[\]\【\】\{\}\s]', '', str(s)).lower()


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


def parse_gender_to_code(val):
    """ZJU convention: 1=male, 2=female"""
    s = str(val).strip()
    if s in ['1', 'male', 'Male', 'M']:
        return 1
    elif s in ['2', 'female', 'Female', 'F']:
        return 2
    return None


def convert_unit(value, unit_str, category):
    """
    Automatic unit conversion (with blind-guess correction)
    """
    if pd.isna(value):
        return value

    unit_norm = str(unit_str).strip().lower() if pd.notna(unit_str) else ""

    # Strategy 1: an explicit string unit is present
    if 'mg/dl' in unit_norm or 'mg/100ml' in unit_norm:
        if category == 'FBG':
            return value / 18.0
        elif category == 'TG':
            return value / 88.5

    # Strategy 2: when the unit is empty, invalid or unrecognized, infer plausibility from physiological extremes and magnitude
    elif not unit_norm or unit_norm in ['nan', 'none', 'null', '']:
        if category == 'FBG':
            # FBG in mmol/L rarely exceeds 30; above 35 it is almost certainly mg/dL.
            if value > 35:
                return value / 18.0
        elif category == 'TG':
            # TG above 20 mmol/L is extremely rare (severe lipemia); above 25 it is almost certainly mg/dL.
            if value > 25:
                return value / 88.5

    # Otherwise (e.g. unit already mmol/L, or ALT/AST values) keep as-is
    return value


# ==========================================
# 3. ZJU Index core computation logic
# ==========================================
def calculate_zju_index_score(bmi, fbg, tg, alt, ast, gender_code):
    ratio = np.nan
    if pd.notna(alt) and pd.notna(ast) and ast != 0:
        ratio = alt / ast

    # Completeness check: block when any item is missing
    if any(pd.isna(x) for x in [bmi, fbg, tg, alt, ast, gender_code]) or ast == 0:
        return "-", ratio, "-", "-", "-"

    try:
        score = bmi + fbg + tg + (3 * ratio)
        if gender_code == 2:
            score += 2

        if score < 32.0:
            risk_level, code, suggestion = "low risk", "K76.1.1", "NAFLD excluded (92.2% sensitivity)"
        elif score > 38.0:
            risk_level, code, suggestion = "high risk", "K76.1.3", "high NAFLD risk; imaging confirmation strongly advised"
        else:
            risk_level, code, suggestion = "medium risk", "K76.1.2", "gray zone; correlate with symptoms or recheck"

        return score, ratio, risk_level, suggestion, code

    except Exception:
        return "-", ratio, "-", "-", "-"


# ==========================================
# 4. Main data-processing routine
# ==========================================
def process_data(base_path, item_path, output_path):
    print("--- Starting ZJU Index (NAFLD) model (adaptive unit inference) ---")

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
        print("Base data file could not be read or is empty")
        return

    # Strip leading zeros
    base_df['app_no'] = base_df['app_no'].astype(str).str.strip().str.lstrip('0')
    base_df['pat_id'] = base_df['pat_id'].astype(str).str.strip().str.lstrip('0')

    # 1. Pivot to wide table
    print("Aggregating exam indicators...")
    records = []
    if not item_df.empty:
        item_df['app_no'] = item_df['app_no'].astype(str).str.strip().str.lstrip('0')
        item_df['pat_id'] = item_df['pat_id'].astype(str).str.strip().str.lstrip('0')

        item_df['cat'] = item_df['res_itm_name'].apply(get_category)
        filtered = item_df.dropna(subset=['cat', 'res_value']).copy()

        for (app_no, pat_id), group in filtered.groupby(['app_no', 'pat_id']):
            row = {'app_no': app_no, 'pat_id': pat_id}
            data_pool = {k: np.nan for k in FIELD_MAPPING.keys()}

            for _, item in group.iterrows():
                # Take the first valid record
                if pd.isna(data_pool[item['cat']]):
                    if item['cat'] == 'gender':
                        data_pool[item['cat']] = item['res_value']
                    else:
                        raw_val = clean_numeric(item['res_value'])

                        # --- Adaptive unit conversion logic ---
                        # Try to read the unit column; pass np.nan for blind guessing when the source has none
                        unit_val = item.get('res_unit', np.nan)
                        if pd.notna(raw_val):
                            raw_val = convert_unit(raw_val, unit_val, item['cat'])
                        # -------------------------

                        data_pool[item['cat']] = raw_val

            # Basic indicator extraction
            for k in ['BMI', 'height', 'weight', 'FBG', 'TG', 'ALT', 'AST']:
                row[k] = data_pool.get(k)
            row['item_sex'] = data_pool.get('gender')

            # BMI derived-computation fallback
            if pd.isna(row['BMI']) and pd.notna(row['height']) and pd.notna(row['weight']) and row['height'] > 0:
                h = row['height'] / 100.0 if row['height'] > 3 else row['height']
                row['BMI'] = row['weight'] / (h ** 2)

            records.append(row)

    wide_items = pd.DataFrame(records) if records else pd.DataFrame(columns=['app_no', 'pat_id'])

    # 2. Merge basic information (left join, keep all patients)
    final = pd.merge(base_df, wide_items, on=['app_no', 'pat_id'], how='left')

    # 3. Scoring and result generation
    results = []
    print("Computing risk scores...")

    # Helper to format output consistently
    def fmt_in(v, is_num=False):
        if pd.isna(v) or v is None or str(v).strip() == "": return "missing"
        return f"{v:.2f}" if is_num else str(v)

    for _, r in final.iterrows():
        raw_sex = r.get('pat_sex') if pd.notna(r.get('pat_sex')) and str(r.get('pat_sex')).strip() != "" else r.get(
            'item_sex')
        gender_code = parse_gender_to_code(raw_sex)

        bmi = r.get('BMI')
        fbg = r.get('FBG')
        tg = r.get('TG')
        alt = r.get('ALT')
        ast = r.get('AST')

        score, ratio, risk_level, suggestion, code = calculate_zju_index_score(
            bmi, fbg, tg, alt, ast, gender_code
        )

        # Assessment-notes logic
        if score == "-":
            remark = "core indicators missing; cannot compute"
        else:
            remark = "data complete, computable"

        results.append({
            'pat_id': r.get('pat_id'),
            'app_no': r.get('app_no'),
            'pat_name': r.get('pat_name', 'unknown'),
            'model_name': 'ZJU Index (NAFLD)',

            # --- Raw feature input data ---
            'sex': raw_sex if pd.notna(raw_sex) else "missing",
            'height (cm)': fmt_in(r.get('height'), True),
            'weight (kg)': fmt_in(r.get('weight'), True),
            'BMI': fmt_in(bmi, True),
            'FBG': fmt_in(fbg, True),
            'TG': fmt_in(tg, True),
            'ALT': fmt_in(alt, True),
            'AST': fmt_in(ast, True),

            # --- Intermediate derived indicators ---
            'ALT/AST_ratio': fmt_in(ratio, True),

            # --- Model output results ---
            'predicted_score': f"{score:.2f}" if score != "-" else "-",
            'risk_level': risk_level,
            'prediction_horizon': 'current',
            'code': code,
            'gauge_reference': GAUGE_REF_STR if score != "-" else "-",
            'recommendation': suggestion,
            'assessment_note': remark
        })

    # 4. Export
    if results:
        res_df = pd.DataFrame(results)

        # Output the result dict's default column order directly
        res_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"--- Assessment complete, exported to {output_path} ---")
        print(f"total exported: {len(res_df)} (including patients with missing indicators)")
    else:
        print("--- WARNING: no valid results ---")


if __name__ == "__main__":
    process_data("base.csv", "item_result.csv", "zju_nafld_result_perfect.csv")