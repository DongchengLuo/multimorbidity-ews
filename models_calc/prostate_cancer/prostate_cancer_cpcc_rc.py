import pandas as pd
import numpy as np
import re
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model_unit_utils import convert_measurement, normalize_unit

# ==========================================
# 1. Field mapping (see the precise field-matching reference table)
# ==========================================
FIELD_MAPPING = {
    'age': ['app_pat_age'],
    'psa_value': ['PSA', 'total_PSA', 'TPSA', 'tPSA'],
    'fPSA/tPSA': ['fPSA/tPSA', 'free_PSA/total_PSA_ratio'],
    'pv_value': ['prostate_volume', 'PV'],
    'dre_abnormal': ['digital_rectal_exam', 'DRE', 'prostate_palpation']
}

MODEL_NAME = "Prostate Cancer Risk Prediction (CPCC-RC)"
REFERENCE = "Chen R, et al. (Urol Oncol. 2016) [CPCC-RC]"


# ==========================================
# 2. Basic cleaning and matching logic
# ==========================================
def normalize_str(s):
    if pd.isna(s):
        return ""
    return re.sub(r'[\(\)\（\）\[\]\【\】\{\}\s]', '', str(s)).lower()


def get_category(name):
    norm_name = normalize_str(name)
    for key, patterns in FIELD_MAPPING.items():
        for p in patterns:
            if normalize_str(p) == norm_name:
                return key
    return None


def clean_numeric(val):
    if pd.isna(val) or str(val).strip() == "":
        return np.nan
    try:
        match = re.search(r"[-+]?\d*\.\d+|\d+", str(val))
        return float(match.group()) if match else np.nan
    except:
        return np.nan


# --- Added: dedicated prostate-volume function (supports length/width/height) ---
def clean_pv_numeric(val, unit):
    s = str(val).strip()
    if pd.isna(val) or s == "":
        return np.nan, 'prostate_volume_missing'
    try:
        # Extract all numbers from the string
        matches = re.findall(r"[-+]?\d*\.\d+|\d+", s)
        numbers = [float(m) for m in matches]

        unit_text = normalize_unit(unit)
        raw_unit = str(unit or '').lower()
        if not unit_text:
            raw_text = s.lower()
            if 'mm' in raw_text:
                unit_text = 'mm'
            elif 'cm' in raw_text:
                unit_text = 'cm'
            elif 'ml' in raw_text or 'cm3' in raw_text or 'cm³' in raw_text:
                unit_text = 'ml'
        elif 'mm' in raw_unit and 'cm' not in raw_unit:
            unit_text = 'mm'
        elif 'cm' in raw_unit and ('3' not in raw_unit and '³' not in raw_unit):
            unit_text = 'cm'

        # All three diameters must be explicit mm or cm; use the TRUS ellipsoid volume formula from the original.
        if len(numbers) >= 3:
            if unit_text == 'mm':
                numbers = [n / 10.0 for n in numbers]
            elif unit_text != 'cm':
                return np.nan, 'prostate three diameters lack explicit mm or cm units'
            return numbers[0] * numbers[1] * numbers[2] * (math.pi / 6), f'{unit_text} three diameters converted to mL via the pi/6 ellipsoid formula'
        # A single value must be an explicit volume unit; do not mistake a length for a volume.
        elif len(numbers) > 0:
            if unit_text == 'ml':
                return numbers[0], 'volume unit is mL'
            return np.nan, 'prostate volume lacks explicit mL/cm3 unit'
        else:
            return np.nan, 'prostate_volume_value_unparseable'
    except:
        return np.nan, 'prostate_volume_unparseable'


def parse_dre(val):
    s = str(val)
    if pd.isna(val) or s.strip() == "":
        return False

    # --- Change: add exclusion logic for negative/normal descriptions ---
    if 'no_abnormality' in s or 'no_abnormality_seen' in s or 'normal' in s:
        return False

    if any(kw in s for kw in ['abnormal', 'nodule', 'hard', 'palpable', 'positive', '1']):
        return True
    return False


# ==========================================
# 3. CPCC-RC core computation logic
# ==========================================
def calculate_cpcc_rc(age, psa, pv, ratio, dre):
    # Substituting Table 1 typical values into Table 2 coefficients reproduces cohort risk magnitude with log10;
    # Using natural log would wrongly push the median development-cohort patient below ~1%.
    log_psa = math.log10(psa)
    log_pv = math.log10(pv)
    dre_val = 1 if dre else 0

    # Model 1 (overall PCa risk)
    z1 = (-1.6842 + (2.0441 * log_psa) + (-3.238 * log_pv) + (0.07109 * age) + (-4.1971 * ratio) + (0.7898 * dre_val))
    pca_prob = 1 / (1 + math.exp(-z1))

    # Model 2 (HGPCa high-grade risk, Gleason >= 7)
    z2 = (-3.0122 + (2.3164 * log_psa) + (-2.9291 * log_pv) + (0.0673 * age) + (-4.3519 * ratio) + (0.7367 * dre_val))
    hgpca_prob = 1 / (1 + math.exp(-z2))

    return pca_prob, hgpca_prob


# ==========================================
# 4. Main data-processing routine
# ==========================================
def process_data(base_path, item_path, output_path):
    print(f"--- Starting {MODEL_NAME} processing (missing-value output supported) ---")

    def read_csv_safe(path):
        for enc in ['gbk', 'utf-8-sig', 'utf-8']:
            try:
                return pd.read_csv(path, encoding=enc, low_memory=False)
            except:
                continue
        return pd.DataFrame()

    base_df = read_csv_safe(base_path)
    item_df = read_csv_safe(item_path)

    if base_df.empty or item_df.empty:
        print("error: cannot read data file or it is empty")
        return

    item_df['cat'] = item_df['res_itm_name'].apply(get_category)
    filtered = item_df.dropna(subset=['cat', 'res_value']).copy()

    records = []

    # Extract indicators
    for (app_no, pat_id), group in filtered.groupby(['app_no', 'pat_id']):
        row = {'app_no': str(app_no).strip(), 'pat_id': str(pat_id).strip()}
        data_pool = {k: [] for k in FIELD_MAPPING.keys()}
        unit_pool = {k: [] for k in FIELD_MAPPING.keys()}

        for _, item in group.iterrows():
            data_pool[item['cat']].append(str(item['res_value']))
            unit_pool[item['cat']].append(item.get('res_unit', ''))

        # Read single values directly; drop the merged-string logic
        age_vals = data_pool.get('age', [])
        psa_vals = data_pool.get('psa_value', [])
        pv_vals = data_pool.get('pv_value', [])
        ratio_vals = data_pool.get('fPSA/tPSA', [])
        dre_vals = data_pool.get('dre_abnormal', [])

        row['age_val'] = clean_numeric(age_vals[0]) if age_vals else np.nan
        psa_result = convert_measurement(
            psa_vals[0] if psa_vals else None,
            unit_pool['psa_value'][0] if psa_vals else '',
            indicator='psa', target_unit='ng/mL'
        )
        row['psa_val'] = psa_result.value if psa_result.ok else np.nan
        row['psa_unit_note'] = psa_result.rule if psa_result.ok else psa_result.error

        if pv_vals:
            row['pv_val'], row['pv_unit_note'] = clean_pv_numeric(
                pv_vals[0], unit_pool['pv_value'][0]
            )
        else:
            row['pv_val'], row['pv_unit_note'] = np.nan, 'prostate_volume_missing'

        ratio_result = convert_measurement(
            ratio_vals[0] if ratio_vals else None,
            unit_pool['fPSA/tPSA'][0] if ratio_vals else '',
            indicator='psa_ratio', target_unit='ratio'
        )
        row['ratio_val'] = ratio_result.value if ratio_result.ok else np.nan
        row['ratio_unit_note'] = ratio_result.rule if ratio_result.ok else ratio_result.error
        row['dre_val'] = parse_dre(dre_vals[0]) if dre_vals else False

        # Keep raw values for traceability
        for cat in FIELD_MAPPING.keys():
            row[f'raw_{cat}'] = " / ".join(data_pool.get(cat, []))

        records.append(row)

    if not records:
        print("no valid indicator data matching the model")
        return

    wide_df = pd.DataFrame(records)

    base_df['app_no'] = base_df['app_no'].astype(str).str.strip()
    base_df['pat_id'] = base_df['pat_id'].astype(str).str.strip()
    base_cols = [c for c in ['app_no', 'pat_id', 'pat_name', 'pat_sex', 'app_pat_age'] if c in base_df.columns]
    final_merged = pd.merge(wide_df, base_df[base_cols],
                            on=['app_no', 'pat_id'], how='inner')

    results = []
    for _, r in final_merged.iterrows():
        age_from_base = clean_numeric(r.get('app_pat_age'))
        if pd.isna(r['age_val']) and pd.notna(age_from_base):
            r['age_val'] = age_from_base
        core_metrics = [r['age_val'], r['psa_val'], r['pv_val'], r['ratio_val']]
        
        # Check core indicators for missing or non-positive values
        is_invalid = any(pd.isna(x) or x <= 0 for x in core_metrics)

        # === Core logic change: assign via branches instead of skipping with continue ===
        if is_invalid:
            pca_risk_display = "-"
            hgpca_risk_display = "-"
            risk_level = "-"
            code = "-"
        else:
            try:
                pca_p, hgpca_p = calculate_cpcc_rc(
                    r['age_val'], r['psa_val'], r['pv_val'],
                    r['ratio_val'], r['dre_val']
                )
                pca_score_pct = pca_p * 100
                hgpca_score_pct = hgpca_p * 100

                # The original gives a continuous predicted probability and decision-curve range without a single high/low cut-off.
                risk_level = "continuous_risk_probability"
                code = "-"

                pca_risk_display = f"{pca_score_pct:.2f}%"
                hgpca_risk_display = f"{hgpca_score_pct:.2f}%"
            except Exception:
                # Guard against math.exp overflow from extreme values
                pca_risk_display = "-"
                hgpca_risk_display = "-"
                risk_level = "-"
                code = "-"

        results.append({
            'pat_id': r['pat_id'],
            'app_no': r['app_no'],
            'name': r.get('pat_name', 'unknown'),
            'sex': r.get('pat_sex', ''),
            'model_name': MODEL_NAME,
            'prostate_cancer_risk': pca_risk_display,
            'high_grade_prostate_cancer_risk (%)': hgpca_risk_display,
            'prediction_horizon': 'current',
            'risk_level': risk_level,
            'code': code,
            'gauge_reference': "0-100",
            'age': r['age_val'] if pd.notna(r['age_val']) else np.nan,
            'PSA_value (ng/mL)': r['psa_val'] if pd.notna(r['psa_val']) else np.nan,
            'PSA_unit_handling': r.get('psa_unit_note', ''),
            'prostate_volume (mL)': r['pv_val'] if pd.notna(r['pv_val']) else np.nan,
            'prostate_volume_unit_handling': r.get('pv_unit_note', ''),
            'f/t_ratio': r['ratio_val'] if pd.notna(r['ratio_val']) else np.nan,
            'f/t_ratio_unit_handling': r.get('ratio_unit_note', ''),
            'abnormal_DRE': r['dre_val'],
            'reference': REFERENCE
        })

    if results:
        res_df = pd.DataFrame(results)
        res_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"--- Assessment complete: {len(res_df)} records (including missing values), exported to {output_path} ---")
    else:
        print("--- Assessment complete, but no complete records met the computation criteria ---")


if __name__ == "__main__":
    process_data(
        "base.csv",
        "item_result.csv",
        "cpcc_rc_risk_analysis.csv"
    )
