import math
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model_unit_utils import convert_measurement


FIELD_MAPPING = {
    'BMI': ['(body mass index) BMI', 'bmi', 'BMI'],
    'height': ['height'],
    'weight': ['weight'],
    'pathology': ['histology_type', 'pathological_diagnosis', 'cytology_type', 'histology'],
    'tnm_stage': ['clinical_stage', 'tnm_stage', 'tumor_stage', 'stage'],
    'cvc_history': ['central_venous_catheter', 'CVC_catheter', 'PICC_catheter', 'CVC'],
    'd_dimer': ['d_dimer', 'D-Dimer', 'Ddi'],
    'pt': ['prothrombin_time', 'PT'],
    'fbg': ['fibrinogen', 'Fbg', 'FIB'],
    'tg': ['triglyceride', 'TG', 'Triglyceride'],
    'ros1': ['ROS1_gene_rearrangement', 'ROS1', 'ROS-1'],
    'chemo_history': ['chemotherapy_history', 'chemotherapy_within_6_months', 'chemotherapy'],
    'radio_history': ['radiotherapy_history', 'radiotherapy_within_6_months', 'radiotherapy'],
}

MODEL_NAME = 'VTE Risk Nomogram for Hospitalized Lung Cancer Patients (Li et al. 2022)'
GAUGE_REF = '0-300.6|300.6 and above (nomogram total)'
NOMOGRAM_CUTOFF = 300.6
COEFFICIENTS = {
    'BMI_cat(>=24)': 0.643,
    'Adenocarcinoma': 1.099,
    'Stage_III_IV': 1.011,
    'CVC_History': 1.543,
    'D_Dimer_cat(>=2.06mg/L)': 1.719,
    'PT_cat(>=11.45sec)': 0.766,
    'Fbg_cat(>=3.33g/L)': 0.563,
    'TG_cat(>=1.37mmol/L)': 0.631,
    'ROS1_Pos': 1.055,
    'Chemo_6M': 0.505,
    'Radio_6M': 0.674,
}


def normalize_str(value):
    if pd.isna(value):
        return ''
    return re.sub(r'[\(\)\（\）\[\]\【\】\{\}\s]', '', str(value)).lower()


def get_category(name):
    norm_name = normalize_str(name)
    for key, patterns in FIELD_MAPPING.items():
        if any(normalize_str(pattern) in norm_name for pattern in patterns):
            return key
    return None


def clean_numeric(value):
    if pd.isna(value) or not str(value).strip():
        return np.nan
    match = re.search(r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)', str(value))
    return float(match.group()) if match else np.nan


def parse_binary(values):
    """Explicit yes/no parsing; missing, conflicting or ambiguous text returns None."""
    texts = [normalize_str(value) for value in values if str(value).strip()]
    if not texts:
        return None
    positives, negatives = [], []
    for text in texts:
        negative = text in {'0', 'no', 'absent', 'negative', 'false'} or any(
            token in text for token in ('denies', 'not_performed', 'not_done', 'not_placed', 'no_chemotherapy', 'no_radiotherapy', 'no_rearrangement')
        )
        positive = text in {'1', 'yes', 'present', 'positive', 'true'} or any(
            token in text for token in ('catheterization', 'rearrangement_positive', 'had_chemotherapy', 'receiving_chemotherapy', 'had_radiotherapy', 'receiving_radiotherapy')
        )
        negatives.append(negative)
        positives.append(positive and not negative)
    if any(positives) and any(negatives):
        return None
    if any(positives):
        return 1
    if all(negatives):
        return 0
    return None


def parse_pathology(values):
    texts = [normalize_str(value) for value in values if str(value).strip()]
    if not texts:
        return None
    return 1 if any('adenocarcinoma' in text or 'adenocarcinoma' in text for text in texts) else 0


def parse_stage(values):
    texts = [normalize_str(value).replace('stage', '') for value in values if str(value).strip()]
    if not texts:
        return None
    for text in texts:
        if re.search(r'(^|[^i])iv($|[^i])|(^|[^i])iii($|[^i])|(stage\s*)?[34]|advanced', text):
            return 1
        if re.search(r'(^|[^i])ii($|[^i])|(^|[^i])i($|[^i])|(stage\s*)?[12]|early', text):
            return 0
    return None


def calculate_vte_risk(bmi, is_adeno, is_advanced, has_cvc, ddi, pt, fbg, tg, ros1, chemo, radio):
    flags = {
        'BMI_cat(>=24)': int(pd.notna(bmi) and bmi >= 24.0),
        'Adenocarcinoma': int(is_adeno == 1),
        'Stage_III_IV': int(is_advanced == 1),
        'CVC_History': int(has_cvc == 1),
        'D_Dimer_cat(>=2.06mg/L)': int(pd.notna(ddi) and ddi >= 2.06),
        'PT_cat(>=11.45sec)': int(pd.notna(pt) and pt >= 11.45),
        'Fbg_cat(>=3.33g/L)': int(pd.notna(fbg) and fbg >= 3.33),
        'TG_cat(>=1.37mmol/L)': int(pd.notna(tg) and tg >= 1.37),
        'ROS1_Pos': int(ros1 == 1),
        'Chemo_6M': int(chemo == 1),
        'Radio_6M': int(radio == 1),
    }
    coefficient_sum = sum(COEFFICIENTS[key] * value for key, value in flags.items())
    logit_p = -6.306 + coefficient_sum
    probability = 1.0 / (1.0 + math.exp(-logit_p))
    # The original nomogram sets the largest regression coefficient (D-dimer 1.719) to 100 points.
    nomogram_points = coefficient_sum * (100.0 / 1.719)
    details = {
        'model_intercept': -6.306,
        'vars': flags,
        'logit_score': round(logit_p, 4),
        'nomogram_points': round(nomogram_points, 2),
        'source_cutoff': NOMOGRAM_CUTOFF,
    }
    return probability, details


def _read_csv_safe(path):
    for encoding in ('utf-8-sig', 'gb18030', 'gbk', 'utf-8'):
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False)
        except Exception:
            continue
    return pd.DataFrame()


def _first_numeric(values):
    for value in values:
        number = clean_numeric(value)
        if pd.notna(number):
            return number
    return np.nan


def _converted_value(group, category, indicator, target_unit):
    errors = []
    for _, measurement in group[group['cat'] == category].iterrows():
        result = convert_measurement(
            measurement.get('res_value'), measurement.get('res_unit', ''),
            indicator=indicator, target_unit=target_unit,
        )
        if result.ok:
            return result.value, f'{category}: {result.rule}', ''
        errors.append(f'{category}: {result.error}')
    return np.nan, '', '；'.join(errors)


def process_vte_data(base_path, item_path, output_path):
    print('--- Starting VTE risk model for hospitalized lung cancer patients ---')
    base_df, item_df = _read_csv_safe(base_path), _read_csv_safe(item_path)
    if base_df.empty:
        print('error: base data file empty or unreadable.')
        return

    for frame in (base_df, item_df):
        if not frame.empty:
            frame['app_no'] = frame['app_no'].astype(str).str.strip()
            frame['pat_id'] = frame['pat_id'].astype(str).str.strip()
    if not item_df.empty:
        item_df['cat'] = item_df['res_itm_name'].apply(get_category)
        item_df = item_df.dropna(subset=['cat']).copy()

    results = []
    for _, base_row in base_df.iterrows():
        app_no, pat_id = str(base_row['app_no']).strip(), str(base_row['pat_id']).strip()
        group = item_df[(item_df['app_no'] == app_no) & (item_df['pat_id'] == pat_id)] if not item_df.empty else pd.DataFrame()
        pool = {key: group[group['cat'] == key]['res_value'].tolist() if not group.empty else [] for key in FIELD_MAPPING}

        bmi = _first_numeric(pool['BMI'])
        if pd.isna(bmi):
            height, weight = _first_numeric(pool['height']), _first_numeric(pool['weight'])
            if pd.notna(height) and pd.notna(weight) and height > 3 and weight > 0:
                bmi = weight / ((height / 100.0) ** 2)

        unit_notes, unit_errors = [], []
        lab_specs = {
            'd_dimer': ('d_dimer_mass', 'mg/L'),
            'pt': ('time', 'sec'),
            'fbg': ('fibrinogen', 'g/L'),
            'tg': ('triglyceride', 'mmol/L'),
        }
        labs = {}
        for category, (indicator, target) in lab_specs.items():
            value, note, error = _converted_value(group, category, indicator, target) if not group.empty else (np.nan, '', '')
            labs[category] = value
            if note: unit_notes.append(note)
            if error: unit_errors.append(error)

        parsed = {
            'histology_type': parse_pathology(pool['pathology']),
            'tnm_stage': parse_stage(pool['tnm_stage']),
            'central_venous_catheter': parse_binary(pool['cvc_history']),
            'ROS1_rearrangement': parse_binary(pool['ros1']),
            'chemotherapy_within_6_months': parse_binary(pool['chemo_history']),
            'radiotherapy_within_6_months': parse_binary(pool['radio_history']),
        }
        probability, details = calculate_vte_risk(
            bmi, parsed['histology_type'], parsed['tnm_stage'], parsed['central_venous_catheter'],
            labs['d_dimer'], labs['pt'], labs['fbg'], labs['tg'], parsed['ROS1_rearrangement'],
            parsed['chemotherapy_within_6_months'], parsed['radiotherapy_within_6_months'],
        )
        high_risk = details['nomogram_points'] >= NOMOGRAM_CUTOFF
        risk_level, code = ('high risk', 'I82.5.3') if high_risk else ('low risk', 'I82.5.1')
        calc_status, reasons = 'computed', []

        results.append({
            'pat_id': pat_id, 'app_no': app_no, 'pat_name': base_row.get('pat_name', 'unknown'),
            'model_name': MODEL_NAME, 'sex': base_row.get('pat_sex', ''),
            'age': base_row.get('app_pat_age', ''), 'calculation_status': calc_status,
            'not_computed_reason': '；'.join(reasons),
            'VTE risk probability (6 months after cancer diagnosis and hospitalization)': f'{probability:.2%}' if probability is not None else '-',
            'nomogram_total_score': details.get('nomogram_points', '-'), 'risk_level': risk_level,
            'code': code, 'gauge_reference': GAUGE_REF,
            'unit_conversion': '；'.join(unit_notes), 'computation_inputs': str(details) if details else '-',
        })

    pd.DataFrame(results).to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f'--- Assessment complete: {len(results)} records processed, results saved to {output_path} ---')


if __name__ == '__main__':
    process_vte_data('base.csv', 'item_result.csv', 'lung_cancer_vte_risk_result.csv')
