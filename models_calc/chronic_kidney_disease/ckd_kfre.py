import pandas as pd
import numpy as np
import re
import math
import warnings
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model_unit_utils import convert_measurement

warnings.filterwarnings('ignore', category=pd.errors.DtypeWarning)

# ==========================================
# 1. Field mapping (exact match)
# ==========================================
FIELD_MAPPING = {
    # --- 1. Basic information ---
    'age': ['app_pat_age', 'age'],
    'gender': ['pat_sex', 'sex'],

    # --- 2. Core computation indicators ---
    'egfr': ['eGFR (CKD-EPI)', 'egfr', 'estimated_gfr'], 
    'scr': ['creatinine (Cr)', 'creatinine', 'serum_creatinine', 'scr'], 
    'acr': ['uacr', 'albumin_creatinine_ratio'], 
    'ALB': ['albumin', 'ALB'], 

    # --- 3. Model 6 extended indicators ---
    'calcium': ['calcium (Ca)', 'calcium', 'ca'],
    'phosphate': ['serum_phosphate', 'phosphate', 'phosphorous', 'p'],
    'bicarbonate': ['bicarbonate', 'CO2_combining_power', 'CO2', 'TCO2', 'hco3', 'co2cp', 'total_CO2']
}

# ==========================================
# 2. Model parameter definitions (KFRE 5-year risk)
# ==========================================
KFRE_PARAMS = {
    'model_3': {
        'baseline_survival': 0.924,
        'coeffs': {'sex_male': 0.2694, 'age': -0.2167, 'egfr': -0.55418, 'acr': 0.45608},
        'means': {'sex_male': 0.56422, 'age': 7.0355, 'egfr': 7.2216, 'acr': 5.2774}
    },
    'model_6': {
        'baseline_survival': 0.929,
        'coeffs': {
            'sex_male': 0.16117, 'age': -0.19883, 'egfr': -0.4936, 'acr': 0.35066,
            'calcium': -0.22129, 'phosphate': 0.24197, 'bicarbonate': -0.074291, 'ALB': -0.33867
        },
        'means': {
            'sex_male': 0.56422, 'age': 7.0355, 'egfr': 7.2216, 'acr': 5.2774,
            'calcium': 9.351, 'phosphate': 3.9221, 'bicarbonate': 25.5441, 'ALB': 3.9925
        }
    }
}

# ==========================================
# 3. Basic utility functions
# ==========================================
def normalize_str(s):
    if pd.isna(s): return ""
    return re.sub(r'[\(\)\（\）\[\]\【\】\{\}\s]', '', str(s)).lower()

def get_category(name):
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

def parse_gender_to_num(val):
    """Returns 1.0 (male) or 0.0 (female); NaN when undetermined"""
    s = normalize_str(val)
    if 'male' in s or 'm' in s or s == '1': return 1.0
    if 'female' in s or 'f' in s or s == '0' or s == '2': return 0.0
    return np.nan

def calculate_egfr_ckd_epi(scr, age, sex_male):
    """Compute eGFR with the CKD-EPI formula; scr must already be in mg/dL."""
    if any(pd.isna(x) for x in [scr, age, sex_male]) or scr <= 0:
        return np.nan

    scr_mgdl = scr
    
    if sex_male == 0.0: # female
        kappa, alpha, multiplier = 0.7, -0.329, 1.018
    else: # male
        kappa, alpha, multiplier = 0.9, -0.411, 1.0
        
    min_val = min(scr_mgdl / kappa, 1)
    max_val = max(scr_mgdl / kappa, 1)
    
    egfr = 141 * (min_val ** alpha) * (max_val ** -1.209) * (0.993 ** age) * multiplier
    return egfr

# ==========================================
# 4. KFRE core computation logic
# ==========================================
def calculate_kfre_score(data_dict):
    """
    Returns: (model_name, risk_prob, risk_label, risk_code)
    """
    try:
        raw_age = data_dict.get('age')
        raw_sex_num = data_dict.get('sex_num')
        raw_egfr = data_dict.get('egfr')
        raw_acr = data_dict.get('acr')

        if any(pd.isna(x) for x in [raw_age, raw_sex_num, raw_egfr, raw_acr]):
            return None, None, "-", "-"

        # KFRE variable transformation
        inputs = {
            'sex_male': raw_sex_num,
            'age': raw_age / 10.0,
            'egfr': raw_egfr / 5.0,
            'acr': math.log(raw_acr) if raw_acr > 0 else math.log(1.0)
        }

        ext_inputs = {
            'calcium': data_dict.get('calcium'),
            'phosphate': data_dict.get('phosphate'),
            'bicarbonate': data_dict.get('bicarbonate'),
            'ALB': data_dict.get('ALB')
        }

        # Model selection
        if not any(pd.isna(v) for v in ext_inputs.values()):
            model_name = "Model 6 (8-Var)"
            params = KFRE_PARAMS['model_6']
            inputs.update(ext_inputs)
        else:
            model_name = "Model 3 (4-Var)"
            params = KFRE_PARAMS['model_3']
            inputs = {key: inputs[key] for key in params['coeffs'] if key in inputs}

        sum_beta_x_minus_xbar = 0.0
        for var, beta in params['coeffs'].items():
            if var not in inputs or pd.isna(inputs[var]): return None, None, "-", "-"
            sum_beta_x_minus_xbar += beta * (inputs[var] - params['means'][var])

        exponent = math.exp(sum_beta_x_minus_xbar)
        risk = 1.0 - (params['baseline_survival'] ** exponent)
        risk_percent = risk * 100.0

        if risk_percent < 5.0:
            risk_code, risk_label = "N18.4.1", "low risk"
        elif 5.0 <= risk_percent < 15.0:
            risk_code, risk_label = "N18.4.2", "medium risk"
        else:
            risk_code, risk_label = "N18.4.3", "high risk"
            
        return model_name, risk_percent, risk_label, risk_code

    except Exception:
        return None, None, "-", "-"

# ==========================================
# 5. Main data-processing routine
# ==========================================
def process_data(base_path, item_path, output_path):
    print("--- Starting KFRE kidney failure risk model ---")

    def read_csv_safe(path):
        for enc in ['utf-8-sig', 'gb18030', 'gbk', 'utf-8']:
            try: return pd.read_csv(path, encoding=enc, low_memory=False)
            except: continue
        return pd.DataFrame()

    base_df = read_csv_safe(base_path)
    item_df = read_csv_safe(item_path)
    if base_df.empty:
        print("Error: cannot read the base data file.")
        return

    # 1. Indicator extraction
    records = []
    if not item_df.empty:
        item_df['cat'] = item_df['res_itm_name'].apply(get_category)
        filtered = item_df.dropna(subset=['cat', 'res_value']).copy()

        for (app_no, pat_id), group in filtered.groupby(['app_no', 'pat_id']):
            row = {'app_no': str(app_no).strip(), 'pat_id': str(pat_id).strip()}
            # Age and sex usually come from base; keep compatible reads from item.
            for cat_key in ['age', 'gender']:
                vals = [clean_numeric(v) for v in group[group['cat'] == cat_key]['res_value'].tolist()]
                valid_vals = [v for v in vals if not pd.isna(v)]
                row[cat_key] = valid_vals[0] if valid_vals else np.nan

            conversion_specs = {
                'egfr': ('egfr', 'mL/min/1.73m2'),
                'scr': ('creatinine', 'mg/dL'),
                'acr': ('acr', 'mg/g'),
                'ALB': ('albumin', 'g/dL'),
                'calcium': ('calcium', 'mg/dL'),
                'phosphate': ('phosphate', 'mg/dL'),
                'bicarbonate': ('bicarbonate', 'mEq/L'),
            }
            unit_notes, unit_errors = [], []
            for cat_key, (indicator, target_unit) in conversion_specs.items():
                cat_rows = group[group['cat'] == cat_key]
                converted_value = np.nan
                cat_errors = []
                for _, measurement in cat_rows.iterrows():
                    result = convert_measurement(
                        measurement.get('res_value'),
                        measurement.get('res_unit', ''),
                        indicator=indicator,
                        target_unit=target_unit,
                    )
                    if result.ok:
                        converted_value = result.value
                        unit_notes.append(f"{cat_key}: {result.rule}")
                        break
                    cat_errors.append(f"{cat_key}: {result.error}")
                if pd.isna(converted_value):
                    unit_errors.extend(cat_errors)
                row[cat_key] = converted_value
            row['unit_conversion'] = '；'.join(unit_notes)
            row['unit_error'] = '；'.join(unit_errors)
            records.append(row)

    wide_items = pd.DataFrame(records) if records else pd.DataFrame(columns=['app_no', 'pat_id'])

    # 2. Merge basic information (left join)
    base_df['app_no'] = base_df['app_no'].astype(str).str.strip()
    base_df['pat_id'] = base_df['pat_id'].astype(str).str.strip()
    
    overlap_cols = [c for c in wide_items.columns if c in base_df.columns and c not in ['app_no', 'pat_id']]
    if overlap_cols: base_df = base_df.drop(columns=overlap_cols)
    
    final = pd.merge(base_df, wide_items, on=['app_no', 'pat_id'], how='left')

    # 3. Loop computation
    results = []
    for _, r in final.iterrows():
        age = clean_numeric(r.get('app_pat_age'))
        sex_raw = r.get('pat_sex')
        sex_num = parse_gender_to_num(sex_raw)

        # eGFR completion logic
        egfr = r.get('egfr')
        egfr_source = "raw_data"
        if pd.isna(egfr) and not pd.isna(r.get('scr')):
            egfr = calculate_egfr_ckd_epi(r.get('scr'), age, sex_num)
            egfr_source = "creatinine_derived" if pd.notna(egfr) else "raw_data"

        data_dict = {
            'age': age, 'sex_num': sex_num, 'egfr': egfr,
            'scr': r.get('scr'), 'acr': r.get('acr'), 'ALB': r.get('ALB'),
            'calcium': r.get('calcium'), 'phosphate': r.get('phosphate'), 
            'bicarbonate': r.get('bicarbonate')
        }

        model_used, risk_percent, risk_label, risk_code = calculate_kfre_score(data_dict)

        missing = []
        if pd.isna(age): missing.append('age')
        if pd.isna(sex_num): missing.append('sex')
        if pd.isna(egfr): missing.append('eGFR')
        if pd.isna(r.get('acr')): missing.append('ACR')
        if missing:
            calc_status = 'input_incomplete'
        elif risk_percent is None:
            calc_status = 'computation_failed'
        else:
            calc_status = 'computed'
        reason_parts = []
        if missing: reason_parts.append('missing:' + '、'.join(missing))
        if r.get('unit_error', ''): reason_parts.append(str(r.get('unit_error')))
        if calc_status == 'computation_failed': reason_parts.append('numeric_anomaly_in_formula')

        # Format for display
        def fmt_in(v, r_type='num'):
            if pd.isna(v) or v is None: return "missing"
            return f"{v:.1f}" if r_type == 'num' else str(v)

        results.append({
            'app_no': r.get('app_no'),
            'pat_id': r.get('pat_id'),
            'pat_name': r.get('pat_name', 'unknown'),
            # --- Raw indicators (missing shown as "missing") ---
            'age': int(age) if pd.notna(age) else "missing",
            'sex': "male" if sex_num == 1.0 else ("female" if sex_num == 0.0 else "missing"),
            'creatinine': fmt_in(r.get('scr')),
            'eGFR': fmt_in(egfr),
            'eGFR_source': egfr_source if pd.notna(egfr) else "missing",
            'ACR(mg/g)': fmt_in(r.get('acr')),
            'unit_conversion': r.get('unit_conversion', ''),
            'unit_error': r.get('unit_error', ''),
            # --- Computation result (shows "-" when not computable) ---
            'model_used': model_used if model_used else "-",
            'risk_probability': f"{risk_percent:.2f}%" if risk_percent is not None else "-",
            'risk_level': risk_label,
            'code': risk_code,
            'calculation_status': calc_status,
            'not_computed_reason': '；'.join(reason_parts),
            'gauge_reference': '0-5|5-15|15-100',
            'extended_indicators': f"Ca(mg/dL):{fmt_in(r.get('calcium'))}, P(mg/dL):{fmt_in(r.get('phosphate'))}, Alb(g/dL):{fmt_in(r.get('ALB'))}, HCO3(mEq/L):{fmt_in(r.get('bicarbonate'))}"
        })

    if results:
        pd.DataFrame(results).to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"Assessment complete: {len(results)} results exported to {output_path}")

if __name__ == "__main__":
    process_data("base.csv", "item_result.csv", "kfre_kidney_failure_risk_result.csv")
