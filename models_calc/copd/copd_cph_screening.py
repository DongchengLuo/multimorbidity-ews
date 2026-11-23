import pandas as pd
import numpy as np
import math
import re
import warnings

# Ignore all warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. Field mapping (for the revised COPD primary-care screening tool)
# ==========================================
FIELD_MAPPING = {
    'region': ['region', 'administrative_division', 'residential_region'],
    'urban': ['residence_type', 'urban_rural', 'urban_code'],
    'education': ['education', 'education_level'],
    'BMI': ['(body mass index) BMI', 'bmi', 'BMI'],
    'height': ['height'],
    'weight': ['weight'],
    # Added 'smoking history' to match the field in item_result.csv
    'smoking_status': ['smoking_status', 'smoker', 'current_smoker', 'smoking_history'],
    'pack_years': ['pack_years', 'cumulative_smoking'],
    'cigs_per_day': ['daily_cigarettes', 'cigarettes_per_day'],
    'smoke_years': ['smoking_years'],
    'cooking_years': ['biofuel_exposure_years', 'solid_fuel_cooking_years', 'cooking_years', 'bio_fuel_years'],
    'family_history': ['copd_family_history', 'respiratory_family_history'],
    'tb_history': ['tuberculosis_history', 'TB history'],
    'sym_sob': ['dyspnea', 'shortness_of_breath', 'dyspnea_on_flat_walking', 'sob'],
    'sym_sputum': ['morning_sputum', 'sputum'],
    'sym_wheeze': ['wheeze', 'wheezing']
}


# ==========================================
# 2. Basic cleaning and helper functions
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
    """Strictly clean a single value to a number; NaN when missing"""
    if pd.isna(val) or val is None or str(val).strip() == "": return np.nan
    try:
        match = re.search(r"[-+]?\d*\.\d+|\d+", str(val))
        return float(match.group()) if match else np.nan
    except:
        return np.nan


def check_bool_match_strict(val, true_keywords):
    """Strict boolean check: return None for empty values instead of defaulting to False"""
    if pd.isna(val) or val is None or str(val).strip() == "": return None
    s = str(val).lower()
    return any(k in s for k in true_keywords)


def parse_region(val):
    if pd.isna(val) or val is None or str(val).strip() == "": return None
    s = str(val)
    if any(k in s for k in ['southwest', 'sichuan', 'chongqing', 'yunnan', 'guizhou']): return 'Southwest'
    if any(k in s for k in ['south_china', 'guangdong', 'guangxi', 'hainan']): return 'South'
    if any(k in s for k in ['east_china', 'shanghai', 'jiangsu', 'zhejiang']): return 'East'
    if any(k in s for k in ['central_china', 'hubei', 'hunan']): return 'Central'
    if any(k in s for k in ['northwest', 'shaanxi', 'gansu']): return 'Northwest'
    if any(k in s for k in ['northeast', 'heilongjiang', 'jilin', 'liaoning']): return 'Northeast'
    return None


def parse_education(val):
    if pd.isna(val) or val is None or str(val).strip() == "": return None
    s = str(val)
    if any(k in s for k in ['college', 'university', 'bachelor', 'postgraduate']): return 'College'
    if any(k in s for k in ['medium', 'high', 'vocational']): return 'Middle/High'
    if any(k in s for k in ['small', 'illiterate']): return 'Primary'
    return None


# ==========================================
# 3. Revised COPD model computation core
# ==========================================
def calculate_risk_corrected(data):
    try:
        age = data.get('age')
        is_female = data.get('is_female')
        is_urban = data.get('is_urban')
        region = data.get('region')
        edu = data.get('edu')
        is_current_smoker = data.get('is_current_smoker')
        pack_years = data.get('pack_years')
        cooking_years = data.get('cooking_years')
        has_family_history = data.get('has_family_history')
        has_tb_history = data.get('has_tb_history')
        bmi = data.get('bmi')
        has_sob = data.get('has_sob')
        has_sputum = data.get('has_sputum')
        has_wheeze = data.get('has_wheeze')

        # --- Missing-value blocking check ---
        check_list = [age, is_female, is_urban, region, edu, is_current_smoker,
                      pack_years, cooking_years, has_family_history, has_tb_history,
                      bmi, has_sob, has_sputum, has_wheeze]

        for item in check_list:
            if item is None or (isinstance(item, float) and np.isnan(item)):
                return None, "missing_required_indicators"

        # --- Compute logit ---
        # Coefficients from Supplementary Table 2 (cm9-136-676-s001.docx).
        logit = math.log(0.021)
        logit += 0.057 * age

        if is_female: logit += -0.360
        if is_urban: logit += -0.395

        region_coeffs = {
            'South': -0.528, 'East': -0.389, 'Central': -0.513,
            'Northwest': -0.416, 'Northeast': 0.134, 'Southwest': 0
        }
        logit += region_coeffs.get(region, 0)

        if edu == 'Middle/High':
            logit += -0.235
        elif edu == 'College':
            logit += -0.568

        if is_current_smoker: logit += 0.335
        logit += 0.004 * pack_years
        if 31 <= cooking_years <= 60:
            logit += 0.114
        elif cooking_years >= 61:
            logit += 0.317

        if has_family_history: logit += 0.195
        if has_tb_history: logit += 0.439

        if bmi < 18.5:
            logit += 0.303
        elif 24.0 <= bmi <= 27.9:
            logit += -0.336
        elif bmi >= 28.0:
            logit += -0.512

        if has_sob: logit += 0.337
        if has_sputum: logit += 0.248
        if has_wheeze: logit += 0.358

        probability = 1 / (1 + math.exp(-logit))
        return probability, f"Logit={logit:.4f}"

    except Exception as e:
        return None, f"computation error: {str(e)}"


# ==========================================
# 4. Data-processing flow
# ==========================================
def process_data(base_path, item_path, output_path):
    print("--- Starting COPD primary-care screening tool (strict missing-value output) ---")

    GAUGE_REF = "0-0.22|0.22-1"

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
        print("Data file could not be read or is empty")
        return

    base_df['app_no'] = base_df['app_no'].astype(str).str.strip()
    base_df['pat_id'] = base_df['pat_id'].astype(str).str.strip()
    item_df['app_no'] = item_df['app_no'].astype(str).str.strip()
    item_df['pat_id'] = item_df['pat_id'].astype(str).str.strip()

    item_df['cat'] = item_df['res_itm_name'].apply(get_category)
    filtered = item_df.dropna(subset=['cat', 'res_value']).copy()

    records = []

    # Iterate the base table to include everyone
    for _, base_row in base_df.iterrows():
        app_no = str(base_row['app_no']).strip()
        pat_id = str(base_row['pat_id']).strip()
        row = {'app_no': app_no, 'pat_id': pat_id}

        # Extract all matched indicators for this patient from the exam table
        data_pool = {}
        group = filtered[(filtered['app_no'] == app_no) & (filtered['pat_id'] == pat_id)]
        for _, item in group.iterrows():
            data_pool[item['cat']] = item['res_value']

        # --- Basic information extraction (read from base_df) ---
        age_val = base_row['app_pat_age'] if 'app_pat_age' in base_df.columns else np.nan
        row['age'] = clean_numeric(age_val)

        sex_val = base_row['pat_sex'] if 'pat_sex' in base_df.columns else None
        sex_str = str(sex_val).strip()
        if sex_str in ['2', '2.0', 'female']:
            row['is_female'] = True
        elif sex_str in ['1', '1.0', 'male']:
            row['is_female'] = False
        else:
            row['is_female'] = None

        # --- Questionnaire feature extraction (read from data_pool) ---
        urban_val = data_pool.get('urban')
        if pd.isna(urban_val) or str(urban_val).strip() == "":
            row['is_urban'] = None
        else:
            urban_str = str(urban_val)
            if 'urban' in urban_str or '1' in urban_str:
                row['is_urban'] = True
            elif 'farming' in urban_str or 'township' in urban_str or '0' in urban_str:
                row['is_urban'] = False
            else:
                row['is_urban'] = None

        row['region'] = parse_region(data_pool.get('region'))
        row['edu'] = parse_education(data_pool.get('education'))

        # BMI
        bmi_val = clean_numeric(data_pool.get('BMI'))
        if pd.isna(bmi_val):
            h = clean_numeric(data_pool.get('height'))
            w = clean_numeric(data_pool.get('weight'))
            if not pd.isna(h) and not pd.isna(w) and h > 0 and w > 0:
                if h > 3: h = h / 100.0
                bmi_val = w / (h * h)
        row['bmi'] = bmi_val

        # Smoking logic
        smk_val = data_pool.get('smoking_status')
        if pd.isna(smk_val) or str(smk_val).strip() == "":
            row['is_current_smoker'] = None
            row['pack_years'] = np.nan
        else:
            smk_str = str(smk_val)
            if 'absent' in smk_str or 'no' in smk_str:
                row['is_current_smoker'] = False
                row['pack_years'] = 0.0
            else:
                row['is_current_smoker'] = True if ('yes' in smk_str or 'present' in smk_str or '1' in smk_str) else None
                py = clean_numeric(data_pool.get('pack_years'))
                if pd.isna(py):
                    cigs = clean_numeric(data_pool.get('cigs_per_day'))
                    yrs = clean_numeric(data_pool.get('smoke_years'))
                    if not pd.isna(cigs) and not pd.isna(yrs):
                        py = (cigs / 20.0) * yrs
                    else:
                        py = np.nan
                row['pack_years'] = py

        row['cooking_years'] = clean_numeric(data_pool.get('cooking_years'))
        row['has_family_history'] = check_bool_match_strict(data_pool.get('family_history'), ['yes', 'present', '1'])
        row['has_tb_history'] = check_bool_match_strict(data_pool.get('tb_history'), ['yes', 'present', '1'])
        row['has_sob'] = check_bool_match_strict(data_pool.get('sym_sob'), ['yes', 'present', '1'])
        row['has_sputum'] = check_bool_match_strict(data_pool.get('sym_sputum'), ['yes', 'present', '1'])
        row['has_wheeze'] = check_bool_match_strict(data_pool.get('sym_wheeze'), ['yes', 'present', '1'])

        records.append(row)

    wide_df = pd.DataFrame(records)

    if wide_df.empty:
        print("no valid data extracted")
        return

    final_df = pd.merge(wide_df, base_df[['app_no', 'pat_id', 'pat_name']], on=['app_no', 'pat_id'], how='inner')

    # --- 5. Batch computation and formatted output ---
    print("computing risk probability...")
    results = []

    def format_val(val, val_type='text'):
        if pd.isna(val) or val is None:
            return "missing"
        if val_type == 'bool':
            return "yes" if val else "no"
        if val_type == 'num':
            return f"{val:.1f}" if isinstance(val, float) and not val.is_integer() else str(int(val))
        trans_dict = {
            'Southwest': 'southwest', 'South': 'south_china', 'East': 'east_china', 'Central': 'central_china',
            'Northwest': 'northwest', 'Northeast': 'northeast',
            'College': 'college_or_above', 'Middle/High': 'middle/vocational_school', 'Primary': 'primary/illiterate'
        }
        return trans_dict.get(str(val), str(val))

    for _, r in final_df.iterrows():
        calc_input = {
            'age': r.get('age'), 'is_female': r.get('is_female'), 'is_urban': r.get('is_urban'),
            'region': r.get('region'), 'edu': r.get('edu'), 'is_current_smoker': r.get('is_current_smoker'),
            'pack_years': r.get('pack_years'), 'cooking_years': r.get('cooking_years'),
            'has_family_history': r.get('has_family_history'), 'has_tb_history': r.get('has_tb_history'),
            'bmi': r.get('bmi'), 'has_sob': r.get('has_sob'), 'has_sputum': r.get('has_sputum'),
            'has_wheeze': r.get('has_wheeze')
        }

        prob, msg = calculate_risk_corrected(calc_input)

        if prob is None:
            prob_str, risk_code, conclusion = "-", "-", "-"
        else:
            prob_str = f"{prob:.4f}"
            risk_code = "J44.2.3" if prob >= 0.22 else "J44.2.1"
            conclusion = "high risk" if prob >= 0.22 else "low risk"

        results.append({
            'app_no': r.get('app_no'),
            'pat_id': r.get('pat_id'),
            'pat_name': r.get('pat_name', 'unknown'),
            'sex': "missing" if calc_input.get('is_female') is None else ("female" if calc_input.get('is_female') else "male"),
            'age': format_val(calc_input.get('age'), 'num'),
            'urban': format_val(calc_input.get('is_urban'), 'bool'),
            'region': format_val(calc_input.get('region')),
            'education_level': format_val(calc_input.get('edu')),
            'BMI': format_val(calc_input.get('bmi'), 'num'),
            'current_smoker': format_val(calc_input.get('is_current_smoker'), 'bool'),
            'pack_years': format_val(calc_input.get('pack_years'), 'num'),
            'solid_fuel_cooking_years': format_val(calc_input.get('cooking_years'), 'num'),
            'copd_family_history': format_val(calc_input.get('has_family_history'), 'bool'),
            'tuberculosis_history': format_val(calc_input.get('has_tb_history'), 'bool'),
            'shortness_of_breath/dyspnea': format_val(calc_input.get('has_sob'), 'bool'),
            'morning_sputum': format_val(calc_input.get('has_sputum'), 'bool'),
            'wheeze': format_val(calc_input.get('has_wheeze'), 'bool'),
            'model_name': "COPD-CPH screening model",
            'risk_probability': prob_str,
            'risk_level': conclusion,
            'code': risk_code,
            'gauge_reference': GAUGE_REF,
            'prediction_horizon': 'current',
            'note': msg if prob is None else ""
        })

    if results:
        res_df = pd.DataFrame(results)

        # Removed the original cols reordering; output as-is
        res_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        valid_count = len(res_df[res_df['risk_probability'] != '-'])
        print(f"--- Assessment complete: {len(res_df)} records processed ---")
        print(f"--- Valid computed records: {valid_count} ---")
        print(f"--- Records not computable due to missing values: {len(res_df) - valid_count} ---")
        print(f"--- Results saved to: {output_path} ---")


if __name__ == "__main__":
    process_data("base.csv", "item_result.csv", "copd_CPH_risk_result.csv")
