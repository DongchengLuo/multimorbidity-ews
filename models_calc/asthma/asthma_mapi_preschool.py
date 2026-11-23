import pandas as pd
import numpy as np
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model_unit_utils import convert_measurement

# ==========================================
# 1. Field mapping (for the mAPI model)
# ==========================================
# Key is the internal standard field name; value is a possible raw column name or keyword
FIELD_MAPPING = {
    # --- Required core indicators ---
    'wheezing_freq': ['wheeze_episodes_within_1y', 'wheeze_frequency', 'wheezing_episodes_per_year', 'wheeze_count'],

    # --- Major criteria ---
    'parental_asthma': ['parental_asthma'],
    'atopic_dermatitis': ['atopic_dermatitis', 'eczema', 'physician_diagnosed_eczema'],
    'aeroallergen_sens': ['aeroallergen_sensitization', 'aeroallergen'],

    # --- Minor criteria ---
    'wheezing_no_cold': ['non_cold_wheeze', 'no_cold_wheeze', 'wheezing_unrelated_to_colds', 'wheeze_without_cold'],
    'eosinophils_pct': ['eosinophil_percentage', 'EOS%', 'eosinophils_percent', 'peripheral_eosinophils'],
    'food_allergen_sens': ['milk_egg_peanut_sensitization', 'milk/egg/peanut_sensitization']
}


# ==========================================
# 2. Basic cleaning and matching logic
# ==========================================
def normalize_str(s):
    """Normalize the string: strip special symbols and lowercase"""
    if pd.isna(s): return ""
    return re.sub(r'[\(\)\（\）\[\]\【\】\{\}\s]', '', str(s)).lower()


def get_category(name):
    """Match the standard field key from the raw column name"""
    norm_name = normalize_str(name)
    for key, patterns in FIELD_MAPPING.items():
        for p in patterns:
            if normalize_str(p) in norm_name: return key
    return None


def clean_numeric(val):
    """Extract numeric value (for counts, percentages)"""
    if pd.isna(val) or str(val).strip() == "": return np.nan
    try:
        # Match floats or integers
        match = re.search(r"[-+]?\d*\.\d+|\d+", str(val))
        return float(match.group()) if match else np.nan
    except:
        return np.nan


def parse_bool_indicator(val_list):
    """Parse a boolean indicator (True if any affirmative word appears in the list)"""
    text = " ".join([str(v) for v in val_list])
    if any(kw in text.lower() for kw in ['no', 'absent', 'none', 'not', '0', 'false', 'negative']):
        return 0
    # Match: 1, True, yes, present, Positive
    is_true = any(kw in text for kw in ['yes', 'present', '1', 'true', 'positive'])
    return 1 if is_true else 0


# ==========================================
# 3. mAPI prediction core logic
# ==========================================
def calculate_mapi_assessment(
        wheezing_freq,
        parental_asthma,
        atopic_dermatitis,
        aeroallergen_sens,
        wheezing_no_cold,
        eosinophils_pct,
        food_allergen_sens
):
    """
    Compute the mAPI index
    Returns: (Risk_Level, Probability_Score, Code, Description, Debug_Info_Dict)
    """
    debug_info = {}

    # --- 0. Data completeness check ---
    # Core threshold: wheeze count must be present
    if pd.isna(wheezing_freq):
        # Change: return a "missing" status instead of None so the outer loop does not skip it
        return "-", "-", "-", "core indicator (wheeze episode count) missing; cannot assess", {"error": "missing_wheezing_freq"}

    debug_info['input_wheezing_freq'] = wheezing_freq
    debug_info['input_eos_pct'] = eosinophils_pct

    # --- 1. Evaluate the primary threshold ---
    # Criterion: >= 4 wheezing episodes in the past year
    is_frequent_wheezing = (wheezing_freq >= 4)
    debug_info['criteria_primary_met'] = is_frequent_wheezing

    # If even the primary threshold is not met, classify as negative
    if not is_frequent_wheezing:
        debug_info['mAPI_result'] = 'Negative'
        return "mapi_negative", 0, "J45.3.1", "fewer than 4 wheeze episodes in the past year; mAPI negative", debug_info

    # --- 2. Count major criteria ---
    major_count = sum([parental_asthma, atopic_dermatitis, aeroallergen_sens])
    debug_info['major_criteria_count'] = major_count
    debug_info['major_details'] = f"parental asthma:{parental_asthma}, dermatitis:{atopic_dermatitis}, aeroallergen:{aeroallergen_sens}"

    # --- 3. Count minor criteria ---
    # EOS >= 4% counts as one item
    is_high_eos = 0
    if not pd.isna(eosinophils_pct):
        is_high_eos = 1 if eosinophils_pct >= 4.0 else 0

    minor_count = sum([wheezing_no_cold, is_high_eos, food_allergen_sens])
    debug_info['minor_criteria_count'] = minor_count
    debug_info[
        'minor_details'] = f"non-cold wheeze:{wheezing_no_cold}, high EOS ({eosinophils_pct}%):{is_high_eos}, food allergy:{food_allergen_sens}"

    # --- 4. Overall evaluation ---
    # Logic: primary threshold AND (>=1 major criterion OR >=2 minor criteria)
    condition_secondary = (major_count >= 1) or (minor_count >= 2)
    is_positive = condition_secondary

    debug_info['mAPI_result'] = "Positive" if is_positive else "Negative"

    if is_positive:
        return "mAPI_positive", 1, "J45.3.3", "meets_mAPI_positive_criteria", debug_info
    else:
        return "mapi_negative", 0, "J45.3.1", "does_not_meet_mAPI_positive_criteria", debug_info


# ==========================================
# 4. Main data-processing routine
# ==========================================
def process_data(base_path, item_path, output_path):
    print("--- Starting mAPI asthma prediction model ---")

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
        print("Error: data file could not be read or is empty")
        return

    # 1. Pivot to wide table
    item_df['cat'] = item_df['res_itm_name'].apply(get_category)
    filtered = item_df.dropna(subset=['cat', 'res_value']).copy()

    records = []
    # Assume the matching fields are app_no and pat_id
    # Note: assume the CSV's 'app_no' matches the required output 'app_no'
    group_cols = ['app_no', 'pat_id']

    # Check whether the column exists
    missing_cols = [c for c in group_cols if c not in filtered.columns]
    if missing_cols:
        print(f"error: item table missing required columns: {missing_cols}")
        return

    for ids, group in filtered.groupby(group_cols):
        row = {}
        # ids is a tuple (app_no, pat_id)
        row['app_no'] = str(ids[0]).strip()
        row['pat_id'] = str(ids[1]).strip()

        data_pool = {k: [] for k in FIELD_MAPPING.keys()}
        unit_pool = {k: [] for k in FIELD_MAPPING.keys()}

        for _, item in group.iterrows():
            data_pool[item['cat']].append(str(item['res_value']))
            unit_pool[item['cat']].append(item.get('res_unit', ''))

        # --- Change: safely extract wheeze frequency and EOS, read only the first value, drop np.nanmean ---
        raw_w_freq = data_pool.get('wheezing_freq', [])
        if raw_w_freq and len(raw_w_freq) > 0:
            w_freq_cleaned = clean_numeric(raw_w_freq[0])
            row['wheezing_freq'] = w_freq_cleaned if not pd.isna(w_freq_cleaned) else np.nan
        else:
            row['wheezing_freq'] = np.nan

        raw_eos = data_pool.get('eosinophils_pct', [])
        if raw_eos and len(raw_eos) > 0:
            eos_result = convert_measurement(
                raw_eos[0], unit_pool['eosinophils_pct'][0],
                indicator='eosinophils', target_unit='%'
            )
            row['eosinophils_pct'] = eos_result.value if eos_result.ok else np.nan
            row['eos_unit_note'] = eos_result.rule if eos_result.ok else eos_result.error
        else:
            row['eosinophils_pct'] = np.nan
            row['eos_unit_note'] = 'eosinophil_percentage_missing'
        # --- End of changes ---

        # Boolean indicators
        row['parental_asthma'] = parse_bool_indicator(data_pool.get('parental_asthma', []))
        row['atopic_dermatitis'] = parse_bool_indicator(data_pool.get('atopic_dermatitis', []))
        row['aeroallergen_sens'] = parse_bool_indicator(data_pool.get('aeroallergen_sens', []))
        row['wheezing_no_cold'] = parse_bool_indicator(data_pool.get('wheezing_no_cold', []))
        row['food_allergen_sens'] = parse_bool_indicator(data_pool.get('food_allergen_sens', []))

        records.append(row)

    wide_items = pd.DataFrame(records)

    if wide_items.empty:
        print("no valid indicator data extracted")
        return

    # 2. Merge basic information
    for col in group_cols:
        if col in base_df.columns:
            base_df[col] = base_df[col].astype(str).str.strip()

    final = pd.merge(wide_items, base_df, on=group_cols, how='inner')

    # 3. Run the model
    results = []

    for _, r in final.iterrows():
        w_freq = r.get('wheezing_freq', np.nan)

        risk_level, score, code, desc, details = calculate_mapi_assessment(
            wheezing_freq=w_freq,
            parental_asthma=r.get('parental_asthma', 0),
            atopic_dermatitis=r.get('atopic_dermatitis', 0),
            aeroallergen_sens=r.get('aeroallergen_sens', 0),
            wheezing_no_cold=r.get('wheezing_no_cold', 0),
            eosinophils_pct=r.get('eosinophils_pct', np.nan),
            food_allergen_sens=r.get('food_allergen_sens', 0)
        )

        # Core logic changed: a result is always returned, so the None check here should no longer skip
        if risk_level is None:
            continue

        # Build output
        out_row = {
            'model_name': 'Modified Asthma Predictive Index (mAPI)',
            'pat_id': r.get('pat_id'),
            'app_no': r.get('app_no'),  # map app_no -> app_no
            'pat_name': r.get('pat_name', 'unknown'),  # map pat_name
            'age': r.get('app_pat_age', '-'),
            'gender': r.get('pat_sex', '-'),
            'code': code,  # J45.3.1 or J45.3.3 or -
            'gauge_reference': '0|1',
            'predicted_score': score,
            'mAPI_result': details.get('mAPI_result', '-'),
            'risk_level': risk_level,
            'risk_description': desc,
            'calculation_details': str(details),
            # Keep raw key data for verification
            'raw_wheezing_freq': w_freq if pd.notna(w_freq) else 'missing',
            'raw_eos_pct': r.get('eosinophils_pct', '-') if pd.notna(r.get('eosinophils_pct')) else 'missing',
            'EOS_unit_handling': r.get('eos_unit_note', 'missing res_unit')
        }

        results.append(out_row)

    if results:
        res_df = pd.DataFrame(results)
        # Reorder columns to put key IDs first
        cols = ['pat_id', 'app_no', 'pat_name', 'code', 'risk_level', 'predicted_score', 'gauge_reference', 'model_name']
        # Fill remaining columns
        cols += [c for c in res_df.columns if c not in cols]
        res_df = res_df[cols]

        res_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"--- Assessment complete: {len(res_df)} results generated, exported to {output_path} ---")
    else:
        print("--- Assessment complete; no results generated ---")


if __name__ == "__main__":
    process_data(
        "base.csv",
        "item_result.csv",
        "mapi_prediction_result.csv"
    )
