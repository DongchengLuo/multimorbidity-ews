import pandas as pd
import numpy as np
import re
import os

# ==========================================
# 1. Field configuration
# ==========================================

# (A) Input matching map: identify data from item_result.csv (exact match)
INPUT_MAPPING = {
    'freq_heartburn': ['Frequency of heartburn', 'heartburn_frequency', 'freq_hb'],
    'sev_heartburn': ['Severity of heartburn', 'heartburn_severity', 'sev_hb'],
    'freq_acidity': ['Frequency of feeling of acidity', 'acid_regurgitation_frequency', 'acid_taste_frequency', 'freq_acid_stomach'],
    'sev_acidity': ['Severity of feeling of acidity', 'acid_regurgitation_severity', 'acid_taste_severity', 'sev_acid_stomach'],
    'freq_regurgitation': ['Frequency of acid regurgitation', 'acid_reflux_frequency', 'freq_regurg'],
    'sev_regurgitation': ['Severity of acid regurgitation', 'acid_reflux_severity', 'sev_regurg'],
    'freq_antacids': ['Frequency of use of antacids', 'antacid_use_frequency', 'antacid_use', 'freq_antacid']
}

# (B) Output column map: headers for the final CSV (Chinese, LLM-readable)
OUTPUT_COLUMNS_MAP = {
    'freq_heartburn': 'heartburn_frequency (1-5)',
    'sev_heartburn': 'heartburn_severity (1-5)',
    'freq_acidity': 'acid_regurgitation_frequency (1-5)',
    'sev_acidity': 'acid_regurgitation_severity (1-5)',
    'freq_regurgitation': 'acid_reflux_frequency (1-5)',
    'sev_regurgitation': 'acid_reflux_severity (1-5)',
    'freq_antacids': 'antacid_use_frequency (1-5)'
}


# ==========================================
# 2. Basic cleaning utilities
# ==========================================
def normalize_str(s):
    if pd.isna(s): return ""
    return re.sub(r'[\(\)\（\）\[\]\【\】\{\}\s_]', '', str(s)).lower()


def get_category(name):
    """Exact-match the standard internal key from the field name"""
    norm_name = normalize_str(name)
    for key, patterns in INPUT_MAPPING.items():
        for p in patterns:
            # Change: switch to exact matching
            if normalize_str(p) == norm_name: return key
    return None


def clean_numeric(val):
    """Clean the score to ensure a value in 1-5"""
    if pd.isna(val) or str(val).strip() == "": return np.nan
    try:
        match = re.search(r"\d+", str(val))
        if match:
            num = int(match.group())
            if 1 <= num <= 5:
                return float(num)
        return np.nan
    except:
        return np.nan


# ==========================================
# 3. Core processing logic
# ==========================================
def process_data(base_path, item_path, output_path):
    print("--- Starting Chinese GERDQ screening model ---")

    # -------------------------------------------------
    # 1. Read files
    # -------------------------------------------------
    def read_csv_safe(path):
        if not os.path.exists(path):
            print(f"error: file not found {path}")
            return pd.DataFrame()
        for enc in ['utf-8-sig', 'gbk', 'utf-8']:
            try:
                return pd.read_csv(path, encoding=enc, low_memory=False)
            except:
                continue
        return pd.DataFrame()

    base_df = read_csv_safe(base_path)
    item_df = read_csv_safe(item_path)

    if base_df.empty:
        print("Error: base data file could not be read or is empty.")
        return

    # Normalize join-key format (strip spaces, leading zeros and .0 suffix)
    base_df['app_no'] = base_df['app_no'].astype(str).str.strip().str.replace(r'\.0$', '', regex=True).str.lstrip('0')
    base_df['pat_id'] = base_df['pat_id'].astype(str).str.strip().str.replace(r'\.0$', '', regex=True).str.lstrip('0')

    # -------------------------------------------------
    # 2. Extract indicators (pivot)
    # -------------------------------------------------
    print("extracting and cleaning indicators...")
    if not item_df.empty:
        item_df['app_no'] = item_df['app_no'].astype(str).str.strip().str.replace(r'\.0$', '', regex=True).str.lstrip('0')
        item_df['pat_id'] = item_df['pat_id'].astype(str).str.strip().str.replace(r'\.0$', '', regex=True).str.lstrip('0')
        
        item_df['std_key'] = item_df['res_itm_name'].apply(get_category)
        valid_rows = item_df.dropna(subset=['std_key', 'res_value']).copy()
        valid_rows['clean_score'] = valid_rows['res_value'].apply(clean_numeric)

        pivot_df = valid_rows.pivot_table(
            index=['app_no', 'pat_id'],
            columns='std_key',
            values='clean_score',
            aggfunc='max'
        ).reset_index()
    else:
        pivot_df = pd.DataFrame(columns=['app_no', 'pat_id'])

    # -------------------------------------------------
    # 3. Join basic information (left join)
    # -------------------------------------------------
    # Change: left join with base_df as the base
    final_df = pd.merge(base_df, pivot_df, on=['app_no', 'pat_id'], how='left')

    # -------------------------------------------------
    # 4. Computation and result generation
    # -------------------------------------------------
    results = []

    GAUGE_REF = "7-11|12-35"  
    REQUIRED_KEYS = list(INPUT_MAPPING.keys())  

    for _, row in final_df.iterrows():
        # A. Extract the score and validate completeness
        scores = {}
        has_missing = False
        current_sum = 0

        for key in REQUIRED_KEYS:
            val = row.get(key, np.nan)
            if pd.isna(val):
                has_missing = True
                scores[key] = "missing" # placeholder for missing input
            else:
                scores[key] = int(val)
                current_sum += val

        # B. Diagnostic logic (assign a placeholder when missing)
        if has_missing:
            total_score_str = "-"
            risk_level = "-"
            code = "-"
            suggestion = "-"
            gauge_str = "-"
        else:
            total_score_str = int(current_sum)
            gauge_str = GAUGE_REF
            if current_sum >= 12:
                risk_level = "high risk"
                code = "K21.1.3"
                suggestion = "suspected GERD; specialist evaluation advised."
            else:
                risk_level = "low risk"
                code = "K21.1.1"
                suggestion = "current likelihood of GERD is low."

        # C. Extract basic information
        age = clean_numeric(row.get('app_pat_age'))
        if pd.isna(age): age = clean_numeric(row.get('age'))

        sex = row.get('pat_sex')
        if pd.isna(sex): sex = row.get('gender', '')

        # D. Build output rows
        out_item = {
            'app_no': row['app_no'],
            'pat_id': row['pat_id'],
            'pat_name': row.get('pat_name', row.get('name', 'unknown')),
            'pat_sex': sex if pd.notna(sex) and sex != "" else "missing",
            'pat_age': int(age) if pd.notna(age) else "missing",
            'Model_Name': 'GERD GERDQ Screening',
            # --- Result items ---
            'predicted_score': total_score_str,
            'risk_level': risk_level,
            'code': code,
            'gauge_reference': gauge_str,
            'suggestion': suggestion
        }

        # E. Flatten item details into the output (Chinese column names)
        for key, val in scores.items():
            cn_col_name = OUTPUT_COLUMNS_MAP[key]
            out_item[cn_col_name] = val

        results.append(out_item)

    # -------------------------------------------------
    # 5. Save results
    # -------------------------------------------------
    if results:
        res_df = pd.DataFrame(results)

        # Define column order: ID -> results -> 7 detail columns
        cols_order = [
            'app_no', 'pat_id', 'pat_name', 'pat_sex', 'pat_age',
            'Model_Name', 'predicted_score', 'risk_level', 'code', 'gauge_reference', 'suggestion'
        ]
        cols_order.extend(OUTPUT_COLUMNS_MAP.values())

        # Drop columns not present in the DataFrame
        final_cols = [c for c in cols_order if c in res_df.columns]
        res_df = res_df[final_cols]

        res_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"--- Assessment complete: {len(res_df)} records, saved to: {output_path} ---")
    else:
        print("--- Assessment complete, no valid results ---")


if __name__ == "__main__":
    # Configure paths
    base_file = "base.csv"
    item_file = "item_result.csv"
    output_file = "gerdq_screening_result.csv"

    process_data(base_file, item_file, output_file)
