# desc_item.py - descriptive report preprocessing
# Merge PES_RESULT_DESC (report text) with PES_apply_item (item details)
# Generate desc_item.csv for later LLM structured extraction

import pandas as pd
import os
import sys


def merge_desc_item(input_dir: str = ".", output_dir: str = "."):
    """
    Merge descriptive report data with exam item details
    
    Parameters:
        input_dir: directory containing the input files (PES_RESULT_DESC.csv and PES_apply_item.csv)
        output_dir: output file directory
    Returns:
        Output file path; returns None when there is no data
    """
    file_desc = os.path.join(input_dir, 'PES_RESULT_DESC.csv')
    file_item = os.path.join(input_dir, 'PES_apply_item.csv')
    output_file = os.path.join(output_dir, 'desc_item.csv')

    # Check whether the file exists
    if not os.path.exists(file_desc):
        print(f"WARNING: descriptive report file not found: {file_desc}; skipping preprocessing")
        return None
    if not os.path.exists(file_item):
        print(f"WARNING: item detail file not found: {file_item}; skipping preprocessing")
        return None

    # Read CSV files
    columns_desc = ['app_no', 'detail_id', 'res_result']
    columns_item = ['app_no', 'detail_id', 'pat_id', 'det_com_name', 'det_dep_name']

    try:
        df_desc = pd.read_csv(file_desc, usecols=columns_desc, dtype={'app_no': str, 'detail_id': str})
        df_item = pd.read_csv(file_item, usecols=columns_item, dtype={'app_no': str, 'detail_id': str})
    except ValueError as e:
        print(f"[FAIL] failed to read file; column mismatch: {e}")
        return None

    # Data cleaning: normalize app_no and detail_id formats
    for df in [df_desc, df_item]:
        df['app_no'] = df['app_no'].astype(str).str.strip()
        df['detail_id'] = df['detail_id'].astype(str).str.strip()
        # Strip the .0 suffix
        df['app_no'] = df['app_no'].apply(lambda x: x[:-2] if x.endswith('.0') else x)
        df['detail_id'] = df['detail_id'].apply(lambda x: x[:-2] if x.endswith('.0') else x)

    # Merge (inner join)
    merged_df = pd.merge(df_item, df_desc, on=['app_no', 'detail_id'], how='inner')

    # Drop rows with empty report content
    merged_df = merged_df.dropna(subset=['res_result'])
    merged_df = merged_df[merged_df['res_result'].str.strip() != '']

    if merged_df.empty:
        print("WARNING: no valid descriptive report data after merging")
        return None

    # Arrange column order
    final_columns = ['pat_id', 'app_no', 'detail_id', 'det_com_name', 'det_dep_name', 'res_result']
    final_df = merged_df[final_columns]

    # Output
    os.makedirs(output_dir, exist_ok=True)
    final_df.to_csv(output_file, index=False, encoding='utf-8-sig')

    print(f"[OK] descriptive reports merged: {len(final_df)} records -> {output_file}")
    return output_file


if __name__ == '__main__':
    if len(sys.argv) > 1:
        input_dir = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else input_dir
        merge_desc_item(input_dir, output_dir)
    else:
        merge_desc_item()
