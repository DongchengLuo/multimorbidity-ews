# desc-item-sum.py - merge descriptive reports with summary conclusions
# Merge desc_item.csv with PES_SUMMARY_COMBINE.csv to add summary conclusions

import pandas as pd
import os
import sys


def merge_desc_with_summary(input_dir: str = ".", output_dir: str = "."):
    """
    Merge desc_item.csv with PES_SUMMARY_COMBINE.csv to add the sum_summary field

    Parameters:
        input_dir: directory containing the input files
        output_dir: output file directory
    Returns:
        Output file path; returns None when there is no data
    """
    desc_item_path = os.path.join(input_dir, 'desc_item.csv')
    summary_path = os.path.join(input_dir, 'PES_SUMMARY_COMBINE.csv')
    output_path = os.path.join(output_dir, 'desc_item_sum.csv')

    # Check whether the file exists
    if not os.path.exists(desc_item_path):
        print(f"WARNING: {desc_item_path} not found; skipping merge")
        return None
    if not os.path.exists(summary_path):
        print(f"WARNING: {summary_path} not found; skipping merge")
        return None

    try:
        # Read descriptive reports
        df_desc = pd.read_csv(desc_item_path, encoding='utf-8-sig', dtype={'app_no': str, 'pat_id': str, 'detail_id': str})

        # Read the summary file, degrading gracefully by column availability
        df_sum = None
        sum_dtypes = {'app_no': str, 'pat_id': str, 'detail_id': str}

        # Attempt 1: app_no + pat_id + detail_id + sum_summary
        try:
            df_sum = pd.read_csv(summary_path, encoding='utf-8-sig', dtype=sum_dtypes,
                                 usecols=['app_no', 'pat_id', 'detail_id', 'sum_summary'])
        except ValueError:
            pass

        # Attempt 2: app_no + detail_id + sum_summary (no pat_id)
        if df_sum is None:
            try:
                df_sum = pd.read_csv(summary_path, encoding='utf-8-sig', dtype=sum_dtypes,
                                     usecols=['app_no', 'detail_id', 'sum_summary'])
            except ValueError:
                pass

        # Attempt 3: app_no + sum_summary (minimal set)
        if df_sum is None:
            try:
                df_sum = pd.read_csv(summary_path, encoding='utf-8-sig', dtype=sum_dtypes,
                                     usecols=['app_no', 'sum_summary'])
            except Exception as e:
                print(f"[FAIL] failed to read summary file: {e}")
                return None

        # Data cleaning
        for df in [df_desc, df_sum]:
            df['app_no'] = df['app_no'].astype(str).str.strip()
            df['app_no'] = df['app_no'].apply(lambda x: x[:-2] if x.endswith('.0') else x)
            if 'pat_id' in df.columns:
                df['pat_id'] = df['pat_id'].astype(str).str.strip()
                df['pat_id'] = df['pat_id'].apply(lambda x: x[:-2] if x.endswith('.0') else x)
            if 'detail_id' in df.columns:
                df['detail_id'] = df['detail_id'].astype(str).str.strip()
                df['detail_id'] = df['detail_id'].apply(lambda x: x[:-2] if x.endswith('.0') else x)

        # Determine the merge keys: columns present in both df_sum and df_desc
        possible_keys = ['app_no', 'pat_id', 'detail_id']
        merge_keys = [k for k in possible_keys if k in df_sum.columns and k in df_desc.columns]
        if not merge_keys:
            merge_keys = ['app_no']
        merged_df = pd.merge(df_desc, df_sum, on=merge_keys, how='left')

    except Exception as e:
        print(f"[FAIL] failed to merge descriptive reports with summary: {e}")
        return None

    # Output
    os.makedirs(output_dir, exist_ok=True)
    merged_df.to_csv(output_path, index=False, encoding='utf-8-sig')

    print(f"[OK] descriptive reports + summary merged: {len(merged_df)} records -> {output_path}")
    return output_path


if __name__ == '__main__':
    if len(sys.argv) > 1:
        input_dir = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else input_dir
        merge_desc_with_summary(input_dir, output_dir)
    else:
        merge_desc_with_summary()
