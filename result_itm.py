# result_itm.py - item data merge functions
import pandas as pd
import os
import sys


def merge_item_data(input_dir: str = ".", output_dir: str = "."):
    """
    Merge item data with configurable input/output directories
    """
    file_apply_item = os.path.join(input_dir, 'PES_apply_item.csv')
    file_pes_result = os.path.join(input_dir, 'PES_RESULT.csv')
    output_path = os.path.join(output_dir, 'item_result.csv')

    os.makedirs(output_dir, exist_ok=True)

    columns_to_keep = [
        'detail_id', 'app_no', 'pat_id', 'det_com_name',
        'det_dep_name', 'res_value', 'res_unit', 'res_exp', 'res_itm_name'
    ]

    try:
        # Read data
        dtypes = {'app_no': str, 'detail_id': str}
        df_item = pd.read_csv(file_apply_item, dtype=dtypes, low_memory=False)
        df_result = pd.read_csv(file_pes_result, dtype=dtypes, low_memory=False)

        # Clean the primary key
        for df in [df_item, df_result]:
            df['app_no'] = df['app_no'].astype(str).str.strip()
            df['detail_id'] = df['detail_id'].astype(str).str.strip()
            # Strip the .0 suffix
            df['app_no'] = df['app_no'].apply(lambda x: x[:-2] if x.endswith('.0') else x)
            df['detail_id'] = df['detail_id'].apply(lambda x: x[:-2] if x.endswith('.0') else x)

        # Merge data
        merged_df = pd.merge(df_item, df_result, on=['app_no', 'detail_id'], how='inner')

        # Filter and save
        existing_cols = [col for col in columns_to_keep if col in merged_df.columns]
        final_df = merged_df[existing_cols]
        final_df.to_csv(output_path, index=False, encoding='utf-8-sig')

        print(f"[OK] matching complete: {len(final_df)} records -> {output_path}")
        return output_path

    except Exception as e:
        print(f"ERROR: {e}")
        raise


if __name__ == "__main__":
    if len(sys.argv) > 1:
        input_dir = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else input_dir
        merge_item_data(input_dir, output_dir)
    else:
        merge_item_data()