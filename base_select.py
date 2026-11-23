# base-select.py (modified)
import pandas as pd
import os
import sys


def process_patient_data(input_dir: str = ".", output_dir: str = "."):
    """
    Process patient data with configurable input/output directories
    """
    # Build the file path
    patient_file = os.path.join(input_dir, 'PES_PATIENT.csv')
    apply_file = os.path.join(input_dir, 'PES_APPLY.csv')
    output_file = os.path.join(output_dir, 'base.csv')

    try:
        print(f"Reading files: {patient_file}, {apply_file}")

        apply_cols = ['app_pat_name', 'app_pat_age', 'app_no', 'pat_id']
        patient_cols = ['pat_id', 'pat_name', 'pat_sex', 'pat_nation', 'pat_address']

        df_apply = pd.read_csv(
            apply_file,
            encoding='utf-8',
            usecols=apply_cols,
            low_memory=False,
            dtype={'pat_id': str, 'app_no': str}
        )

        df_patient = pd.read_csv(
            patient_file,
            encoding='utf-8',
            usecols=patient_cols,
            low_memory=False,
            dtype={'pat_id': str}
        )

        # Clean data
        df_apply['pat_id'] = df_apply['pat_id'].str.strip()
        df_patient['pat_id'] = df_patient['pat_id'].str.strip()

        # Merge data
        merged_df = pd.merge(df_apply, df_patient, on='pat_id', how='inner')

        # Keep rows with matching names
        merged_df['app_pat_name'] = merged_df['app_pat_name'].astype(str).str.strip()
        merged_df['pat_name'] = merged_df['pat_name'].astype(str).str.strip()
        filtered_df = merged_df[merged_df['app_pat_name'] == merged_df['pat_name']]

        # Define output columns
        target_columns = [
            'pat_name', 'app_pat_name', 'app_pat_age', 'app_no',
            'pat_id', 'pat_sex', 'pat_nation', 'pat_address'
        ]

        final_df = filtered_df[target_columns]

        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)
        final_df.to_csv(output_file, index=False, encoding='utf-8')

        print(f"[OK] done: {len(final_df)} records -> {output_file}")
        return output_file

    except FileNotFoundError as e:
        print(f"[FAIL] file not found: {e}")
        raise
    except Exception as e:
        print(f"ERROR: {e}")
        raise


if __name__ == "__main__":
    # Backward compatible: use the current directory when no argument is given
    if len(sys.argv) > 1:
        input_dir = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else input_dir
        process_patient_data(input_dir, output_dir)
    else:
        process_patient_data()
