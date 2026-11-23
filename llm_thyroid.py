# thyroid_llm.py - LLM extraction of thyroid C-TIRADS features
# Extract structured features from thyroid ultrasound reports

import pandas as pd
import requests
import json
import re
import os
import sys
import concurrent.futures


# ==========================================
# 1. LLM configuration (read from llm_config, avoid hard-coded model names)
# ==========================================
from llm_config import LLM_API_URL as LLM_URL, LLM_MODEL


# ==========================================
# 2. LLM information extraction
# ==========================================
def extract_thyroid_features(report_text):
    """Extract thyroid C-TIRADS features from one report via the LLM."""
    prompt = f"""
# Role
You are a rigorous thyroid ultrasound data analyst. Extract structured C-TIRADS features from thyroid ultrasound reports.

# Output
Output only one valid JSON object; no explanation, Markdown or reasoning.
The JSON must contain exactly these six fields:
"vertical_position", "solid", "markedly_hypoechoic", "irregular_margin", "microcalcification", "comet_tail_sign".

# General principles
1. Extract only information explicitly stated in the source; do not supplement with medical knowledge or defaults.
2. If the report explicitly states there is no thyroid nodule, output "no_thyroid_nodule" for all six fields.
3. If a nodule exists but a feature is not described, output "not_mentioned"; if contradictory or unreliable, output "indeterminate".
4. When multiple nodules exist, describe the one with the highest C-TIRADS suspicion (usually the largest or most suspicious).

# Field definitions
- vertical_position: "present" when the nodule is taller-than-wide (anteroposterior > transverse); "absent" when wider-than-tall; otherwise "not_mentioned".
- solid: "present" for a solid nodule; "absent" for a purely cystic or spongiform nodule; "part_solid" for mixed cystic-solid.
- markedly_hypoechoic: "present" when the nodule is markedly hypoechoic (darker than strap muscles); "absent" otherwise.
- irregular_margin: "present" for irregular, lobulated or spiculated margins; "absent" for smooth margins.
- microcalcification: "present" for punctate echogenic foci without posterior acoustic shadowing; "absent" otherwise.
- comet_tail_sign: "present" when comet-tail artifact is described; "absent" otherwise.

[Source report]
{report_text}
"""

    input_path = os.path.join(input_dir, "desc_item.csv")

    print("--- Starting thyroid C-TIRADS feature extraction ---")

    if not os.path.exists(input_path):
        print(f"  WARNING: {input_path} not found; skipping thyroid feature extraction")
        return None

    try:
        df = pd.read_csv(input_path, encoding='utf-8-sig')
    except Exception:
        df = pd.read_csv(input_path, encoding='utf-8')

    df = df.dropna(subset=['res_result'])

    # Filter thyroid ultrasound reports
    pattern = '|'.join([re.escape(k) for k in THYROID_KEYWORDS])
    df_filtered = df[df['det_com_name'].str.contains(pattern, na=False)].copy()

    print(f"  Total raw reports: {len(df)}")
    print(f"  thyroid ultrasound reports: {len(df_filtered)}")

    if df_filtered.empty:
        print("  WARNING: no thyroid ultrasound reports; skipping")
        return None

    all_results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_single_record, row) for _, row in df_filtered.iterrows()]
        for future in concurrent.futures.as_completed(futures):
            try:
                res = future.result()
                if res:
                    all_results.extend(res)
            except Exception as e:
                print(f"  Thread error: {e}")

    if all_results:
        output_df = pd.DataFrame(all_results)
        output_df = output_df.sort_values(by=['app_no', 'res_itm_name'])
        print(f"  [OK] thyroid feature extraction complete: {len(output_df)} records")
        return output_df
    else:
        print("  WARNING: no valid data extracted")
        return None


if __name__ == "__main__":
    if len(sys.argv) > 1:
        input_dir = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else input_dir
        run_thyroid_extraction(input_dir, output_dir)
    else:
        run_thyroid_extraction()
