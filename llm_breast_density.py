# structured_birads_density.py - LLM extraction of breast BI-RADS density
# Extract BI-RADS density grade from mammography reports

import pandas as pd
import numpy as np
import json
import re
import os
import sys
from openai import OpenAI
from tqdm import tqdm


# ==========================================
# 1. LLM configuration (read from llm_config, avoid hard-coded model names)
# ==========================================
from llm_config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, require_llm_config

_client = None


def _get_client():
    """Lazy client creation: without LLM config, error only at the actual call site; module import is unaffected."""
    global _client
    if _client is None:
        require_llm_config()
        _client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
    return _client


# ==========================================
# 2. Prompts and extraction functions
# ==========================================
SYSTEM_PROMPT = """You are a professional clinical-epidemiology data-extraction expert, focused on precisely extracting medical indicators from unstructured radiology reports.
Your current task: from the provided "breast medical imaging report" text, strictly extract the [BI-RADS breast density] category value.

[Clinical knowledge and classification rules]
- Category 1: almost entirely fatty.
- Category 2: scattered fibroglandular density.
- Category 3: heterogeneously dense.
- Category 4: extremely dense.
*Note: the 5th edition of BI-RADS uses a, b, c, d for density, equivalent to 1, 2, 3, 4 above.*

[Strict rules]
1. Never confuse: extract the overall glandular "density" grade, never the lesion assessment category (e.g. BI-RADS 3, 4a).
2. Version mapping: map a, b, c, d to 1, 2, 3, 4.
3. Modality check: for an ultrasound report without an explicit BI-RADS density grade, or when density is not mentioned, return null. Never speculate.

[Output format]
Output only valid JSON, without ```json markers, structured as:
{
  "reasoning": "brief reasoning",
  "birads_density": <1, 2, 3, 4 or null>
}"""


def extract_birads_density(report_text):
    """Call the LLM to extract BI-RADS density."""
    if pd.isna(report_text) or str(report_text).strip() == "":
        return np.nan, "empty text"

    try:
        response = _get_client().chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"[Report text]\n{report_text}"}
            ],
            temperature=0.01,
            max_tokens=2048,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}}
        )

        message_content = response.choices[0].message.content
        if message_content is None or message_content.strip() == "":
            return np.nan, "LLM returned empty content"

        result_content = message_content.strip()

        # Regex JSON extraction
        json_match = re.search(r'\{[\s\S]*\}', result_content)

        if json_match:
            data = json.loads(json_match.group(0))
            density_val = data.get("birads_density")
            reasoning = data.get("reasoning", "")

            if density_val in [1, 2, 3, 4]:
                return density_val, reasoning
            elif density_val is None or str(density_val).lower() == "null":
                return np.nan, reasoning
            else:
                return np.nan, f"invalid value: {density_val}"
        else:
            return np.nan, "no valid JSON found"

    except json.JSONDecodeError as e:
        return np.nan, f"JSON parse failed: {str(e)}"
    except Exception as e:
        return np.nan, f"extraction error: {str(e)}"


# ==========================================
# 3. Data filtering
# ==========================================
def is_mammography_exam(det_com_name):
    """Check whether this is a mammography exam"""
    if pd.isna(det_com_name):
        return False
    name_str = str(det_com_name).lower()
    return 'mammograph' in name_str or 'breast x-ray' in name_str


# ==========================================
# 4. Main processing function
# ==========================================
def run_breast_density_extraction(input_dir: str = ".", output_dir: str = "."):
    """
    Filter mammography reports from desc_item.csv and extract the BI-RADS density grade via LLM

    Parameters:
        input_dir: input directory (containing desc_item.csv)
        output_dir: output directory
    Returns:
        Extract the result DataFrame; return None when there is no data
    """
    input_path = os.path.join(input_dir, "desc_item.csv")

    print("--- Starting breast BI-RADS density extraction ---")

    if not os.path.exists(input_path):
        print(f"  WARNING: {input_path} not found; skipping breast density extraction")
        return None

    # Read data
    df = pd.DataFrame()
    for enc in ['utf-8-sig', 'gbk', 'utf-8']:
        try:
            df = pd.read_csv(input_path, encoding=enc, low_memory=False)
            break
        except:
            continue

    if df.empty:
        print(f"  WARNING: cannot read file: {input_path}")
        return None

    # Filter mammography records
    df_filtered = df[df['det_com_name'].apply(is_mammography_exam)].copy()

    print(f"  Total raw reports: {len(df)}")
    print(f"  mammography reports: {len(df_filtered)}")

    if df_filtered.empty:
        print("  WARNING: no mammography reports; skipping")
        return None

    # Call the LLM for extraction record by record
    results = []
    total = len(df_filtered)

    print("  calling the LLM for extraction...")
    with tqdm(total=total, desc="  progress", unit="reports") as pbar:
        for _, row in df_filtered.iterrows():
            pbar.set_postfix(app_no=str(row['app_no']))

            density_val, reasoning = extract_density_info(row['res_result'])

            # Unified output format: res_itm_name + res_value
            results.append({
                'app_no': str(row['app_no']).strip(),
                'pat_id': str(row['pat_id']).strip(),
                'res_itm_name': 'breast_density_birads',
                'res_value': str(int(density_val)) if pd.notna(density_val) else ''
            })

            pbar.update(1)

    if results:
        output_df = pd.DataFrame(results)
        # Drop empty rows from failed extraction
        output_df = output_df[output_df['res_value'] != '']
        print(f"  [OK] breast density extraction complete: {len(output_df)}/{total}")
        return output_df
    else:
        print("  WARNING: no valid data extracted")
        return None


if __name__ == "__main__":
    if len(sys.argv) > 1:
        input_dir = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else input_dir
        run_breast_density_extraction(input_dir, output_dir)
    else:
        run_breast_density_extraction()
