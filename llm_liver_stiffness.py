# llm_liver_stiffness.py - liver stiffness (LSM) and fat attenuation (CAP) extraction
# Extract FibroScan data from PES_SUMMARY_COMBINE.sum_summary or desc_item_sum

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
# 2. Regex extraction (preferred, fast)
# ==========================================
def extract_by_regex(text):
    """
    Try regex extraction of LSM (E value/kPa) and CAP (dB/m) from text
    Return a dict, or None when extraction fails
    """
    if not text or not isinstance(text, str):
        return None

    result = {}

    # Match E value / LSM / liver stiffness (kPa)
    # Common formats: E (kPa)5.7 / E(kPa):5.7 / LSM:5.7kPa / liver stiffness 5.7kPa
    lsm_patterns = [
        r'E[（\(]\s*kPa\s*[）\)]\s*[:：]?\s*([\d.]+)',
        r'LSM\s*[:：]?\s*([\d.]+)\s*(?:kPa)?',
        r'liver\s*stiffness\s*[:]?\s*([\d.]+)\s*(?:kPa)?',
        r'stiffness\s*[:]?\s*([\d.]+)\s*(?:kPa)?',
        r'([\d.]+)\s*kPa',
    ]

    for pattern in lsm_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                val = float(match.group(1))
                if 1.0 <= val <= 75.0:  # plausible LSM range
                    result['liver_stiffness_noninvasive'] = val
                    break
            except ValueError:
                continue

    # Match CAP (dB/m)
    # Common formats: CAP (dB/m):232 / CAP:232dB/m / fat attenuation 232
    cap_patterns = [
        r'CAP[（\(]\s*dB/m\s*[）\)]\s*[:：]?\s*([\d.]+)',
        r'CAP\s*[:：]?\s*([\d.]+)\s*(?:dB/m)?',
        r'fat\s*attenuation\s*[:]?\s*([\d.]+)',
        r'([\d.]+)\s*dB/m',
    ]

    for pattern in cap_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                val = float(match.group(1))
                if 100 <= val <= 400:  # plausible CAP range
                    result['CAP'] = val
                    break
            except ValueError:
                continue

    return result if result else None


# ==========================================
# 3. LLM extraction (fallback when regex fails)
# ==========================================
def extract_by_llm(text):
    """
    Use the LLM to extract liver stiffness and CAP values from text
    """
    prompt = f"""
    # Role
    You are a liver imaging data analyst. Your task is to extract FibroScan (transient elastography) values from the [exam report text].

    # Task
    Extract from the text below:
    1. Liver stiffness (LSM / E value), unit kPa
    2. Fat attenuation parameter (CAP), unit dB/m

    # Constraints
    1. If the text does not mention liver stiffness or FibroScan, output "none" for all fields
    2. Extract numbers only, without units
    3. If only one value is present, output "none" for the other

    # Output format (pure JSON)
    - "liver_stiffness_noninvasive": (number / "none")
    - "CAP": (number / "none")

    [Raw text]: {text}
    """

    payload = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
        "chat_template_kwargs": {"enable_thinking": False}
    }

    try:
        response = requests.post(LLM_URL, json=payload,
                                 headers={"Content-Type": "application/json"}, timeout=60)
        result = json.loads(response.json()['choices'][0]['message']['content'])
        # Clean results
        cleaned = {}
        for key in ['liver_stiffness_noninvasive', 'CAP']:
            val = result.get(key, 'none')
            if val and str(val).strip() not in ['none', 'null', '', 'None']:
                try:
                    cleaned[key] = float(val)
                except ValueError:
                    pass
        return cleaned if cleaned else None
    except Exception as e:
        # Retry after the first failure
        try:
            response = requests.post(LLM_URL, json=payload,
                                     headers={"Content-Type": "application/json"}, timeout=120)
            result = json.loads(response.json()['choices'][0]['message']['content'])
            cleaned = {}
            for key in ['liver_stiffness_noninvasive', 'CAP']:
                val = result.get(key, 'none')
                if val and str(val).strip() not in ['none', 'null', '', 'None']:
                    try:
                        cleaned[key] = float(val)
                    except ValueError:
                        pass
            return cleaned if cleaned else None
        except Exception as e2:
            print(f"  LLM extraction error: {e2}")
            return None


# ==========================================
# 4. Single-record processing logic
# ==========================================
def process_single_record(row):
    """Process a single record: regex first, LLM on failure"""
    app_no = row['app_no']
    pat_id = row.get('pat_id', '')

    # Prefer sum_summary, then res_result
    text = str(row.get('sum_summary', '') or '')
    if not text.strip():
        text = str(row.get('res_result', '') or '')

    if not text.strip():
        return []

    # Prefer regex extraction
    extracted = extract_by_regex(text)

    # Use the LLM when regex fails and the text contains relevant keywords
    if not extracted:
        keywords = ['stiffness', 'LSM', 'CAP', 'kPa', 'dB/m', 'attenuation', 'FibroScan', 'elastography']
        if any(k in text for k in keywords):
            extracted = extract_by_llm(text)

    if not extracted:
        return []

    # Convert to the standard record format
    records = []
    for key, value in extracted.items():
        records.append({
            'app_no': app_no,
            'pat_id': pat_id,
            'res_itm_name': key,
            'res_value': value
        })

    return records


# ==========================================
# 5. Main processing function
# ==========================================
LIVER_KEYWORDS = ['stiffness', 'LSM', 'CAP', 'kPa', 'dB/m', 'attenuation', 'FibroScan', 'elastography']


def run_liver_stiffness_extraction(input_dir: str = ".", output_dir: str = ".", max_workers: int = 5):
    """
    Extract liver stiffness data from desc_item_sum.csv or PES_SUMMARY_COMBINE.csv

    Parameters:
        input_dir: input directory
        output_dir: output directory
        max_workers: number of concurrent threads
    Returns:
        Extract the result DataFrame; return None when there is no data
    """
    print("--- Starting liver stiffness (LSM/CAP) extraction ---")

    # Read from desc_item_sum.csv (includes det_com_name and sum_summary)
    desc_sum_path = os.path.join(input_dir, "desc_item_sum.csv")
    summary_path = os.path.join(input_dir, "PES_SUMMARY_COMBINE.csv")

    df = None

    # Prefer filtering desc_item_sum.csv by det_com_name
    if os.path.exists(desc_sum_path):
        try:
            df_desc = pd.read_csv(desc_sum_path, encoding='utf-8-sig')
            if 'det_com_name' in df_desc.columns:
                # Filter by exam item name
                det_keywords = ['liver_stiffness_noninvasive', 'liver stiffness', 'liver elastography', 'FibroScan']
                det_pattern = '|'.join(det_keywords)
                df = df_desc[df_desc['det_com_name'].str.contains(det_pattern, na=False, case=False)]
                if not df.empty:
                    print(f"  filtered {len(df)} records from desc_item_sum.csv by det_com_name")
        except Exception:
            pass

    # Fall back to sum_summary keywords when det_com_name yields nothing
    if df is None or df.empty:
        source_path = desc_sum_path if os.path.exists(desc_sum_path) else summary_path
        if os.path.exists(source_path):
            try:
                df_src = pd.read_csv(source_path, encoding='utf-8-sig')
                col = 'sum_summary' if 'sum_summary' in df_src.columns else None
                if col:
                    pattern = '|'.join(LIVER_KEYWORDS)
                    df = df_src[df_src[col].str.contains(pattern, na=False, case=False)]
                    if not df.empty:
                        print(f"  filtered {len(df)} records from {os.path.basename(source_path)} by sum_summary keywords")
            except Exception:
                pass

    if df is None or df.empty:
        print("  no records with liver stiffness data; skipping")
        return None

    # Ensure app_no and pat_id exist
    if 'pat_id' not in df.columns:
        df['pat_id'] = ''

    all_results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_single_record, row) for _, row in df.iterrows()]
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
        print(f"  extraction complete: {len(output_df)} records (LSM: {len(output_df[output_df['res_itm_name'].str.contains('stiffness')])}, CAP: {len(output_df[output_df['res_itm_name']=='CAP'])} )")
        return output_df
    else:
        print("  No valid data extracted")
        return None


if __name__ == "__main__":
    if len(sys.argv) > 1:
        input_dir = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else input_dir
        result = run_liver_stiffness_extraction(input_dir, output_dir)
        if result is not None:
            print(result.to_string())
    else:
        run_liver_stiffness_extraction()
