# llm_prostate_volume.py - prostate volume extraction
# Extract prostate size from ultrasound reports and compute estimated volume
# Formula: V ~ AP diameter x transverse diameter x 0.52 (ml, inputs in cm)

import pandas as pd
import re
import os
import sys


# ==========================================
# 1. Regex extraction of prostate size
# ==========================================
def extract_prostate_dimensions(text):
    """
    Extract prostate dimensions from ultrasound report text.
    Support two or three dimensions, cm/mm; return values in cm.
    """
    if not text or not isinstance(text, str):
        return None

    # First locate the prostate-related description paragraphs
    prostate_match = re.search(r'prostate[^.]*', text, re.I)
    if not prostate_match:
        return None

    prostate_text = prostate_match.group(0)

    # Skip when the description is "normal size and shape" with no numbers
    if 'normal' in prostate_text.lower() and not re.search(r'\d', prostate_text):
        return None

    # Extend the size regex only:
    # 1) Accept phrasings such as "enlarged prostate, size about ..." and "(size ...)";
    # 2) Accept cm/mm; convert mm to cm after matching;
    # 3) Keep the original size notation directly after "prostate" as a fallback.
    patterns = [
        r'size\s*(?:about|approximately)?\s*[:]?\s*'
        r'([\d.]+)\s*[×xX]\s*([\d.]+)\s*'
        r'(?:[×xX]\s*([\d.]+)\s*)?(cm|mm)',
        r'prostate[^.;\n]*?([\d.]+)\s*[xX]\s*([\d.]+)\s*'
        r'(?:[×xX]\s*([\d.]+)\s*)?(cm|mm)',
    ]

    for pattern in patterns:
        match = re.search(pattern, prostate_text, flags=re.IGNORECASE)
        if match:
            unit = match.group(4).lower()
            scale = 0.1 if unit == 'mm' else 1.0
            d1 = round(float(match.group(1)) * scale, 10)
            d2 = round(float(match.group(2)) * scale, 10)
            d3 = round(float(match.group(3)) * scale, 10) if match.group(3) else None

            # Validate plausibility (prostate size is usually 2-8 cm)
            if 1.0 <= d1 <= 10.0 and 1.0 <= d2 <= 10.0:
                if d3 and 1.0 <= d3 <= 10.0:
                    return (d1, d2, d3)
                return (d1, d2)

    return None


def calculate_volume(dimensions):
    """
    Compute prostate volume (ml)
    Two dimensions: V = d1 x d2 x 0.52
    Three dimensions: V = d1 x d2 x d3 x pi/6
    """
    if dimensions is None:
        return None

    if len(dimensions) == 3:
        import math
        d1, d2, d3 = dimensions
        return round(d1 * d2 * d3 * math.pi / 6, 2)
    elif len(dimensions) == 2:
        d1, d2 = dimensions
        return round(d1 * d2 * 0.52, 2)

    return None


# ==========================================
# 2. Main processing function
# ==========================================
PROSTATE_KEYWORDS = ['prostate']


def run_prostate_volume_extraction(input_dir: str = ".", output_dir: str = "."):
    """Filter prostate ultrasound reports from desc_item.csv, extract dimensions and compute volume."""
    input_path = os.path.join(input_dir, "desc_item.csv")

    print("--- Starting prostate volume extraction ---")

    if not os.path.exists(input_path):
        print(f"  {input_path} not found; skipping")
        return None

    try:
        df = pd.read_csv(input_path, encoding='utf-8-sig')
    except Exception:
        df = pd.read_csv(input_path, encoding='utf-8')

    df = df.dropna(subset=['res_result'])
    df_filtered = df[df['res_result'].str.contains('prostate', case=False, na=False)].copy()

    print(f"  Total raw reports: {len(df)}")
    print(f"  reports describing the prostate: {len(df_filtered)}")

    if df_filtered.empty:
        print("  no prostate-related reports; skipping")
        return None

    all_results = []

    for _, row in df_filtered.iterrows():
        app_no = row['app_no']
        pat_id = row.get('pat_id', '')
        text = str(row['res_result'])

        dimensions = extract_prostate_dimensions(text)
        if dimensions is None:
            continue

        volume = calculate_volume(dimensions)
        if volume is None:
            continue

        all_results.append({
            'app_no': app_no,
            'pat_id': pat_id,
            'res_itm_name': 'prostate_volume',
            'res_value': volume
        })
        all_results.append({
            'app_no': app_no,
            'pat_id': pat_id,
            'res_itm_name': 'ap_diameter',
            'res_value': dimensions[0]
        })
        all_results.append({
            'app_no': app_no,
            'pat_id': pat_id,
            'res_itm_name': 'transverse_diameter',
            'res_value': dimensions[1]
        })

    if all_results:
        output_df = pd.DataFrame(all_results)
        output_df = output_df.sort_values(by=['app_no', 'res_itm_name'])
        volume_count = len(output_df[output_df['res_itm_name'] == 'prostate_volume'])
        print(f"  extraction complete: {volume_count} subjects with prostate volume")
        return output_df

    print("  no valid prostate size data extracted")
    return None


if __name__ == "__main__":
    if len(sys.argv) > 1:
        input_dir = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else input_dir
        result = run_prostate_volume_extraction(input_dir, output_dir)
        if result is not None:
            print(result.to_string())
    else:
        run_prostate_volume_extraction()
