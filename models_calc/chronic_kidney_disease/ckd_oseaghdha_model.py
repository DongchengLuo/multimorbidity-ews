import pandas as pd
import numpy as np
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model_unit_utils import convert_measurement

# ==========================================
# 1. Field mapping (exact match + decoupled logic)
# ==========================================
FIELD_MAPPING = {
    'age': ['app_pat_age', 'age'],
    'gender': ['pat_sex', 'sex'],
    # Dedicated disease-indicator columns
    'dm_specific': ['has_diabetes', 'diabetes', 'dm_history'],
    'hbp_specific': ['has_hypertension', 'hypertension', 'hbp_history'],
    # Generic fuzzy history columns (conflict source)
    'history_common': ['past_history', 'past_medical_history', 'personal_history'],
    'egfr': ['egfr', 'estimated_gfr', 'eGFR', 'gfr'],
    'scr': ['creatinine (Cr)', 'creatinine', 'serum_creatinine'],
    'acr': ['urine_albumin_creatinine_ratio', 'UACR', 'uacr', 'microalbumin_creatinine_ratio', 'albumin_creatinine'],
    'urine_protein_qual': ['urine_protein_qualitative', 'urine_albumin (ALBU)', 'urine_protein', 'proteinuria', 'dipstick']
}


# ==========================================
# 2. Basic cleaning and matching logic
# ==========================================
def normalize_str(s):
    if pd.isna(s): return ""
    return re.sub(r'[\(\)\（\）\[\]\【\】\{\}\s]', '', str(s)).lower()


def get_category(name):
    """Exact-match field classification"""
    norm_name = normalize_str(name)
    for key, patterns in FIELD_MAPPING.items():
        for p in patterns:
            if normalize_str(p) == norm_name: return key
    return None


def clean_numeric(val):
    if pd.isna(val) or str(val).strip() == "": return np.nan
    try:
        match = re.search(r"[-+]?\d*\.\d+|\d+", str(val))
        return float(match.group()) if match else np.nan
    except:
        return np.nan


def parse_sex(val):
    s = normalize_str(val)
    if 'female' in s or 'f' in s or s == '0' or '2' in s: return 0
    if 'male' in s or 'm' in s or s == '1': return 1
    return np.nan


def check_disease_status(val_list, positive_keywords, allow_generic=False):
    """Generic history must mention the disease; only dedicated fields may contain just yes/no."""
    if not val_list: return np.nan
    combined_str = "|".join([str(v) for v in val_list if pd.notna(v)]).lower()
    negative_found = False
    clauses = re.split(r'[。；;|]', combined_str)
    for keyword in [kw.strip().lower() for kw in positive_keywords]:
        if not keyword:
            continue
        for clause in clauses:
            if keyword not in clause:
                continue
            before = clause.split(keyword, 1)[0]
            tail = re.split(r'[，,、]', before)[-1]
            explicit_positive = re.search(r'(present|existing|has|diagnosed|confirmed)\s*$', tail)
            if (re.search(r'(denies|no|not|none|not found|excluded|non)\s*[^.;|]{0,20}$', before)
                    and not explicit_positive):
                negative_found = True
            else:
                return 1
    if negative_found:
        return 0
    if allow_generic:
        if any(x in combined_str for x in ['no', 'absent', 'false', 'negative', 'neg', '-']):
            return 0
        if any(x in combined_str for x in ['yes', 'present', 'true', 'positive', 'pos', '+']):
            return 1
    return np.nan


def parse_uacr_strict(val_list, unit_list):
    if not val_list: return np.nan, 'uacr_missing'
    nums = []
    notes = []
    for index, value in enumerate(val_list):
        unit = unit_list[index] if index < len(unit_list) else ''
        result = convert_measurement(
            value, unit, indicator='acr', target_unit='mg/g'
        )
        if result.ok:
            nums.append(result.value)
            notes.append(result.rule)
        else:
            notes.append(result.error)
    if nums:
        return (1 if any(n >= 30 for n in nums) else 0), '；'.join(notes)
    return np.nan, '；'.join(notes)


# ==========================================
# 3. eGFR computation and scoring logic
# ==========================================
def calculate_egfr_mdrd(scr_mgdl, age, sex):
    if any(pd.isna(x) for x in [scr_mgdl, age, sex]) or scr_mgdl <= 0: return np.nan
    # The original main analysis cites the 1999 four-variable MDRD; the FHS development cohort was white, so no Black coefficient is applied.
    female_factor = 0.742 if sex == 0 else 1.0
    return 186 * (scr_mgdl ** -1.154) * (age ** -0.203) * female_factor


def calculate_ckd_oseaghdha_score(age, has_db, has_hbp, egfr, urine_prot):
    # Strict mode: do not compute when any core indicator is missing
    if any(pd.isna(x) for x in [age, has_db, has_hbp, egfr, urine_prot]):
        return None, "-"

    score = 0
    # Age score
    age_bins = [30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
    age_scores = [0, 0, 1, 2, 3, 4, 4, 5, 6, 7, 8, 9]
    score += age_scores[np.digitize(age, age_bins)]

    score += 1 if has_db == 1 else 0
    score += 1 if has_hbp == 1 else 0

    # eGFR score
    if egfr < 75:
        score += 3
    elif 75 <= egfr <= 89:
        score += 1

    score += 1 if urine_prot == 1 else 0

    risk_map = {
        0: "0%", 1: "0%", 2: "1%", 3: "1%", 4: "2%", 5: "3%", 6: "5%",
        7: "9%", 8: "14%", 9: "20%", 10: "30%", 11: "41%", 12: "54%",
        13: "66%", 14: "76%", 15: "84%"
    }
    return score, risk_map.get(score, ">84%")


# ==========================================
# 4. Main data-processing routine
# ==========================================
def process_data(base_path, item_path, output_path):
    print(f"--- Starting CKD O'Seaghdha risk model (V2: unified history scan) ---")

    def read_csv_safe(path):
        for enc in ['gbk', 'utf-8-sig', 'utf-8']:
            try:
                return pd.read_csv(path, encoding=enc, low_memory=False)
            except:
                continue
        return pd.DataFrame()

    base_df = read_csv_safe(base_path)
    item_df = read_csv_safe(item_path)
    if base_df.empty:
        print("Base data file could not be read or is empty")
        return

    # 1. Pivot indicators to wide table
    records = []
    if not item_df.empty:
        item_df['cat'] = item_df['res_itm_name'].apply(get_category)
        filtered = item_df.dropna(subset=['cat', 'res_value']).copy()

        for (app_no, pat_id), group in filtered.groupby(['app_no', 'pat_id']):
            row = {'app_no': str(app_no).strip(), 'pat_id': str(pat_id).strip()}
            data_pool = {k: [] for k in FIELD_MAPPING.keys()}
            unit_pool = {k: [] for k in FIELD_MAPPING.keys()}
            for _, item in group.iterrows():
                data_pool[item['cat']].append(str(item['res_value']))
                unit_pool[item['cat']].append(item.get('res_unit', ''))

            egfr_unit = unit_pool['egfr'][0] if unit_pool['egfr'] else ''
            if data_pool['egfr'] and '1.73' not in str(egfr_unit):
                row['egfr_val'] = np.nan
                row['egfr_unit_note'] = 'eGFR unit not explicitly mL/min/1.73m2'
            else:
                egfr_result = convert_measurement(
                    data_pool['egfr'][0] if data_pool['egfr'] else None,
                    egfr_unit,
                    indicator='egfr', target_unit='mL/min/1.73m2'
                )
                row['egfr_val'] = egfr_result.value if egfr_result.ok else np.nan
                row['egfr_unit_note'] = egfr_result.rule if egfr_result.ok else egfr_result.error

            scr_result = convert_measurement(
                data_pool['scr'][0] if data_pool['scr'] else None,
                unit_pool['scr'][0] if unit_pool['scr'] else '',
                indicator='creatinine', target_unit='mg/dL'
            )
            row['scr_val'] = scr_result.value if scr_result.ok else np.nan
            row['scr_unit_note'] = scr_result.rule if scr_result.ok else scr_result.error

            # --- Core improvement: unified history scan ---
            # Merge all relevant columns: dm_specific, hbp_specific, history_common
            all_history_text = data_pool['dm_specific'] + data_pool['hbp_specific'] + data_pool['history_common']

            # Determine diabetes (DM)
            db_common = check_disease_status(data_pool['history_common'], ['diabetes', 'dm', 't2dm'])
            db_specific = check_disease_status(data_pool['dm_specific'], ['diabetes', 'dm', 't2dm'], True)
            row['db_flag'] = db_common if pd.notna(db_common) else db_specific
            hbp_common = check_disease_status(data_pool['history_common'], ['hypertension', 'hbp', 'htn'])
            hbp_specific = check_disease_status(data_pool['hbp_specific'], ['hypertension', 'hbp', 'htn'], True)
            row['hbp_flag'] = hbp_common if pd.notna(hbp_common) else hbp_specific

            # Proteinuria determination (UACR or urine protein dipstick)
            uacr_hit, row['uacr_unit_note'] = parse_uacr_strict(data_pool['acr'], unit_pool['acr'])
            qual_hit = check_disease_status(data_pool['urine_protein_qual'], ['positive', 'pos', '+', 'trace'])
            row['prot_flag'] = 1 if (uacr_hit == 1 or qual_hit == 1) else (
                0 if (uacr_hit == 0 or qual_hit == 0) else np.nan)

            # Keep the raw text for result-table verification
            row['raw_history'] = " | ".join(all_history_text)
            row['raw_scr'] = " | ".join(data_pool['scr'])
            row['raw_egfr'] = " | ".join(data_pool['egfr'])
            row['raw_acr_prot'] = " | ".join(data_pool['acr'] + data_pool['urine_protein_qual'])

            records.append(row)

    wide_items = pd.DataFrame(records) if records else pd.DataFrame(columns=['app_no', 'pat_id'])

    # 2. Merge basic information
    base_df['app_no'] = base_df['app_no'].astype(str).str.strip()
    base_df['pat_id'] = base_df['pat_id'].astype(str).str.strip()
    final = pd.merge(base_df, wide_items, on=['app_no', 'pat_id'], how='left')

    # 3. Loop computation results
    results = []
    GAUGE_REF = "0-3|4-6|7-15"

    for _, r in final.iterrows():
        age = clean_numeric(r.get('app_pat_age'))
        sex = parse_sex(r.get('pat_sex'))
        egfr = r.get('egfr_val')
        scr = r.get('scr_val')
        egfr_note = ""

        # Derive eGFR from creatinine when it is not reported directly
        if pd.isna(egfr) and pd.notna(scr):
            egfr = calculate_egfr_mdrd(scr, age, sex)
            egfr_note = "(derived)" if pd.notna(egfr) else ""
        egfr_unit_note = r.get('egfr_unit_note', 'missing res_unit')
        if egfr_note:
            egfr_unit_note = 'no direct eGFR; derived from Scr via the four-variable MDRD per the source'

        # Score computation
        score, risk_prob = calculate_ckd_oseaghdha_score(age, r.get('db_flag'), r.get('hbp_flag'), egfr,
                                                         r.get('prot_flag'))

        # Risk tiering
        risk_label, risk_code = ("-", "-")
        if score is not None:
            if score <= 3:
                risk_label, risk_code = "low risk", "N18.3.1"
            elif score <= 6:
                risk_label, risk_code = "medium risk", "N18.3.2"
            else:
                risk_label, risk_code = "high risk", "N18.3.3"

        results.append({
            'app_no': r.get('app_no'),
            'pat_id': r.get('pat_id'),
            'pat_name': r.get('pat_name', 'unknown'),
            'sex': r.get('pat_sex', 'missing'),
            'age': int(age) if pd.notna(age) else "missing",
            'diabetes_determination': "yes" if r.get('db_flag') == 1 else ("no" if r.get('db_flag') == 0 else "missing"),
            'hypertension_determination': "yes" if r.get('hbp_flag') == 1 else ("no" if r.get('hbp_flag') == 0 else "missing"),
            'serum_creatinine (mg/dL)': scr if pd.notna(scr) else "missing",
            'creatinine_unit_handling': r.get('scr_unit_note', 'missing res_unit'),
            'eGFR_final (mL/min/1.73m2)': f"{egfr:.1f}{egfr_note}" if pd.notna(egfr) else "missing",
            'eGFR_unit_handling': egfr_unit_note,
            'urine_protein_determination': "positive" if r.get('prot_flag') == 1 else ("negative" if r.get('prot_flag') == 0 else "missing"),
            'uacr_unit_handling': r.get('uacr_unit_note', 'uacr_missing'),
            'raw_history_summary': r.get('raw_history', 'missing'),
            'total_score': score if score is not None else "-",
            '10-year_CKD_probability': risk_prob,
            'risk_level': risk_label,
            'code': risk_code,
            'gauge_reference': GAUGE_REF if score is not None else "-",
            'model_name': 'CKD-O-Seaghdha Model'
        })

    # 4. Export
    if results:
        pd.DataFrame(results).to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"--- Assessment complete, results exported to {output_path} ---")


if __name__ == "__main__":
    process_data("base.csv", "item_result.csv", "ckd_oseaghdha_risk_N18_result.csv")
