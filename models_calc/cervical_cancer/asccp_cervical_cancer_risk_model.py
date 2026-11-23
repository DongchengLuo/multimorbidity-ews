import pandas as pd
import numpy as np
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from asccp_risk_engine import (
    GUIDELINE_VERSION,
    infer_scenario,
    lookup_partial_genotyping_risk,
    lookup_risk,
)

# ==========================================
# 1. Field mapping
# ==========================================
FIELD_MAPPING = {
    'hpv_result': ['hpv', 'hpv_test', 'hpv_result', 'human_papillomavirus'],
    'cytology_result': ['cytology', 'tct', 'pap', 'cytology_result', 'liquid_based_cytology'],
    'history': ['past_results', 'history', 'past_history', 'prior_test_result'],
    'scenario': ['asccp_scenario', 'asccp_clinical_scenario', 'cervical_screening_scenario'],
    'precolposcopy_result': ['pre_colposcopy_screening_result', 'precolposcopy_result'],
    'colposcopy_histology': ['colposcopy_biopsy_pathology', 'cervical_biopsy_result', 'colposcopy_histology'],
    'negative_followup_count': ['consecutive_negative_followups', 'negative_followup_count'],
    'followup_test_type': ['negative_followup_test_type', 'followup_test_type'],
    'hpv_genotype': ['hpv_genotype', 'hpv_partial_genotyping', 'HPV_genotype'],
    'pregnancy': ['pregnancy_status', 'pregnant', 'pregnancy'],
    'immunosuppressed': ['immunosuppression_status', 'immunosuppressed'],
    'age': ['app_pat_age', 'age'],
    'pat_name': ['name', 'patient_name'],
    'gender': ['pat_sex', 'sex']
}

# ==========================================
# 2. Basic cleaning logic
# ==========================================
def normalize_str(s):
    if pd.isna(s): return ""
    return re.sub(r'[\(\)\（\）\[\]\【\】\{\}\s]', '', str(s)).lower()


def get_category(name):
    norm_name = normalize_str(name)
    for key, patterns in FIELD_MAPPING.items():
        for p in patterns:
            if normalize_str(p) == norm_name: return key
    return None


def parse_hpv_result(val):
    """Parse HPV results only; a plain hyphen is no longer treated as negative."""
    s = normalize_str(val).replace('－', '-').replace('—', '-')
    if not s:
        return None
    negative_tokens = ['negative', 'neg', 'not_detected', 'notdetected']
    positive_tokens = ['positive', 'pos', 'detected']
    if s in {'-', '(-)'} or (s.endswith('-') and 'hpv' in s) or any(token in s for token in negative_tokens):
        return "hpv-negative"
    if s in {'+', '(+)'} or (s.endswith('+') and 'hpv' in s) or any(token in s for token in positive_tokens):
        return "hpv-positive"
    # Treat as positive when an HPV genotype is explicitly reported without negative semantics.
    if re.search(r'hpv(?:16|18|31|33|35|39|45|51|52|56|58|59|66|68)', s):
        return "hpv-positive"
    return None


def parse_cytology_result(val):
    """Parse cervical cytology categories only, preferring full terms such as ASC-US and ASC-H."""
    s = normalize_str(val).replace('－', '-').replace('—', '-')
    if not s:
        return None
    aliases = [
        ('ASC-US', ['asc-us', 'ascus', 'ASC-US']),
        ('ASC-H', ['asc-h', 'asch', 'ASC-H']),
        ('HSIL+', ['hsil', 'hsil+', 'HSIL']),
        ('LSIL', ['lsil', 'LSIL']),
        ('AGC', ['agc', 'AGC']),
        ('NILM', ['nilm', 'NILM']),
    ]
    for canonical, patterns in aliases:
        if any(pattern in s for pattern in patterns):
            return canonical
    return None


def parse_history_result(val):
    """Parse the recent prior-screening combination that determines the ASCCP scenario."""
    s = normalize_str(val)
    if not s or any(token in s for token in ['unknown', 'no_prior_result']):
        return "unknown"
    hpv = parse_hpv_result(val)
    cytology = parse_cytology_result(val)
    if hpv and cytology:
        return f"{hpv} {cytology}"
    if hpv == "hpv-negative":
        recent = any(token in s for token in ['past_5_years', 'within_5_years', 'within5y', 'within5years'])
        return "hpv-negative" if recent else "hpv-negative_undated"
    return hpv or cytology or "unknown"


def parse_positive_flag(val):
    s = normalize_str(val)
    if not s:
        return False
    return s in {'1', 'yes', 'present', 'positive', 'true', 'pregnancy', 'immunosuppression'}


def parse_hpv_genotype(val):
    """Detect partial HPV16/18 genotyping that changes management."""
    s = normalize_str(val).replace('type', '')
    if not s or any(token in s for token in ('untyped', 'unknown', 'pooled')):
        return ""
    has_16 = bool(re.search(r'(?:hpv)?16(?:positive|pos|\+)?', s))
    has_18 = bool(re.search(r'(?:hpv)?18(?:positive|pos|\+)?', s))
    if has_16 and has_18:
        return "HPV16/18"
    if has_16:
        return "HPV16"
    if has_18:
        return "HPV18"
    if 'other_high_risk' in s or 'otherhighrisk' in s:
        return "other_high_risk_type"
    return ""


def parse_age(val):
    if pd.isna(val):
        return None
    match = re.search(r'\d+(?:\.\d+)?', str(val))
    return float(match.group()) if match else None


def parse_gender_is_female(val):
    """Determine whether female: returns (is_female, raw sex string)"""
    if pd.isna(val) or str(val).strip() == "": return False, "missing"
    s = str(val).strip().lower()
    is_f = any(g in s for g in ['female', 'f']) or (s in ['2', '2.0'])
    return is_f, s


# ==========================================
# 3. Risk decision logic
# ==========================================
def get_risk_details(imm_risk, five_yr_risk):
    if pd.isna(imm_risk):
        return "-", "-", "-"

    if imm_risk >= 4.0:
        code = "C53.1.3"
        range_str = f"immediate risk ({imm_risk}%) >= 4.0%"
        rec = "treatment_preferred" if imm_risk >= 60.0 else ("treatment_or_colposcopy_biopsy" if imm_risk >= 25.0 else "colposcopy_biopsy")
    elif five_yr_risk >= 0.15:
        code = "C53.1.2"
        range_str = "5-year risk 0.15%-4.0%"
        rec = "1-year follow-up" if five_yr_risk >= 0.55 else "3-year follow-up"
    else:
        code = "C53.1.1"
        range_str = "5-year risk < 0.15%"
        rec = "return to routine screening at 5 years"

    return rec, code, range_str


# ==========================================
# 4. Main data-processing routine
# ==========================================
def process_asccp_data(base_path, item_path, output_path):
    print("--- Starting ASCCP cervical cancer management model (no-drop version) ---")

    def read_csv_safe(path):
        for enc in ['utf-8-sig', 'gb18030', 'gbk', 'utf-8']:
            try:
                return pd.read_csv(path, encoding=enc, low_memory=False)
            except:
                continue
        return pd.DataFrame()

    base_df = read_csv_safe(base_path)
    item_df = read_csv_safe(item_path)

    # 1. Wide-table processing (first valid value only)
    records = []
    if not item_df.empty:
        item_df['cat'] = item_df['res_itm_name'].apply(get_category)
        filtered = item_df.dropna(subset=['cat', 'res_value']).copy()

        for (app_no, pat_id), group in filtered.groupby(['app_no', 'pat_id']):
            row = {'app_no': str(app_no).strip(), 'pat_id': str(pat_id).strip()}
            data_pool = {k: None for k in FIELD_MAPPING.keys()}

            for _, item in group.iterrows():
                cat, val = item['cat'], str(item['res_value'])
                if data_pool[cat] is None: data_pool[cat] = val

            row['hpv'] = parse_hpv_result(data_pool['hpv_result'])
            row['hpv_genotype'] = parse_hpv_genotype(data_pool['hpv_genotype'] or data_pool['hpv_result'])
            if not row['hpv'] and row['hpv_genotype']:
                row['hpv'] = 'hpv-positive'
            row['tct'] = parse_cytology_result(data_pool['cytology_result'])
            row['hist'] = parse_history_result(data_pool['history'])
            row['gender_itm'] = data_pool['gender']
            row['scenario_raw'] = data_pool['scenario']
            row['precolposcopy_result'] = data_pool['precolposcopy_result']
            row['colposcopy_histology'] = data_pool['colposcopy_histology']
            row['negative_followup_count'] = data_pool['negative_followup_count']
            row['followup_test_type'] = data_pool['followup_test_type']
            row['pregnancy'] = data_pool['pregnancy']
            row['immunosuppressed'] = data_pool['immunosuppressed']
            records.append(row)

    # ================================================================
    # Core fix: empty-table guard
    # Create normally when records has data; otherwise create an empty table with base headers
    # ================================================================
    if records:
        wide_items = pd.DataFrame(records)
    else:
        wide_items = pd.DataFrame(columns=[
            'app_no', 'pat_id', 'hpv', 'tct', 'hist', 'gender_itm', 'scenario_raw',
            'precolposcopy_result', 'colposcopy_histology', 'negative_followup_count',
            'followup_test_type', 'pregnancy', 'immunosuppressed'
            , 'hpv_genotype'
        ])
    # ================================================================

    # 2. Left join so everyone in the base table is retained
    base_df['app_no'] = base_df['app_no'].astype(str).str.strip()
    base_df['pat_id'] = base_df['pat_id'].astype(str).str.strip()
    
    # The previous step guarantees wide_items has app_no and pat_id, so this merge cannot fail
    final = pd.merge(base_df, wide_items, left_on=['app_no', 'pat_id'], right_on=['app_no', 'pat_id'], how='left')

    # 3. Loop computation
    results = []
    for _, r in final.iterrows():
        # Sex-detection logic
        raw_gender = r.get('gender_itm') if pd.notna(r.get('gender_itm')) else r.get('pat_sex')
        is_female, gender_display = parse_gender_is_female(raw_gender)

        # Past-history fallback: use the unknown-history case from the formal table when missing.
        raw_hist = r.get('hist')
        calc_hist = raw_hist if pd.notna(raw_hist) else "unknown"
        hist_display = "unknown" if pd.isna(raw_hist) else raw_hist

        hpv = r.get('hpv') if pd.notna(r.get('hpv')) else ""
        tct = r.get('tct') if pd.notna(r.get('tct')) else ""
        hpv_genotype = r.get('hpv_genotype') if pd.notna(r.get('hpv_genotype')) else ""
        age = parse_age(r.get('app_pat_age'))
        scenario = infer_scenario(r.get('scenario_raw'), calc_hist)

        evaluation = {
            'status': 'not covered', 'reason': '', 'table_id': '',
            'guideline_version': GUIDELINE_VERSION, 'source_page': '',
            'immediate_risk': None, 'one_year_risk': None, 'five_year_risk': None,
            'risk_layer': '-', 'code': '-'
        }

        if not is_female:
            evaluation.update(status='not_applicable', reason='not applicable to non-female')
        elif age is not None and age < 25:
            evaluation.update(status='special_population_not_covered', reason='the risk-table cohort is 25-65 years; under 25 requires ASCCP-specific rules for younger women')
        elif parse_positive_flag(r.get('pregnancy')):
            evaluation.update(status='special_population_not_covered', reason='pregnancy requires ASCCP-specific pregnancy rules')
        elif parse_positive_flag(r.get('immunosuppressed')):
            evaluation.update(status='special_population_not_covered', reason='immunosuppressed individuals require dedicated guidelines')
        elif hpv_genotype:
            if hpv != 'hpv-positive':
                evaluation.update(
                    status='input_error',
                    reason='partial genotyping positive conflicts with the current HPV result',
                )
            else:
                evaluation = lookup_partial_genotyping_risk(
                    scenario=scenario,
                    current_genotype=hpv_genotype,
                    current_cytology=tct,
                )
                evaluation.setdefault('guideline_version', GUIDELINE_VERSION)
        else:
            evaluation = lookup_risk(
                scenario=scenario,
                current_hpv=hpv,
                current_cytology=tct,
                precolposcopy_result=r.get('precolposcopy_result'),
                colposcopy_histology=r.get('colposcopy_histology'),
                negative_followup_count=r.get('negative_followup_count'),
                followup_test_type=r.get('followup_test_type'),
            )
            evaluation.setdefault('guideline_version', GUIDELINE_VERSION)

        imm_risk = evaluation.get('immediate_risk')
        one_year_risk = evaluation.get('one_year_risk')
        five_yr_risk = evaluation.get('five_year_risk')
        reason = evaluation.get('reason', '')

        results.append({
            'pat_id': r['pat_id'],
            'app_no': r['app_no'],
            'name': r.get('pat_name', 'unknown'),
            'sex': gender_display,
            'age': r.get('app_pat_age', 'missing'),
            'past_history': hist_display,
            'asccp_clinical_scenario': scenario or "unrecognized",
            'current_HPV': hpv or "missing",
            'hpv_partial_genotyping': hpv_genotype or "untyped/combined_high_risk",
            'current_TCT': tct or "missing",
            'calculation_status': evaluation.get('status', 'not covered'),
            'not_computed_reason': reason,
            'risk_probability': imm_risk if imm_risk is not None else "-",
            'CIN3+ 1-year risk (%)': one_year_risk if one_year_risk is not None else "-",
            'CIN3+ 5-year risk (%)': five_yr_risk if five_yr_risk is not None else "-",
            'risk_level': evaluation.get('risk_layer', '-'),
            'code': evaluation.get('code', '-'),
            'source_table': evaluation.get('table_id', ''),
            'source_page': evaluation.get('source_page', ''),
            'guideline_version': evaluation.get('guideline_version', GUIDELINE_VERSION),
            'model_name': "ASCCP 2019 Risk-Based Guidelines"
        })

    # 4. Export
    if results:
        res_df = pd.DataFrame(results)
        res_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"--- Assessment complete: {len(res_df)} records, exported to {output_path} ---")


if __name__ == "__main__":
    process_asccp_data("base.csv", "item_result.csv", "asccp_risk_result.csv")
