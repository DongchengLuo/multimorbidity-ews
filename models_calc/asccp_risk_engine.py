"""2019 ASCCP published risk tables (Egemen et al., Tables 1A-5B).

The engine performs exact lookups only. It never interpolates or invents a risk
for an undocumented combination. Special populations and unsupported inputs are
returned as blocked results for clinician/guideline review.
"""

from __future__ import annotations

import csv
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional


TABLE_PATH = Path(__file__).with_name("asccp_risk_table.csv")
PARTIAL_GENOTYPING_TABLE_PATH = Path(__file__).with_name(
    "asccp_partial_genotyping_risk_table.csv"
)
GUIDELINE_VERSION = "2019-ASCCP-Egemen-2020"
PARTIAL_GENOTYPING_VERSION = "2020-Demarco-ASCCP-Partial-Genotyping"

SCENARIO_ALIASES = {
    "initial_unknown": "initial_unknown",
    "first_screening_prior_history_unknown": "initial_unknown",
    "prior_hpv_result_unknown": "initial_unknown",
    "initial_prior_hpv_negative": "initial_prior_hpv_negative",
    "prior_hpv_negative_within_5y": "initial_prior_hpv_negative",
    "hpv_negative_past_5y": "initial_prior_hpv_negative",
    "surveillance_hpv_negative_ascus": "surveillance_hpv_negative_ascus",
    "prior_hpv_negative_asc_us_followup": "surveillance_hpv_negative_ascus",
    "surveillance_hpv_negative_lsil": "surveillance_hpv_negative_lsil",
    "prior_hpv_negative_lsil_followup": "surveillance_hpv_negative_lsil",
    "surveillance_hpv_positive_nilm": "surveillance_hpv_positive_nilm",
    "prior_hpv_positive_nilm_followup": "surveillance_hpv_positive_nilm",
    "colposcopy_biopsy_result": "colposcopy_biopsy_result",
    "colposcopy_biopsy_result": "colposcopy_biopsy_result",
    "postcolposcopy_low_grade": "postcolposcopy_low_grade",
    "followup_after_low_grade_colposcopy_referral": "postcolposcopy_low_grade",
    "postcolposcopy_high_grade": "postcolposcopy_high_grade",
    "followup_after_high_grade_colposcopy_referral": "postcolposcopy_high_grade",
    "post_treatment_first_followup": "post_treatment_first_followup",
    "first_followup_after_cin2_or_cin3_treatment": "post_treatment_first_followup",
    "post_treatment_long_term": "post_treatment_long_term",
    "long_term_followup_after_cin2_or_cin3_treatment": "post_treatment_long_term",
}

ACTION_LABELS = {
    "followup_5y": "return_to_routine_screening_after_5y",
    "followup_3y": "repeat_hpv_or_cotest_after_3y",
    "followup_1y": "repeat_hpv_or_cotest_after_1y",
    "colposcopy": "recommend_colposcopy_biopsy",
    "colposcopy_or_treatment": "colposcopy_biopsy_or_treatment; apply ASCCP high-risk rules",
    "treatment": "proceed_to_treatment_per_histology",
}


def _text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "-"} else text


def _norm(value: Any) -> str:
    return re.sub(r"[\s_（）()]+", "", _text(value)).lower()


def _float_or_none(value: Any) -> Optional[float]:
    text = _text(value)
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


@lru_cache(maxsize=1)
def load_risk_rows() -> List[Dict[str, Any]]:
    with TABLE_PATH.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        for field in ("immediate_risk", "one_year_risk", "five_year_risk"):
            row[field] = _float_or_none(row[field])
        row["negative_followup_count"] = (
            int(row["negative_followup_count"])
            if _text(row["negative_followup_count"]) else None
        )
    return rows


@lru_cache(maxsize=1)
def load_partial_genotyping_rows() -> List[Dict[str, Any]]:
    """Load Demarco Table 1; the source table contains risk, not a formula."""
    with PARTIAL_GENOTYPING_TABLE_PATH.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        for field in ("immediate_risk", "five_year_risk"):
            row[field] = _float_or_none(row[field])
        for field in ("sample_n", "cin3_cases", "source_page"):
            row[field] = int(row[field])
    return rows


def canonical_partial_genotype(value: Any) -> str:
    """Map exact partial-genotyping results to Demarco's hierarchical groups."""
    text = _text(value)
    norm = _norm(text).replace("type", "")
    if norm in {"hpv16", "16", "hpv16_positive", "16_positive"}:
        return "HPV16"
    if norm in {"hpv18", "18", "hpv18_positive", "18_positive"}:
        return "HPV18"
    if norm in {
        "hr12", "other_high_risk", "other_high_risk_positive", "other_12_high_risk_types", "other_12_high_risk_positive",
        "otherhighrisk", "otherhighriskpositive",
    }:
        return "HR12"
    # "HPV16/18 positive" may be a combined report that cannot distinguish 16 from 18; do not classify it arbitrarily.
    if "16/18" in norm or "16_and_18" in norm or "16、18" in norm:
        return "HPV16/18"
    return ""


def _risk_only_result(row: Dict[str, Any]) -> Dict[str, Any]:
    immediate = row["immediate_risk"]
    five_year = row["five_year_risk"]
    if immediate is not None and immediate >= 4.0:
        code, risk_layer = "C53.1.3", "high risk"
    elif five_year is not None and five_year >= 0.55:
        code, risk_layer = "C53.1.2", "medium risk"
    else:
        code, risk_layer = "C53.1.1", "low risk"
    return {
        "status": "computed",
        "table_id": row["source_table"],
        "guideline_version": row["guideline_version"],
        "source_page": row["source_page"],
        "immediate_risk": immediate,
        "one_year_risk": None,
        "five_year_risk": five_year,
        "risk_layer": risk_layer,
        "code": code,
    }


def lookup_partial_genotyping_risk(
    *,
    scenario: str,
    current_genotype: Any,
    current_cytology: str,
) -> Dict[str, Any]:
    """Exact lookup for Demarco Table 1 (unknown prior HPV testing only)."""
    genotype = canonical_partial_genotype(current_genotype)
    if genotype == "HPV16/18":
        return {
            "status": "not covered",
            "reason": "HPV16/18 combined positive cannot distinguish 16 from 18; source risk group cannot be uniquely matched",
        }
    if genotype not in {"HPV16", "HPV18", "HR12"}:
        return {"status": "not covered", "reason": "unrecognized_partial_genotyping_result"}
    if scenario != "initial_unknown":
        return {
            "status": "not covered",
            "reason": "Demarco Table 1 covers only first genotyping results with no known prior HPV testing",
        }
    candidates = [
        row for row in load_partial_genotyping_rows()
        if row["scenario"] == scenario
        and row["current_genotype"] == genotype
        and row["current_cytology"] == current_cytology
    ]
    if len(candidates) != 1:
        return {
            "status": "not covered",
            "reason": "Demarco Table 1 lacks this genotype-cytology combination"
            if not candidates else "duplicate match in partial-genotyping risk table",
        }
    return _risk_only_result(candidates[0])


def canonical_scenario(value: Any) -> str:
    raw = _text(value)
    if raw in SCENARIO_ALIASES:
        return SCENARIO_ALIASES[raw]
    return SCENARIO_ALIASES.get(_norm(raw), "")


def infer_scenario(explicit_scenario: Any, history: str) -> str:
    explicit = canonical_scenario(explicit_scenario)
    if explicit:
        return explicit
    mapping = {
        "unknown": "initial_unknown",
        "hpv-negative": "initial_prior_hpv_negative",
        "hpv-negative_undated": "initial_unknown",
        "hpv-negative ASC-US": "surveillance_hpv_negative_ascus",
        "hpv-negative LSIL": "surveillance_hpv_negative_lsil",
        "hpv-positive NILM": "surveillance_hpv_positive_nilm",
    }
    return mapping.get(history, "")


def canonical_cytology_for_group(cytology: str) -> str:
    if cytology in {"ASC-US", "LSIL"}:
        return "ASC-US/LSIL"
    if cytology in {"ASC-H", "AGC", "HSIL+"}:
        return "high_grade"
    return cytology


def canonical_histology(value: Any) -> str:
    s = _norm(value).replace("＜", "<")
    if not s:
        return ""
    if s in {"normal", "<cin1", "below_cin1"}:
        return "<CIN1"
    if s in {"cin1"}:
        return "CIN1"
    if s in {"<cin2", "below_cin2", "cin1_or_normal", "normal/cin1"}:
        return "<CIN2"
    if s.startswith("cin2"):
        return "CIN2"
    if s.startswith("cin3"):
        return "CIN3"
    if s == "ais" or "adenocarcinoma_in_situ" in s:
        return "AIS"
    if s in {"cancer", "cervical_cancer"}:
        return "Cancer"
    return _text(value)


def canonical_precolposcopy(value: Any) -> str:
    s = _norm(value)
    if not s:
        return ""
    if "hpv" in s and "positive" in s and "nilm" in s and ("2_times" in s or "consecutive" in s or "x2" in s):
        return "HPV-positive NILM x2"
    if "hpv-positive" in s and "nilm" in s and "x2" in s:
        return "HPV-positive NILM x2"
    if "hpv" in s and ("positive" in s or "positive" in s):
        for cytology in ("ASC-US", "LSIL"):
            if cytology.lower() in s:
                return f"HPV-positive {cytology}"
    for cytology in ("ASC-H", "AGC", "HSIL+"):
        if cytology.lower() in s:
            return cytology
    if s in {"lowgrade", "low_grade"}:
        return "low_grade"
    if s in {"highgrade", "high_grade"}:
        return "high_grade"
    return _text(value)


def canonical_followup_type(value: Any) -> str:
    s = _norm(value)
    if s in {"cotestnegative", "cotest_negative"}:
        return "cotest_negative"
    if s in {"hpvnegative", "hpv_negative"}:
        return "hpv_negative"
    return _text(value)


def _result(row: Dict[str, Any], current_cytology: str) -> Dict[str, Any]:
    action = row["recommended_management"]
    if action == "special_high_grade_after_colposcopy":
        if current_cytology == "HSIL+":
            action_label = "treatment_first; apply ASCCP high-grade cytology rules"
        else:
            action_label = "recommend colposcopy; apply ASC-H/AGC special rules"
    else:
        action_label = ACTION_LABELS.get(action, action)

    if action in {"colposcopy", "colposcopy_or_treatment", "treatment", "special_high_grade_after_colposcopy"}:
        code, risk_layer = "C53.1.3", "high risk"
    elif action == "followup_1y":
        code, risk_layer = "C53.1.2", "medium risk"
    else:
        code, risk_layer = "C53.1.1", "low risk"

    return {
        "status": "computed",
        "table_id": row["table_id"],
        "guideline_version": row["guideline_version"],
        "source_page": int(row["source_page"]),
        "immediate_risk": row["immediate_risk"],
        "one_year_risk": row["one_year_risk"],
        "five_year_risk": row["five_year_risk"],
        "recommended_management": action_label,
        "risk_layer": risk_layer,
        "code": code,
        "special_rule": row["special_rule"],
    }


def lookup_risk(
    *,
    scenario: str,
    current_hpv: str = "",
    current_cytology: str = "",
    precolposcopy_result: Any = "",
    colposcopy_histology: Any = "",
    negative_followup_count: Any = None,
    followup_test_type: Any = "",
) -> Dict[str, Any]:
    rows = [row for row in load_risk_rows() if row["scenario"] == scenario]
    if not rows:
        return {"status": "not covered", "reason": "unrecognized_ASCCP_clinical_scenario"}

    count = None
    if _text(negative_followup_count):
        try:
            count = int(float(str(negative_followup_count)))
        except ValueError:
            return {"status": "input_error", "reason": "negative_followup_count_must_be_integer"}
    followup_type = canonical_followup_type(followup_test_type)
    histology = canonical_histology(colposcopy_histology)
    precol = canonical_precolposcopy(precolposcopy_result)
    grouped_cytology = canonical_cytology_for_group(current_cytology)

    candidates = []
    for row in rows:
        if scenario in {"initial_unknown", "initial_prior_hpv_negative", "surveillance_hpv_negative_ascus"}:
            ok = row["current_hpv"] == current_hpv and row["current_cytology"] == current_cytology
        elif scenario in {"surveillance_hpv_negative_lsil", "surveillance_hpv_positive_nilm"}:
            if count is not None or followup_type:
                ok = row["negative_followup_count"] == count and row["followup_test_type"] == followup_type
            else:
                ok = row["current_hpv"] == current_hpv and row["current_cytology"] == current_cytology
        elif scenario == "colposcopy_biopsy_result":
            if histology in {"CIN2", "CIN3", "AIS", "Cancer"}:
                ok = row["colposcopy_histology"] == histology and not row["precolposcopy_result"]
            else:
                ok = row["precolposcopy_result"] == precol and row["colposcopy_histology"] == histology
        elif scenario in {"postcolposcopy_low_grade", "postcolposcopy_high_grade", "post_treatment_first_followup"}:
            ok = row["current_hpv"] == current_hpv and row["current_cytology"] == grouped_cytology
        elif scenario == "post_treatment_long_term":
            ok = row["negative_followup_count"] == count and row["followup_test_type"] == followup_type
        else:
            ok = False
        if ok:
            candidates.append(row)

    if len(candidates) != 1:
        return {
            "status": "not covered",
            "reason": "no such input combination in official Tables 1A-5B" if not candidates else "duplicate match in risk table",
        }
    return _result(candidates[0], current_cytology)


def validate_table() -> Dict[str, Any]:
    rows = load_risk_rows()
    expected_counts = {"1A": 12, "1B": 12, "2A": 12, "2B": 13, "2C": 14, "3": 16, "4A": 6, "4B": 6, "5A": 6, "5B": 4}
    actual = {table: sum(row["table_id"] == table for row in rows) for table in expected_counts}
    key_fields = (
        "scenario", "current_hpv", "current_cytology", "precolposcopy_result",
        "colposcopy_histology", "negative_followup_count", "followup_test_type",
    )
    keys = [tuple(row[field] for field in key_fields) for row in rows]
    duplicate_keys = len(keys) - len(set(keys))
    allowed_actions = {
        "followup_5y", "followup_3y", "followup_1y", "colposcopy",
        "colposcopy_or_treatment", "treatment", "special_high_grade_after_colposcopy",
    }
    invalid_actions = sorted({
        row["recommended_management"] for row in rows
        if row["recommended_management"] not in allowed_actions
    })
    invalid_risks = []
    for index, row in enumerate(rows, start=2):
        for field in ("immediate_risk", "one_year_risk", "five_year_risk"):
            value = row[field]
            if value is not None and not 0 <= value <= 100:
                invalid_risks.append({"csv_line": index, "field": field, "value": value})
    missing_provenance = sum(
        not _text(row["source_page"]) or not _text(row["guideline_version"])
        for row in rows
    )
    return {
        "row_count": len(rows),
        "counts": actual,
        "duplicate_keys": duplicate_keys,
        "invalid_actions": invalid_actions,
        "invalid_risks": invalid_risks,
        "missing_provenance": missing_provenance,
        "valid": (
            len(rows) == 101
            and actual == expected_counts
            and duplicate_keys == 0
            and not invalid_actions
            and not invalid_risks
            and missing_provenance == 0
        ),
    }


def validate_partial_genotyping_table() -> Dict[str, Any]:
    rows = load_partial_genotyping_rows()
    keys = [(row["current_genotype"], row["current_cytology"]) for row in rows]
    expected_genotypes = {"HPV16", "HPV18", "HR12"}
    expected_cytology = {"NILM", "ASC-US", "LSIL", "ASC-H", "AGC", "HSIL+"}
    invalid_risks = [
        (index, field, row[field])
        for index, row in enumerate(rows, start=2)
        for field in ("immediate_risk", "five_year_risk")
        if row[field] is None or not 0 <= row[field] <= 100
    ]
    expected_keys = {
        (genotype, cytology)
        for genotype in expected_genotypes
        for cytology in expected_cytology
    }
    valid = (
        len(rows) == 18
        and set(keys) == expected_keys
        and len(keys) == len(set(keys))
        and not invalid_risks
        and all(row["guideline_version"] == PARTIAL_GENOTYPING_VERSION for row in rows)
    )
    return {
        "row_count": len(rows),
        "duplicate_keys": len(keys) - len(set(keys)),
        "invalid_risks": invalid_risks,
        "valid": valid,
    }
