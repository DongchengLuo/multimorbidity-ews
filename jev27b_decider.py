# -*- coding: utf-8 -*-
"""
JEV-27B decision-model integration (multimorbidity primary-prevention report - knowledge-edge fitness scoring)
================================================================================
Role: plug JEV-27B-VL between "subgraph retrieval done" and "aggregate output" (a pipeline intermediate step)
Decision service: score each knowledge edge in the explanatory subgraph for "exam fit" and attach the score as an edge attribute
Embed into the subgraph (edges.json / graphml) for direct use by report generation.

Design conventions (as implemented):
  1. The full record is not LLM-condensed: basic information (sex/age etc. for physiological fit) +
     lab_items + desc_items fetched by app.attach_items_to_records (same as the UI
     the same data as the [Full Record] button), only negative-denoised and then spliced into state as-is,
     No further summarization/compression.
  2. Negative denoising (basic: exclude normal only, never include abnormal):
       - Core safety exclusion band: numeric res_itm results inside the res_exp reference range -> removed;
       - Clearly-normal characters: a res_itm qualitative result matches the normal word list exactly, or the result text contains
         clearly-normal wording -> removed.
     Note: normal-word exclusion applies only to res_itm (lab/quantitative) items, not descriptive
     Exam items (PES_RESULT_DESC, e.g. ultrasound/DR/endoscopy reports); descriptive items are always kept.
     Keep all other items (including abnormal ones such as osteopenia, stones and cysts).
  3. Edge text assembly: take all document_sources (not just the first 2),
     text_summary uses the full text (no truncation).
  4. SCORE_QUESTION is tailored to the Personal Multimorbidity Primary-Prevention and Health Management Report,
     Anchor both ends at 5 / 0 to prevent inverted scoring.
  5. Embed scoring results into the subgraph: each edge gains decision_score / decision_probabilities /
     decision_gate attribute, written back to graphml edge attributes.
  6. Concurrent scoring: JEV_DECIDE_CONCURRENCY workers (default 48; server
     --max-num-seqs 8, the rest queue server-side); a single edge's state contains full-text evidence,
     serial execution takes too long.

Called as a module by pipeline_runner.py:
  import jev27b_decider
  records = jev27b_decider.decide_records(records, progress_cb=...)

Can also run standalone (score already-persisted edges.json):
  python jev27b_decider.py <edges_json_path> [app_no]
================================================================================
"""
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import requests

try:
    import networkx as nx
except ImportError:  # pragma: no cover - networkx is installed by default
    nx = None

# ===================== Service configuration =====================
# /v1/decide is the preferred endpoint; in containerized deployments vLLM often listens on the docker bridge network,
# localhost is unreachable, so probe JEV_BASE_URLS in order and fall back to the chat API only if all fail.
JEV_DECIDE_URL = os.getenv("JEV_DECIDE_URL", "http://localhost:8080/v1/decide")
JEV_BASE_URLS = [
    u.strip().rstrip("/")
    for u in os.getenv(
        "JEV_BASE_URLS", "http://localhost:8080,http://127.0.0.1:8080"
    ).split(",") if u.strip()
]
# Per-request timeout: requests queue when concurrency exceeds the server's --max-num-seqs,
# Long edges (5k-7k tokens) take ~20-25s each; allow a longer timeout after queuing or retries will time out.
JEV_TIMEOUT = float(os.getenv("JEV_DECIDE_TIMEOUT", "300"))
JEV_MAX_RETRIES = int(os.getenv("JEV_DECIDE_MAX_RETRIES", "2"))
JEV_MODEL_NAME = os.getenv("JEV_DECIDE_MODEL", "jev-decision")
JEV_CHAT_MAX_TOKENS = int(os.getenv("JEV_CHAT_MAX_TOKENS", "32768"))
# Long contexts make the model prone to repetition degeneration (1 of 30 edges degenerated into 56 loops),
# 1.15 Removable degeneracy that does not change the decision
JEV_REPETITION_PENALTY = float(os.getenv("JEV_REPETITION_PENALTY", "1.15"))
# Concurrent scoring threads: 8 is the measured throughput knee (matches --max-num-seqs 8),
# About 31 edges/min; higher concurrency only queues and adds no throughput.
JEV_CONCURRENCY = int(os.getenv("JEV_DECIDE_CONCURRENCY", "48"))

# Explicit output-format constraint: without it the model writes long reasoning and rarely gives an explicit score,
# Relies on fallback parsing (19 of 30 measured cases used the fallback; scores are less reliable)
SCORE_OUTPUT_FORMAT = (
    "\nExplain briefly in no more than 3 sentences, then output on the last line: Final score: X (X is a number from 0 to 5)"
)

# Gating thresholds (pass / block / manual review)
PASS_THRESHOLD = float(os.getenv("JEV_PASS_THRESHOLD", "3.5"))
BLOCK_THRESHOLD = float(os.getenv("JEV_BLOCK_THRESHOLD", "2.0"))

# Negative-denoising switch (on by default; applies only to res_itm lab items)
DENOISE_LAB = os.getenv("JEV_DENOISE_LAB", "1") != "0"

# Master switch: off by default. The normal report flow skips decision scoring (too slow,
# and scores are used only for edge gating, not report quality).
# Set JEV_ENABLED=1 explicitly to enable it.
JEV_ENABLED = os.getenv("JEV_ENABLED", "0") != "0"

# Max edges scored per patient (0 = all); used to bound runtime in large exports:
# Measured ~7-9 edges/min; 138 edges take ~15-20 min per patient.
JEV_MAX_EDGES_PER_PATIENT = int(os.getenv("JEV_MAX_EDGES_PER_PATIENT", "0"))

# ===================== Scoring prompt (tailored for the multimorbidity primary-prevention report) =====================

SCORE_QUESTION = (
    "Task: assess the value of the [knowledge edge under evaluation] in supporting this patient's "
    "Personal Multimorbidity Primary-Prevention and Health-Management Report. "
    "Output a score from 0 to 5, rating the edge's actual value for individualized risk interpretation, "
    "cross-disease association, shared-risk-factor identification, and prevention/screening strategy - "
    "not whether the patient already has the disease involved in the edge.\n"

    "[Rules for using patient information]\n"
    "1. Base the judgement on the patient's actual exam results, medical history, lifestyle information and patient_risk_vector. "
    "risk_vector indicates predicted risk or priority of attention; it does not mean a confirmed disease and must not be treated as disease probability.\n"
    "2. Distinguish clearly: definitely present, definitely absent, unknown/not provided. "
    "Unknown is not the same as absent; the absence of a diagnosis does not mean there is no prevention need.\n"
    "3. If a risk factor is definitely absent, do not attribute it to the patient, "
    "but do not score the edge 0 for that reason alone; also judge whether it has other reasonable prevention, risk-association or screening value.\n"
    "4. A normal test result may reduce the edge's value for explaining current disease or existing abnormalities, "
    "but does not automatically negate its value for future risk prediction and primary prevention. "
    "Do not infer that the patient has a disease from the risk vector alone.\n"

    "[Semantic rules for knowledge edges]\n"
    "First confirm the medical proposition, relation type and direction the edge actually expresses. "
    "Prefer relation_detail, source_entity, target_entity, relation_type and text_summary in full_kg_edge_data. "
    "If source_term and target_term only indicate subgraph traversal direction, do not treat them as medical causal direction. "
    "If the actual direction differs from the displayed direction, restore the true medical proposition first; "
    "if the direction is contradictory and the true relation cannot be confirmed, lower the score or flag for manual review.\n"

    "[Scoring objective]\n"
    "Judge whether the edge reasonably supports any of the following report tasks: "
    "(1) explaining the patient's existing abnormalities, history or known risk factors; "
    "(2) explaining the risk pathway of any target disease in patient_risk_vector; "
    "(3) identifying shared risk factors or cross-disease risk hubs linking multiple diseases; "
    "(4) supporting modifiable risk-factor management, multi-disease prevention or intervention-conflict coordination; "
    "(5) supporting early screening or follow-up advice matched to the patient's age, risk and eligibility. "
    "An edge need not satisfy several objectives at once, but it must have a genuine, reasonable report use.\n"

    "[0-5 scoring criteria]\n"
    "0: completely inapplicable, an unresolvable contradiction in the medical relation, or no substantive link to the patient's existing abnormalities, "
    "any target risk disease, shared risk factors or prevention/screening strategy. "
    "The mere absence of a diagnosis, a normal test or unknown exposure information is not enough to score 0.\n"
    "1: only a very indirect general association; weak patient relevance; almost no added report value.\n"
    "2: conditionally supports a target disease's risk interpretation or general prevention, "
    "but patient specificity is limited, key exposure is unknown, or the link is indirect.\n"
    "3: substantively supports at least one report task and has a reasonable link to the patient's risk targets, exam results, "
    "history or feasible preventive measures.\n"
    "4: directly linked to an important abnormality or multiple target diseases, "
    "supporting cross-disease risk-hub identification, a clear shared prevention strategy or a targeted screening decision.\n"
    "5: key evidence for this patient's report, directly connecting multiple relevant risk targets or important abnormalities, "
    "with reliable evidence and clear eligibility, materially affecting individualized risk management or prevention decisions.\n"

    "[Scoring principles]\n"
    "The score should combine patient relevance, correctness of the medical relation, evidence reliability and actual report use. "
    "Knowledge-graph edge weight, document count or evidence level must not determine the score alone. "
    "Do not give high scores to irrelevant edges to avoid 0, and do not over-filter edges with reasonable primary-prevention value "
    "just because the patient is currently asymptomatic or undiagnosed. "
    "Judge holistically and choose the single score that best reflects the edge's overall supporting value."
)


# ===================== Negative-denoising word list =====================
# Applies only to res_itm lab items: drop qualitative results whose whole value matches (clearly normal/safe)
NORMAL_VALUE_TOKENS = {"normal", "within normal range", "essentially normal", "basically normal", "no_abnormality", "no abnormality found", "no_abnormality_seen", "no obvious abnormality", "no abnormal signs", "no abnormal echogenicity", "no definite abnormality", "no_special_findings", "unremarkable", "not seen", "not found", "no abnormality detected", "absent", "no", "negative", "(-)", "-", "—", "none", "na", "n/a", "nil", "wnl", "no deformity", "no enlargement", "not enlarged", "not palpable", "no palpable enlargement", "regular", "clear", "no tenderness", "no murmur", "not heard", "no murmur heard"}

# Clearly-normal characters: apply only to res_itm lab items; drop results containing them.
# Note: does not apply to descriptive exam items (PES_RESULT_DESC); those are always kept.
NORMAL_TEXT_KEYWORDS = [
    "no_abnormality_seen", "no obvious abnormality", "no_abnormality", "no abnormal signs", "no abnormal echogenicity",
    "no definite abnormality", "essentially normal", "no_special_findings", "unremarkable", "no mass", "no obvious mass",
    "no obvious mass lesion", "no obvious lesion", "no obvious effusion", "normal morphology", "normal echogenicity",
]

# Reference-range parsing (core safety exclusion band)
_RANGE_PATTERNS = [
    (re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*[-~—－]\s*(-?\d+(?:\.\d+)?)\s*$"), "between"),
    (re.compile(r"^\s*(?:<=|≤|＜)\s*(-?\d+(?:\.\d+)?)\s*$"), "le"),
    (re.compile(r"^\s*(?:>=|≥|＞)\s*(-?\d+(?:\.\d+)?)\s*$"), "ge"),
    (re.compile(r"^\s*(?:<|＜)\s*(-?\d+(?:\.\d+)?)\s*$"), "lt"),
    (re.compile(r"^\s*(?:>|＞)\s*(-?\d+(?:\.\d+)?)\s*$"), "gt"),
]


# ===================== Negative denoising =====================
def _to_float(text):
    try:
        return float(str(text).strip())
    except (TypeError, ValueError):
        return None


def _in_safe_band(value, res_exp):
    """Whether a numeric result falls inside the res_exp reference range (core safety exclusion band)."""
    v = _to_float(value)
    if v is None or not res_exp:
        return False
    exp = str(res_exp).strip()
    for pattern, kind in _RANGE_PATTERNS:
        m = pattern.match(exp)
        if not m:
            continue
        nums = [float(x) for x in m.groups() if x is not None]
        if kind == "between" and len(nums) == 2:
            return nums[0] <= v <= nums[1]
        if kind == "le" and nums:
            return v <= nums[0]
        if kind == "ge" and nums:
            return v >= nums[0]
        if kind == "lt" and nums:
            return v < nums[0]
        if kind == "gt" and nums:
            return v > nums[0]
    return False


def is_normal_lab_item(item):
    """Whether a res_itm lab item is negative noise (core safety exclusion band / clearly-normal characters).

    Only "exclude normal": do not judge abnormality; always keep abnormal items.
    """
    value = str(item.get("res_value", "") or "").strip()
    if not value:
        return True
    if value.lower().strip("。. ") in NORMAL_VALUE_TOKENS:
        return True
    if any(k in value for k in NORMAL_TEXT_KEYWORDS):
        return True
    if DENOISE_LAB and _in_safe_band(value, item.get("res_exp")):
        return True
    return False


def _lab_status(item):
    """Three-state lab-item classification: 'normal' / 'abnormal' / '' (undetermined).

    Used only to label items in the non-denoised record, helping the generator distinguish "tested and normal" from
    "tested and abnormal", avoiding misreading normal items as "not tested" and supporting negative reassurance.
    """
    value = str(item.get("res_value", "") or "").strip()
    if not value:
        return ""
    flag = str(item.get("res_sum_flag") or "").strip()
    if flag and any(k in flag for k in ("↑", "↓", "H", "L", "+", "positive", "high", "low")):
        return "ABNORMAL"
    if is_normal_lab_item(item):
        return "NORMAL"
    v = _to_float(value)
    if v is not None and item.get("res_exp"):
        return "ABNORMAL"
    return ""


def denoise_archive(record):
    """Negative-denoise the full record; returns (kept lab_items, kept desc_items, stats).

    Denoising rules (exclude normal only, never include abnormal):
      - res_itm lab items (PES_RESULT): core safety exclusion band (values within res_exp
        inside the reference range) + clearly-normal characters (qualitative exact match / contains normal wording) -> removed;
      - Descriptive exam items (PES_RESULT_DESC: ultrasound/DR/endoscopy etc.): normal-word exclusion does not apply
        For such items, always keep them to avoid deleting compound descriptions other than "no obvious abnormality".
    """
    lab_all = record.get("lab_items") or []
    desc_all = record.get("desc_items") or []
    lab_keep = [it for it in lab_all if not is_normal_lab_item(it)]
    desc_keep = list(desc_all)
    stats = {
        "lab_total": len(lab_all),
        "lab_kept": len(lab_keep),
        "lab_dropped": len(lab_all) - len(lab_keep),
        "desc_total": len(desc_all),
        "desc_kept": len(desc_keep),
        "desc_dropped": len(desc_all) - len(desc_keep),
    }
    return lab_keep, desc_keep, stats


# ===================== Full record text (not condensed) =====================
def fetch_patient_basics(app_no):
    """Fetch basic information (sex/age/name etc.).

    The decision model must judge physiological fit (e.g. pregnancy edges for a male, age-related risk),
    Therefore basic information must enter state together with lab details.
    """
    try:
        import app as app_module

        # Snapshot mode: read basic information directly from local CSVs
        if getattr(app_module, "SNAPSHOT_MODE", False):
            import snapshot_store

            tables = snapshot_store.fetch_tables([str(app_no).strip()])
            df_a = tables.get("PES_APPLY.csv")
            df_p = tables.get("PES_PATIENT.csv")
            if df_a is None or df_a.empty:
                return {}
            df_a = df_a.copy()
            df_a["app_no"] = df_a["app_no"].astype(str).str.strip()
            df_a = df_a[df_a["app_no"] == str(app_no).strip()]
            if df_a.empty:
                return {}
            row = df_a.iloc[0].to_dict()
            if df_p is not None and not df_p.empty and "pat_id" in row:
                df_p = df_p.copy()
                df_p["pat_id"] = df_p["pat_id"].astype(str).str.strip()
                match = df_p[df_p["pat_id"] == str(row["pat_id"]).strip()]
                if not match.empty:
                    row.update(match.iloc[0].to_dict())
            return row

        sql = """
            SELECT TOP 1
                a.app_no, a.pat_id, a.app_pat_name, a.app_pat_age,
                p.pat_name, p.pat_sex, p.pat_nation, p.pat_address
            FROM dbo.PES_APPLY a WITH (NOLOCK)
            INNER JOIN dbo.PES_PATIENT p WITH (NOLOCK) ON a.pat_id = p.pat_id
            WHERE a.app_no = %s
        """
        df = app_module.query_dataframe(sql, [str(app_no).strip()])
        if df.empty:
            return {}
        return df.to_dict(orient="records")[0]
    except Exception as e:  # pragma: no cover - degrade when the database is unreachable
        print(f"[jev27b] failed to fetch basic info app_no={app_no}: {e}")
        return {}


def fetch_patient_archive(app_no):
    """Fetch the full record for this app_no (basic info + lab details + descriptive reports)."""
    record = {}
    try:
        import app as app_module
        enriched = app_module.attach_items_to_records(
            [{"app_no": str(app_no).strip()}], max_app_nos=1
        )
        record = enriched[0] if enriched else {}
    except Exception as e:  # pragma: no cover - degrade when the database is unreachable
        print(f"[jev27b] failed to fetch full record app_no={app_no}: {e}")

    basics = fetch_patient_basics(app_no)
    if basics:
        merged = dict(basics)
        merged.update(record)   # lab_items / desc_items follow the detail results
        record = merged
    return record


def _sex_label(value):
    """Convert pat_sex to text (1=male, 2=female) so the decision model can judge physiological fit."""
    text = str(value or "").strip()
    if text == "1":
        return "Male"
    if text == "2":
        return "Female"
    return text


def build_archive_text(record, app_no=None, denoise=True):
    """Assemble the exam record into the state prefix (no LLM condensation).

    denoise=True  : keep only abnormal items. Used for JEV edge scoring (saves context).
    denoise=False : keep all tested items and label each as [normal] / [abnormal].
                    **Report generation must use False** - otherwise the model cannot see tested-normal items,
                    would misread "normal" as "not tested" and lose the basis for negative reassurance.
    """
    lab_all = record.get("lab_items") or []
    desc_all = record.get("desc_items") or []
    if denoise:
        lab_keep, desc_keep, stats = denoise_archive(record)
    else:
        lab_keep = list(lab_all)
        desc_keep = list(desc_all)
        stats = {
            "lab_total": len(lab_all),
            "lab_kept": len(lab_keep),
            "lab_dropped": 0,
            "desc_total": len(desc_all),
            "desc_kept": len(desc_keep),
            "desc_dropped": 0,
        }

    if denoise:
        lines = ["[Complete health examination archive (negative findings denoised, otherwise verbatim, no summarization)]"]
    else:
        lines = ["[Complete health examination archive (all items kept, each tested item annotated normal/abnormal)]"]

    # 1) Basic information (sex/age etc., for the decision model to judge physiological fit)
    basic_pairs = [
        ("Appointment No.", record.get("app_no")),
        ("Name", record.get("pat_name") or record.get("app_pat_name")),
        ("Sex", _sex_label(record.get("pat_sex"))),
        ("Age", record.get("app_pat_age")),
        ("Ethnicity", record.get("pat_nation")),
        ("Address", record.get("pat_address")),
    ]
    basics = [f"{label}={value}" for label, value in basic_pairs if value not in (None, "")]
    if basics:
        lines.append("1. Basic information: " + "; ".join(basics))

    # 2) Lab and exam items (with reference ranges, all retained)
    if denoise:
        lines.append(f"2. Lab and exam items (after denoising, {stats['lab_kept']}/{stats['lab_total']} kept):")
    else:
        lines.append(f"2. Lab and exam items (all {stats['lab_total']} items, each annotated):")
    if lab_keep:
        for idx, it in enumerate(lab_keep, 1):
            name = it.get("fieldName") or it.get("res_itm_name") or it.get("det_com_name") or ""
            dept = it.get("det_dep_name") or ""
            combo = it.get("det_com_name") or ""
            value = it.get("res_value") or ""
            unit = it.get("res_unit") or ""
            exp = it.get("res_exp") or ""
            head = f"{idx}. [{dept}/{combo}] {name}: {value}"
            if unit:
                head += f" {unit}"
            if exp:
                head += f" (reference range: {exp})"
            if not denoise:
                status = _lab_status(it)
                if status:
                    head += f"【{status}】"
            lines.append(head)
    else:
        lines.append("(none)")

    # 3) Descriptive reports (full text retained)
    if denoise:
        lines.append(f"3. Descriptive reports (after denoising, {stats['desc_kept']}/{stats['desc_total']} kept):")
    else:
        lines.append(f"3. Descriptive reports (all {stats['desc_total']}, full text):")
    if desc_keep:
        for idx, it in enumerate(desc_keep, 1):
            name = it.get("det_com_name") or ""
            dept = it.get("det_dep_name") or ""
            text = str(it.get("res_result") or "").strip()
            lines.append(f"{idx}. [{dept}] {name}")
            lines.append(f"    {text}")
    else:
        lines.append("(none)")

    return "\n".join(lines), stats


# ===================== Edge information text (all doc_sources + full text_summary) =====================
def build_edge_text(edge):
    """Extract a single edge's key semantics and all evidence conclusions (no truncation of documents or summaries)."""
    src = edge.get("source_term", "")
    dst = edge.get("target_term", "")
    sub = edge.get("subgraph_edge_data") or {}
    edge_type = sub.get("type", "")

    parts = [f"Relation: ({src}) -[{edge_type}]-> ({dst})"]
    meta = []
    if sub.get("weight") is not None:
        meta.append(f"subgraph weight={sub.get('weight')}")
    if sub.get("nav_weight") is not None:
        meta.append(f"navigation weight={sub.get('nav_weight')}")
    if sub.get("document_count") is not None:
        meta.append(f"documents={sub.get('document_count')}")
    if meta:
        parts.append("Subgraph attributes: " + " | ".join(meta))

    full = edge.get("full_kg_edge_data") or {}
    doc_sources = full.get("document_sources") or {}
    if isinstance(doc_sources, dict):
        doc_items = list(doc_sources.items())
    elif isinstance(doc_sources, list):
        doc_items = [(str(i), d) for i, d in enumerate(doc_sources)]
    else:
        doc_items = []

    if doc_items:
        parts.append(f"Evidence (all document_sources, {len(doc_items)} items, summaries untruncated):")
        for i, (doc_name, doc_info) in enumerate(doc_items, 1):
            doc_info = doc_info or {}
            detail = doc_info.get("relation_detail") or {}
            summary = detail.get("text_summary") or doc_info.get("text_summary") or ""
            rel_type = detail.get("relation_type") or doc_info.get("relation_type") or ""
            strength = detail.get("strength_qualitative") or ""
            title = doc_info.get("document_title") or doc_name
            block = [f"[{i}] document: {title}"]
            if rel_type:
                block.append(f"    relation type: {rel_type}")
            if strength:
                block.append(f"    evidence strength: {strength}")
            block.append(f"    text_summary: {summary if summary else '(no summary)'}")
            parts.append("\n".join(block))
    else:
        parts.append("Evidence: no additional evidence context")

    return "\n".join(parts)


# ===================== Decision-service calls =====================
def _extract_score(payload):
    """Robustly extract the score from the decide endpoint response."""
    if isinstance(payload, (int, float)):
        return float(payload), None
    if not isinstance(payload, dict):
        return None, None

    def _num(v):
        if isinstance(v, bool):
            return None
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            m = re.search(r"-?\d+(?:\.\d+)?", v)
            if m:
                return float(m.group(0))
        return None

    # Native JEV-27B /v1/decide returns:
    #   {"kind":"score","options":["0".."5"],"probabilities":[...],
    #    "choice_index":0,"choice":"0",...}
    # The score is either choice (string) or choice_index (index into options).
    if "choice" in payload:
        n = _num(payload["choice"])
        if n is not None:
            return n, payload.get("probabilities")
    if "choice_index" in payload:
        n = _num(payload["choice_index"])
        if n is not None:
            return n, payload.get("probabilities")

    for key in ("score", "value", "decision", "rating", "result"):
        if key in payload:
            n = _num(payload[key])
            if n is not None:
                return n, payload.get("probabilities")
    data = payload.get("data")
    if isinstance(data, dict):
        n, probs = _extract_score(data)
        if n is not None:
            return n, probs
    for key in ("choices", "outputs", "output"):
        v = payload.get(key)
        if isinstance(v, list) and v:
            n, probs = _extract_score(v[0])
            if n is not None:
                return n, probs
    return None, payload.get("probabilities")


# Score extraction on the chat fallback path: prefer explicit markers so indicator numbers in reasoning are not mistaken for scores
_SCORE_TEXT_PATTERNS = [
    r"final score[:]?\s*([0-5](?:\.\d+)?)",
    r"final score[:]?\s*([0-5](?:\.\d+)?)",
    r"final answer[:]?\s*([0-5](?:\.\d+)?)",
    r"[Ss]core\s*[:：=]\s*([0-5](?:\.\d+)?)",
    r"score[:]\s*([0-5](?:\.\d+)?)",
    r"rating[:]\s*([0-5](?:\.\d+)?)",
    r"give\s*([0-5](?:\.\d+)?)\s*points",
]


def _is_degenerate(text):
    """Detect repetition degeneration in output (occasional with long context, about 1/30 measured).

    Criterion: a fragment of length >=8 repeats >=5 times and the repeated region covers 30%+ of the text.
    """
    if not text or len(text) < 200:
        return False
    for m in re.finditer(r"(.{8,80}?)(?:\1){4,}", text, re.S):
        if (m.end() - m.start()) / len(text) > 0.3:
            return True
    return False


def _parse_score_from_text(text, finish_reason=None):
    """Extract the final score from chat output; return None when it cannot be determined reliably.

    Reliability contract:
      - Prefer explicit conclusion lines such as "final score / final answer";
      - If output is truncated by max_tokens (finish_reason == 'length') with no explicit conclusion line,
        Indicates the model is still reasoning; do not guess a score (return None) to avoid mistaking reasoning numbers for a conclusion.
    """
    if not text:
        return None
    if _is_degenerate(text):
        print("[jev27b] degenerate output (repetition loop); invalid, not scored")
        return None

    # 1) Explicit conclusion lines (most reliable)
    explicit = re.findall(
        r"final(?: score| answer)?[:]?\s*([0-5](?:\.\d+)?)", text
    )
    if explicit:
        return float(explicit[-1])

    # 2) Output truncated: model gave no conclusion, do not guess a score
    if finish_reason == "length":
        print("[jev27b] output truncated by max_tokens with no explicit conclusion; unscored")
        return None

    # 3) Other explicit markers
    for pat in _SCORE_TEXT_PATTERNS:
        m = re.findall(pat, text)
        if m:
            return float(m[-1])
    # 4) Fallback: accept only rows containing a single 0-5 number, scanning from the end
    for line in reversed([l for l in text.strip().splitlines() if l.strip()]):
        s = line.strip().strip("。.；;，,、")
        if re.fullmatch(r"[0-5](?:\.[05])?", s):
            return float(s)
    return None


def _decide_urls(url=None):
    """Yield /v1/decide addresses to probe in order (explicit url first)."""
    if url:
        return [url]
    seen, out = set(), []
    for base in [JEV_DECIDE_URL.rsplit("/v1/decide", 1)[0]] + JEV_BASE_URLS:
        u = f"{base}/v1/decide"
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def _score_via_chat(archive_text, edge_text, timeout):
    """Fallback when /v1/decide is unavailable: use the native vLLM chat endpoint + LoRA.

    The model reasons before concluding, so the final score must be extracted from the text.
    Returns (score, None, raw); on parse failure returns (None, None, raw).
    """
    state = f"{archive_text}\n\n[Knowledge-graph edge under evaluation]\n{edge_text}"
    prompt = f"{state}\n\n{SCORE_QUESTION}{SCORE_OUTPUT_FORMAT}"
    last_err = None
    for base in JEV_BASE_URLS:
        try:
            resp = requests.post(
                f"{base}/v1/chat/completions",
                json={
                    "model": JEV_MODEL_NAME,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": JEV_CHAT_MAX_TOKENS,
                    "temperature": 0,
                    "repetition_penalty": JEV_REPETITION_PENALTY,
                },
                timeout=timeout,
            )
            if resp.status_code != 200:
                last_err = f"HTTP {resp.status_code}: {resp.text[:200]}"
                continue
            choice = resp.json()["choices"][0]
            text = choice["message"]["content"]
            finish_reason = choice.get("finish_reason")
            score = _parse_score_from_text(text, finish_reason=finish_reason)
            raw = {
                "channel": "chat", "text": text,
                "finish_reason": finish_reason, "parsed_score": score,
            }
            if score is None:
                last_err = f"no reliable score parsed from chat output: {text[-200:]}"
                continue
            return score, None, raw
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
    print(f"[jev27b] chat fallback failed: {last_err}")
    return None, None, {"channel": "chat", "error": last_err}


def score_edge(archive_text, edge_text, url=None, timeout=None, retries=None):
    """Score a single edge with JEV-27B; returns (score, probabilities, raw).

    Prefer /v1/decide (probing multiple addresses); fall back to the chat endpoint if all fail.
    """
    timeout = JEV_TIMEOUT if timeout is None else timeout
    retries = JEV_MAX_RETRIES if retries is None else retries

    payload = {
        "kind": "score",
        "state": f"{archive_text}\n\n[Knowledge-graph edge under evaluation]\n{edge_text}",
        "question": SCORE_QUESTION,
    }

    last_err = None
    for decide_url in _decide_urls(url):
        for attempt in range(retries + 1):
            try:
                resp = requests.post(decide_url, json=payload, timeout=timeout)
                if resp.status_code == 400 and "maximum context length" in resp.text:
                    # Very long edges (all doc_sources + full text_summary) exceed the server's
                    # --max-model-len: retry and chat fallback are pointless; mark as unscored.
                    # Fix: raise the server's --max-model-len to 32768 (see README).
                    print(f"[jev27b] edge text exceeds context limit; unscored (raise --max-model-len to 32768): "
                          f"state {len(payload['state'])} chars")
                    return None, None, {
                        "channel": "decide",
                        "error": "context_overflow",
                        "state_chars": len(payload["state"]),
                        "detail": resp.text[:300],
                    }
                if resp.status_code != 200:
                    last_err = f"{decide_url} HTTP {resp.status_code}: {resp.text[:200]}"
                else:
                    try:
                        data = resp.json()
                    except ValueError:
                        last_err = f"response is not JSON: {resp.text[:200]}"
                        data = None
                    if data is not None:
                        score, probs = _extract_score(data)
                        if score is None:
                            last_err = f"no score in response: {str(data)[:200]}"
                        else:
                            return score, probs, data
            except Exception as e:
                last_err = f"{decide_url} {type(e).__name__}: {e}"
            if attempt < retries:
                time.sleep(1.0 * (attempt + 1))

    print(f"[jev27b] /v1/decide unavailable ({last_err}); falling back to chat")
    return _score_via_chat(archive_text, edge_text, max(timeout, 600.0))


def _gate(score):
    if score is None:
        return "UNSCORED"
    if score >= PASS_THRESHOLD:
        return "PASS"
    if score < BLOCK_THRESHOLD:
        return "BLOCK"
    return "REVIEW"


# ===================== Embed scoring results into the subgraph =====================
def _embed_into_edges_json(edges_path, scored, app_no, archive_stats):
    """Write scores back to edges.json (one attribute per edge + top-level decision metadata)."""
    path = Path(edges_path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    details = data.get("edge_details") or []
    for edge in details:
        key = (edge.get("source_term"), edge.get("target_term"))
        rec = scored.get(key)
        if not rec:
            continue
        score = rec.get("score")
        edge["decision_score"] = score
        edge["decision_probabilities"] = rec.get("probabilities")
        edge["decision_gate"] = _gate(score)
        edge["decision_raw"] = rec.get("raw")
        sub = edge.get("subgraph_edge_data")
        if isinstance(sub, dict):
            sub["decision_score"] = score
            sub["decision_gate"] = _gate(score)

    scored_cnt = sum(1 for e in details if e.get("decision_score") is not None)
    channels = sorted({
        ((v.get("raw") or {}).get("channel") or "decide") for v in scored.values()
    })
    data["decision_model"] = {
        "model": JEV_MODEL_NAME,
        "api": JEV_DECIDE_URL,
        "channel": ",".join(channels),
        "kind": "score",
        "question": SCORE_QUESTION,
        "archive_denoised": True,
        "archive_stats": archive_stats,
        "edge_count": len(details),
        "scored_edges": scored_cnt,
        "pass_threshold": PASS_THRESHOLD,
        "block_threshold": BLOCK_THRESHOLD,
        "finished_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return scored_cnt


def _embed_into_graphml(graphml_path, scored):
    """Sync scores back to graphml edge attributes (failure does not affect the main flow)."""
    if nx is None or not graphml_path or not os.path.exists(graphml_path):
        return False
    try:
        graph = nx.read_graphml(graphml_path)
        for u, v, data in graph.edges(data=True):
            rec = scored.get((u, v)) or scored.get((v, u))
            if not rec:
                continue
            score = rec.get("score")
            data["decision_score"] = float(score) if score is not None else -1.0
            data["decision_gate"] = _gate(score)
        nx.write_graphml(graph, graphml_path)
        return True
    except Exception as e:
        print(f"[jev27b] graphml write-back failed (edges.json unaffected): {e}")
        return False


# ===================== Pipeline entry point =====================
def _score_edges(archive_text, details, app_no="", progress_cb=None):
    """Concurrently score all edges in the subgraph; returns {(source, target): {score, probabilities, raw}}.

    Each edge's state = denoised full record + all of that edge's document_sources / full text_summary.
    """
    scored = {}
    total = len(details)
    if total == 0:
        return scored

    def work(edge):
        edge_text = build_edge_text(edge)
        score, probs, raw = score_edge(archive_text, edge_text)
        return edge, score, probs, raw

    workers = max(1, min(JEV_CONCURRENCY, total))
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(work, edge) for edge in details]
        for fut in as_completed(futures):
            try:
                edge, score, probs, raw = fut.result()
            except Exception as e:
                print(f"[jev27b] single-edge scoring error: {e}")
                continue
            scored[(edge.get("source_term"), edge.get("target_term"))] = {
                "score": score, "probabilities": probs, "raw": raw,
            }
            done += 1
            if progress_cb:
                try:
                    progress_cb(done, total, app_no, edge, score)
                except Exception:
                    pass
    return scored


def decide_subgraph(record, progress_cb=None):
    """Score a single patient's subgraph and embed the scores into the subgraph file."""
    app_no = str(record.get("eid") or "")
    files = record.get("subgraph_files") or {}
    edges_path = files.get("edges_json")
    if not edges_path or not os.path.exists(edges_path):
        return record

    with open(edges_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    details = data.get("edge_details") or []
    if not details:
        return record

    # 1) Full record (kept as-is after negative denoising, not condensed)
    archive_record = fetch_patient_archive(app_no)
    archive_text, archive_stats = build_archive_text(archive_record, app_no)
    if not archive_record:
        record["decision_status"] = "NO_ARCHIVE"
        return record

    # 2) Score edges one by one (concurrent; server max-num-seqs allows parallel requests)
    #    Optional: score only the top-N edges by subgraph weight (JEV_MAX_EDGES_PER_PATIENT)
    target_edges = details
    skipped = 0
    if JEV_MAX_EDGES_PER_PATIENT > 0 and len(details) > JEV_MAX_EDGES_PER_PATIENT:
        target_edges = sorted(
            details,
            key=lambda e: (e.get("subgraph_edge_data") or {}).get("weight") or 0,
            reverse=True,
        )[:JEV_MAX_EDGES_PER_PATIENT]
        skipped = len(details) - len(target_edges)
        print(f"[jev27b] {app_no}: scoring only the top {len(target_edges)} edges by weight, skipping {skipped}")

    scored = _score_edges(archive_text, target_edges, app_no=app_no, progress_cb=progress_cb)

    # 3) Embed subgraph
    scored_cnt = _embed_into_edges_json(edges_path, scored, app_no, archive_stats)
    _embed_into_graphml(files.get("graphml"), scored)

    gates = [_gate(v["score"]) for v in scored.values()]
    channels = sorted({
        ((v.get("raw") or {}).get("channel") or "decide") for v in scored.values()
    })
    record["decision_status"] = "SUCCESS"
    record["decision_model"] = {
        "model": JEV_MODEL_NAME,
        "api": JEV_DECIDE_URL,
        "channel": ",".join(channels),
        "question": SCORE_QUESTION,
        "archive_stats": archive_stats,
        "scored_edges": scored_cnt,
        "skipped_edges": skipped,
        "pass": gates.count("PASS"),
        "review": gates.count("REVIEW"),
        "block": gates.count("BLOCK"),
        "unscored": gates.count("UNSCORED"),
        "pass_threshold": PASS_THRESHOLD,
        "block_threshold": BLOCK_THRESHOLD,
    }
    return record


def decide_records(records, progress_cb=None):
    """Batch-score successfully retrieved patient subgraphs (a pipeline intermediate step)."""
    if not JEV_ENABLED:
        print("[jev27b] JEV_ENABLED=0; skipping decision scoring")
        for rec in records:
            rec.setdefault("decision_status", "DISABLED")
        return records

    out = []
    for rec in records:
        if rec.get("status") != "SUCCESS":
            out.append(rec)
            continue
        try:
            rec = decide_subgraph(rec, progress_cb=progress_cb)
        except Exception as e:
            rec["decision_status"] = f"ERROR: {e}"
            print(f"[jev27b] decision scoring error eid={rec.get('eid')}: {e}")
        out.append(rec)
    return out


# ===================== Standalone entry point =====================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python jev27b_decider.py <edges_json_path> [app_no]")
        sys.exit(1)

    edges_json = sys.argv[1]
    eid = sys.argv[2] if len(sys.argv) > 2 else Path(edges_json).name.replace("_edges.json", "").replace("eid_", "")
    rec = {"eid": eid, "status": "SUCCESS", "subgraph_files": {"edges_json": edges_json}}

    def _cli_progress(done, total, app_no, edge, score):
        print(f"[{done}/{total}] ({edge.get('source_term')}) -> ({edge.get('target_term')}) "
              f"[{_gate(score)}] score: {score}")

    decide_subgraph(rec, progress_cb=_cli_progress)
    print(json.dumps(rec.get("decision_model", {}), ensure_ascii=False, indent=2))
