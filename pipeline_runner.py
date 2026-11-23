# -*- coding: utf-8 -*-
"""
Export pipeline (integrated core module)
================================================================================
After the UI selects app_no for export, this module runs the full chain in a background thread:

  Stage 1 [risk scoring]  calls cli.py's core BatchProcessor.process_batch,
                     Run multi-disease risk model computation + stacking score computation;
  Stage 2 [risk-tier assembly] multiple scoring models for one disease are first unified into one disease-level
                     Risk tiering (majority vote + higher tier on ties + high-risk veto),
                     mapped to English standard disease names via disease_name_mapping.json,
                     Assemble into the patient risk vector {English standard disease name: RWR weight};
                     Only diseases at or above medium risk (unified tier) enter subgraph retrieval
                     (see build_patient_vectors); stacking values are deprecated,
                     kept in the output for manual verification only;
  Stage 3 [subgraph retrieval]  calls subgraph_construct + hybrid_retrieval_engine,
                     Run patient-weighted RWR dynamic graph retrieval for each patient
                     (Fast Lane steady-state comorbidity module matching is disabled),
                     Persist full subgraph data (graphml / edges.json / png);
  Stage 3.5 [decision scoring] calls jev27b_decider with the negative-denoised full record (not condensed)
                     Concatenated with all document_sources / full text_summary and sent to
                     JEV-27B-VL decision service (POST /v1/decide, kind=score),
                     Score edges one by one with questions tailored to the Personal Multimorbidity Primary-Prevention and Health Management Report,
                     and embed decision_score as an edge attribute in the subgraph;
  Stage 4 [aggregate output]  pipeline_result_{task_id}.json and
                     stacking_summary_{task_id}.csv is written to debug_output/.

Debug convention: all retrieval results in this stage go to debug_output/ under the project root.
================================================================================
"""
import glob
import json
import os
import threading
import time
import traceback
from collections import Counter
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DEBUG_OUTPUT_DIR = PROJECT_ROOT / "debug_output"
DISEASE_MAPPING_FILE = PROJECT_ROOT / "disease_name_mapping.json"

MAX_EXPORT_APP_NOS = 200          # per-export cap to prevent accidental bulk runs
MAX_LOG_LINES = 200               # retained log lines per task
MAX_TASKS = 50                    # retained entries in the task registry

# Disease admission line for subgraph retrieval: risk tier priority >= 2 (medium/high/very high)
# Same scale as batch_processor.risk_level_priority (2=medium, 3=high, 4=very high)
MIN_RETRIEVAL_RISK_PRIORITY = 2

# ===================== Subgraph evidence compression (sub.py) =====================
# Keep all edges, compress only the document_sources under each edge; arguments match the sub.py CLI.
SUB_DIRECT_DOCS = int(os.getenv("SUB_DIRECT_DOCS", "3"))
SUB_NEAR_DOCS = int(os.getenv("SUB_NEAR_DOCS", "2"))
SUB_BACKGROUND_DOCS = int(os.getenv("SUB_BACKGROUND_DOCS", "1"))
SUB_MAX_EXTRA_CRITICAL = int(os.getenv("SUB_MAX_EXTRA_CRITICAL", "3"))
SUB_MAX_SUMMARY_CHARS = int(os.getenv("SUB_MAX_SUMMARY_CHARS", "0"))

# ===================== Task registry =====================
TASKS = {}
TASKS_ORDER = []
_LOCK = threading.Lock()
_PIPELINE_LOCK = threading.Lock()   # serialize pipeline runs to protect the singleton engine

# ===================== Singleton =====================
_BP = None
_DISEASE_MAPPING = None


def _load_disease_mapping():
    global _DISEASE_MAPPING
    if _DISEASE_MAPPING is None:
        with open(DISEASE_MAPPING_FILE, encoding="utf-8") as f:
            _DISEASE_MAPPING = json.load(f).get("mapping", {})
    return _DISEASE_MAPPING


def _get_batch_processor():
    """BatchProcessor singleton (the risk-scoring core in cli.py)."""
    global _BP
    if _BP is None:
        from batch_processor import BatchProcessor
        print("[pipeline_runner] Initializing BatchProcessor (risk scoring + stacking)...")
        _BP = BatchProcessor()
    return _BP


# ===================== Task management =====================
def create_task(app_nos, source="search", enable_decide=False, enable_report=True):
    """Create an export task and return the task dict (not yet started)."""
    task_id = f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.urandom(3).hex()}"
    task = {
        "task_id": task_id,
        "status": "queued",           # queued / running / done / error
        "stage": "queued",
        "progress": 0,
        "message": "Task created, waiting to run",
        "app_nos": list(app_nos),
        "source": source,
        "enable_decide": bool(enable_decide),
        "enable_report": bool(enable_report),
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "finished_at": None,
        "log": [],
        "result": None,
        "error": None,
    }
    with _LOCK:
        TASKS[task_id] = task
        TASKS_ORDER.append(task_id)
        while len(TASKS_ORDER) > MAX_TASKS:
            old = TASKS_ORDER.pop(0)
            TASKS.pop(old, None)
    return task


def get_task(task_id):
    with _LOCK:
        return dict(TASKS.get(task_id)) if task_id in TASKS else None


def list_tasks():
    with _LOCK:
        return [dict(TASKS[tid]) for tid in reversed(TASKS_ORDER) if tid in TASKS]


def _update(task_id, **fields):
    with _LOCK:
        task = TASKS.get(task_id)
        if not task:
            return
        task.update(fields)
        msg = fields.get("message")
        if msg:
            stamp = datetime.now().strftime("%H:%M:%S")
            task["log"].append(f"[{stamp}] {msg}")
            if len(task["log"]) > MAX_LOG_LINES:
                task["log"] = task["log"][-MAX_LOG_LINES:]


def _status_class(status):
    """Retrieval status -> UI badge style class."""
    if not status:
        return "warn"
    if status.startswith("SUCCESS"):
        return "ok"
    if status in ("ERROR", "INCOHERENT"):
        return "error"
    return "warn"


# ===================== Stage 2: risk tiering -> patient risk vector =====================
#
# Since 2026-10-20, the retrieval trigger signal switched from stacking values to the risk tier.
# Reason: 99% of stacking results involve imputation (2136 of 2162); models such as thyroid cancer
# used=0 still outputs a constant; input scales mix percent/fraction/raw; per-disease baseline
# Large differences (COPD 0.0030 vs breast cancer 0.0074); absolute values are not comparable across diseases.
#
# New approach:
#   1. Trigger signal = disease-level unified tier (ordinal, naturally comparable across diseases)
#   2. RWR weight = tier base weight x vote-support coefficient (votes refine granularity)
#   3. Stacking values no longer enter the vector; kept in output for manual verification only
#   4. Fast Lane (steady-state comorbidity module matching) disabled; everything uses Smart Lane dynamic retrieval

# Tier base weights: medium / high / very high risk
RISK_TIER_WEIGHT = {
    2: 0.50,
    3: 0.75,
    4: 1.00,
}

# Vote-support coefficient range: support 0 -> 0.80, support 1.0 -> 1.00
# Upper bound 1.0 keeps tiers disjoint: medium max 0.50 < high min 0.60
SUPPORT_FACTOR_FLOOR = 0.80
SUPPORT_FACTOR_SPAN = 0.20

# Tier line that triggers high-risk veto protection: any model reaching high risk or above
HIGH_RISK_PROTECTION_PRIORITY = 3

# Tier / source labels (for CSV and UI display)
_TIER_LABEL = {0: "No valid tier", 1: "Low Risk", 2: "Medium Risk", 3: "High Risk", 4: "Very High Risk"}
_SOURCE_LABEL = {"vote": "Vote", "protect": "Veto protection", "none": "-"}


def _risk_priority(risk_level) -> int:
    """Risk-tier priority (reuses batch_processor's rules; higher value = higher risk)."""
    from batch_processor import risk_level_priority
    return risk_level_priority(risk_level)


def _unify_disease_risk(model_results) -> dict:
    """
    Unified tiering across multiple risk models for one disease + vote-refined weights.

    Tiering rules (same source as batch_processor._resolve_disease_risk, so the UI display
    consistent with the retrieval trigger):
      1. Count only models with a valid tier (priority > 0); "continuous risk" and "-" have no threshold
         output does not participate in voting;
      2. Majority vote; on a tie take the higher tier;
      3. High-risk veto: if the vote is below medium risk but any model reaches
         high/very high risk, raise that disease's tier to medium (do not adopt high risk directly,
         Only ensure nothing is missed - a miss loses an entire block of subgraph evidence, while over-inclusion just adds one
         low-weight seed nodes).

    Weight rules:
      weight = RISK_TIER_WEIGHT[tier] x (0.80 + 0.20 x support rate)
      Support rate = votes for the final tier / models with a valid tier.
      For veto cases, the support count is the vote count of the high-risk model that triggered it.

    Returns:
        {tier, n_valid, n_support, source, weight}
        tier:    0=no valid tier 1=low 2=medium 3=high 4=very high
        source:  'vote'=by vote / 'protect'=raised by veto protection / 'none'=no valid tier
        weight:  RWR weight (None when tier < 2; not used for retrieval)
    """
    prios = [_risk_priority(m.get("risk_level")) for m in model_results or []]
    valid = [p for p in prios if p > 0]
    n_valid = len(valid)

    if n_valid == 0:
        return {"tier": 0, "n_valid": 0, "n_support": 0, "source": "none", "weight": None}

    counter = Counter(valid)
    vote_tier = sorted(counter.keys(), key=lambda l: (counter[l], l), reverse=True)[0]
    n_support = counter[vote_tier]
    source = "vote"
    tier = vote_tier

    if vote_tier < MIN_RETRIEVAL_RISK_PRIORITY:
        n_high = sum(1 for p in valid if p >= HIGH_RISK_PROTECTION_PRIORITY)
        if n_high > 0:
            tier = MIN_RETRIEVAL_RISK_PRIORITY
            n_support = n_high
            source = "protect"

    if tier not in RISK_TIER_WEIGHT:
        return {"tier": tier, "n_valid": n_valid, "n_support": n_support,
                "source": source, "weight": None}

    ratio = n_support / n_valid
    factor = SUPPORT_FACTOR_FLOOR + SUPPORT_FACTOR_SPAN * ratio
    weight = round(RISK_TIER_WEIGHT[tier] * factor, 4)

    return {"tier": tier, "n_valid": n_valid, "n_support": n_support,
            "source": source, "weight": weight}


def _parse_risk_5y(stacking_item):
    """Extract the 5-year absolute risk (0-1) from a stacking result. Output verification only; does not drive retrieval."""
    risk = stacking_item.get("risk_5y")
    if risk is not None:
        try:
            return float(risk)
        except (TypeError, ValueError):
            pass
    pct = stacking_item.get("risk_5y_percent")
    if pct is not None:
        try:
            return float(str(pct).replace("%", "").strip()) / 100.0
        except (TypeError, ValueError):
            pass
    return None


def _resolve_batch_dir(bp):
    """Get the batch directory used by this pipeline; fall back to the latest batch containing batch_results.json."""
    d = getattr(bp, "last_batch_dir", None)
    if d and os.path.exists(os.path.join(d, "batch_results.json")):
        return d
    root = os.path.dirname(os.path.abspath(__file__))
    cands = sorted(glob.glob(os.path.join(root, "batches", "batch_*")),
                   key=os.path.getmtime, reverse=True)
    for c in cands:
        if os.path.exists(os.path.join(c, "batch_results.json")):
            return c
    return d


def build_patient_vectors(batch_results):
    """
    Assemble BatchProcessor output into patient risk vectors.

    Input filtering rules (effective 2026-10-20): stacking values deprecated; use the "risk tier
    tier" driven. Multiple scoring models for one disease are first unified by _unify_disease_risk into
    One disease-level tier (majority vote + higher tier on ties + high-risk veto); tiers at or above
    Only medium-risk and above diseases enter the patient risk vector; the value is "tier base weight x vote
    support-rate coefficient", not the stacking 5-year absolute risk.

    stacking risk_5y stays in the diseases record for manual verification only; it does not drive retrieval.

    Returns:
        vectors:    {app_no: {English standard disease name: RWR weight}} (medium risk and above only)
        per_patient: {app_no: {pat_id, diseases: [...], uncomputable: [...]}}
                    each disease carries risk_tier / risk_weight / tier_source /
                    vote_valid / vote_support / elevated / selected, for verification.
    """
    mapping = _load_disease_mapping()
    vectors = {}
    per_patient = {}

    for item in batch_results:
        app_no = str(item.get("app_no"))
        pat_id = item.get("pat_id")
        vec = {}
        diseases = []

        for assessment in item.get("computable", []):
            disease_cn = assessment.get("disease_name")
            english_name = mapping.get(disease_cn)
            stacking_results = assessment.get("stacking_results") or []

            # stacking 5-year absolute risk: kept for manual verification only, no longer drives retrieval
            risk_5y = None
            for s in stacking_results:
                r = _parse_risk_5y(s)
                if r is not None:
                    risk_5y = r if risk_5y is None else max(risk_5y, r)

            unified = _unify_disease_risk(assessment.get("model_results"))
            tier = unified["tier"]
            weight = unified["weight"]

            # Only tiers at or above medium risk (after unification) qualify for retrieval
            elevated = tier >= MIN_RETRIEVAL_RISK_PRIORITY
            # Actual retrieval input: tier qualified + English standard disease name
            selected = elevated and bool(english_name)

            diseases.append({
                "disease_name": disease_cn,
                "english_name": english_name,
                "risk_5y": risk_5y,
                "risk_level": assessment.get("risk_level"),
                "risk_tier": tier,
                "risk_weight": weight,
                "tier_source": unified["source"],
                "vote_valid": unified["n_valid"],
                "vote_support": unified["n_support"],
                "elevated": elevated,
                "selected": selected,
                "stacking_models": len(stacking_results),
            })

            if selected:
                vec[english_name] = max(vec.get(english_name, 0.0), weight)

        per_patient[app_no] = {
            "pat_id": pat_id,
            "diseases": diseases,
            "uncomputable": item.get("not_computable", []),
        }
        if vec:
            vectors[app_no] = vec

    return vectors, per_patient


# ===================== Subgraph evidence compression (sub.py) =====================
def compress_subgraphs(records, progress_cb=None):
    """Structure-preserving evidence compression on successfully retrieved patient subgraphs (sub.py).

    Keep all edges and their source/target/relation; compress only each edge's
    document_sources (duplicated / low-information evidence). Compressed output is written to
    <app_no>/eid_<app_no>_edges_compressed.json, and
    record["subgraph_files"]["edges_json"] points to the compressed file,
    The original file is kept as edges_json_original.
    """
    import sub as sub_module

    out = []
    total = len(records)
    for idx, rec in enumerate(records, 1):
        if rec.get("status") != "SUCCESS":
            out.append(rec)
            continue
        files = rec.get("subgraph_files") or {}
        src = files.get("edges_json")
        if not src or not os.path.exists(src):
            out.append(rec)
            continue
        try:
            with open(src, "r", encoding="utf-8") as f:
                data = json.load(f)
            compressed, stats = sub_module.compress_patient_subgraph(
                data,
                direct_docs=SUB_DIRECT_DOCS,
                near_docs=SUB_NEAR_DOCS,
                background_docs=SUB_BACKGROUND_DOCS,
                max_extra_critical=SUB_MAX_EXTRA_CRITICAL,
                max_summary_chars=SUB_MAX_SUMMARY_CHARS,
            )
            dst = Path(src).with_name(
                Path(src).name.replace("_edges.json", "_edges_compressed.json")
            )
            out_bytes = sub_module.json_bytes(compressed, pretty=False)
            dst.write_bytes(out_bytes)
            orig_bytes = os.path.getsize(src)
            reduction = (1.0 - len(out_bytes) / orig_bytes) * 100.0 if orig_bytes else 0.0
            pass_checks = (
                stats.get("edge_structure_recall") == 1.0
                and stats.get("conflict_edge_recall") == 1.0
                and (stats.get("actionable_metric_type_coverage") or 0.0) >= 0.999999
            )
            files["edges_json_original"] = src
            files["edges_json"] = str(dst)
            rec["compression"] = {
                "status": "SUCCESS",
                "edge_count": stats.get("edge_count"),
                "docs_before": stats.get("original_document_sources"),
                "docs_after": stats.get("kept_document_sources"),
                "size_before": orig_bytes,
                "size_after": len(out_bytes),
                "size_reduction_percent": round(reduction, 2),
                "pass_structure_checks": pass_checks,
            }
        except Exception as e:
            rec["compression"] = {"status": f"ERROR: {e}"}
            print(f"[pipeline_runner] Subgraph compression failed eid={rec.get('eid')}: {e}")
        if progress_cb:
            try:
                progress_cb(idx, total, rec.get("eid"), rec.get("compression") or {})
            except Exception:
                pass
        out.append(rec)
    return out


# ===================== Pipeline main flow =====================
def run_pipeline(task_id):
    """Run the full pipeline in a background thread (started by a Flask endpoint)."""
    task = get_task(task_id)
    if not task:
        return
    app_nos = task["app_nos"]
    source = task.get("source", "search")
    enable_decide = bool(task.get("enable_decide", False))
    enable_report = bool(task.get("enable_report", True))

    with _PIPELINE_LOCK:
        try:
            _update(task_id, status="running", stage="init", progress=2,
                    message=f"Starting {len(app_nos)} app_no (source: {source})")

            # ---------- Stage 1: risk scoring (core BatchProcessor in cli.py) ----------
            _update(task_id, stage="risk", progress=5, message="Initializing risk scoring engine...")
            bp = _get_batch_processor()

            def risk_progress(stage, percent, message):
                # Risk scoring accounts for 5%-45% of total progress
                _update(task_id, stage=stage, progress=int(5 + percent * 0.40), message=message)

            _update(task_id, stage="risk", progress=6, message="Running multi-disease risk models and stacking scoring...")
            batch_results = bp.process_batch(app_nos, progress_callback=risk_progress)
            _update(task_id, stage="risk", progress=46,
                    message=f"Risk scoring complete: {len(batch_results)} patient results")

            # ---------- Stage 2: assemble the patient risk vector ----------
            _update(task_id, stage="vector", progress=48, message="Assembling patient risk vectors by risk tier...")
            vectors, per_patient = build_patient_vectors(batch_results)
            selected_cnt = sum(
                1 for info in per_patient.values() for d in info["diseases"] if d.get("selected")
            )
            protect_cnt = sum(
                1 for info in per_patient.values() for d in info["diseases"]
                if d.get("selected") and d.get("tier_source") == "protect"
            )
            _update(task_id, stage="vector", progress=50,
                    message=f"Risk vector assembly complete (tier-driven, stacking values deprecated): {selected_cnt} diseases qualified"
                            f" ({protect_cnt} included by high-risk veto protection),"
                            f"{len(vectors)}/{len(app_nos)} patients have valid risk vectors")

            # ---------- Stage 3: subgraph retrieval (subgraph_construct + hybrid_retrieval_engine) ----------
            import subgraph_construct

            out_dir = DEBUG_OUTPUT_DIR / task_id
            out_dir.mkdir(parents=True, exist_ok=True)

            records = []
            if vectors:
                _update(task_id, stage="subgraph", progress=51,
                        message="Loading knowledge-graph retrieval engine (first run takes ~1-3 minutes)...")

                def sg_progress(done, total, app_no, record):
                    p = 51 + int(42 * done / max(total, 1))
                    status = record.get("status")
                    _update(task_id, stage="subgraph", progress=p,
                            message=f"Subgraph search {done}/{total}: {app_no} -> {status}")

                records = subgraph_construct.retrieve_for_patients(
                    vectors, str(out_dir), progress_cb=sg_progress
                )
            record_map = {str(r.get("eid")): r for r in records}

            # ---------- Stage 3.4: subgraph evidence compression (sub.py, required) ----------
            if records:
                _update(task_id, stage="compress", progress=92,
                        message="Compressing redundant subgraph evidence (keep all edges, compress document_sources only)...")

                def compress_progress(done, total, app_no, stats):
                    p = 92 + int(2 * done / max(total, 1))
                    _update(task_id, stage="compress", progress=p,
                            message=f"Subgraph compression {done}/{total}: {app_no} "
                                    f"docs {stats.get('docs_before')}->{stats.get('docs_after')} "
                                    f"size -{stats.get('size_reduction_percent')}%")

                records = compress_subgraphs(records, progress_cb=compress_progress)
                record_map = {str(r.get("eid")): r for r in records}
                compressed_ok = [
                    r for r in records
                    if (r.get("compression") or {}).get("status") == "SUCCESS"
                ]
                _update(task_id, stage="compress", progress=94,
                        message=f"Subgraph compression complete: {len(compressed_ok)}/{len(records)} patient subgraphs compressed")

            # ---------- Stage 3.5: JEV-27B decision-model scoring (optional, selected in the UI) ----------
            if enable_decide:
                _update(task_id, stage="decide", progress=94,
                        message="Calling JEV-27B decision model to score subgraph knowledge edges (full records not condensed)...")
                try:
                    import jev27b_decider

                    def decide_progress(done, total, app_no, edge, score):
                        p = 94 + int(4 * done / max(total, 1))
                        _update(task_id, stage="decide", progress=p,
                                message=f"Decision scoring {app_no} edge {done}/{total}: "
                                        f"({edge.get('source_term')}) -> ({edge.get('target_term')}) score {score}")

                    records = jev27b_decider.decide_records(records, progress_cb=decide_progress)
                    record_map = {str(r.get("eid")): r for r in records}
                    decided = [r for r in records if r.get("decision_status") == "SUCCESS"]
                    _update(task_id, stage="decide", progress=98,
                            message=f"Decision scoring complete: {len(decided)}/{len(records)} patient subgraphs embedded with decision_score")
                except Exception as e:
                    print(f"[pipeline_runner] JEV-27B decision scoring stage failed: {e}")
                    _update(task_id, stage="decide", progress=98,
                            message=f"Decision scoring stage failed (subgraph output unaffected): {e}")
            else:
                _update(task_id, stage="decide", progress=98,
                        message="Decision model scoring skipped (not selected); subgraph compression output used directly")

            # ---------- Stage 3.6: multimorbidity primary-prevention report (optional) ----------
            if enable_report:
                _update(task_id, stage="report", progress=98,
                        message="Generating multi-disease primary-prevention reports (Markdown) and converting to visual format...")
                try:
                    import report_generator

                    _update(task_id, stage="report", progress=98,
                            message=f"Concurrent report generation: patient-level concurrency "
                                    f"{report_generator.REPORT_LLM_MAX_CONCURRENCY}，"
                                    f"LLM in-flight limit {report_generator.REPORT_LLM_MAX_INFLIGHT}, "
                                    f"overall wait limit "
                                    f"{report_generator.REPORT_LLM_WAIT_TIMEOUT:.0f}s")

                    def report_progress(done, total, app_no, status):
                        p = 98 + int(1 * done / max(total, 1))
                        _update(task_id, stage="report", progress=p,
                                message=f"Report generation {done}/{total}: {app_no} -> {status}")

                    records = report_generator.generate_reports(
                        records, batch_results, per_patient=per_patient,
                        progress_cb=report_progress,
                    )
                    record_map = {str(r.get("eid")): r for r in records}
                    ok = [r for r in records if r.get("report_status") == "SUCCESS"]
                    _update(task_id, stage="report", progress=99,
                            message=f"Report generation complete: {len(ok)}/{len(records)} patients")

                    # ---------- Stage 3.7: md -> single-file HTML / vector PDF ----------
                    # Same code path as `python -m report_pipeline`; rendering makes no LLM calls.
                    if ok:
                        _update(task_id, stage="render", progress=99,
                                message=f"Rendering {len(ok)} md reports to HTML/PDF...")
                        try:
                            from report_pipeline.auto import ReportRenderer

                            def render_progress(done, total, app_no, files):
                                p = 99
                                _update(task_id, stage="render", progress=p,
                                        message=f"Report rendering {done}/{total}: {app_no} -> "
                                                f"{'HTML+PDF' if files.get('pdf') else ('HTML' if files.get('html') else 'skipped')}")

                            renderer = ReportRenderer(_resolve_batch_dir(bp))
                            renderer.render_records(records, progress_cb=render_progress)
                            n_html = sum(1 for r in records
                                         if (r.get("report_files") or {}).get("html"))
                            n_pdf = sum(1 for r in records
                                        if (r.get("report_files") or {}).get("pdf"))
                            _update(task_id, stage="render", progress=99,
                                    message=f"Report rendering complete: HTML {n_html} / PDF {n_pdf}")
                        except Exception as e:
                            print(f"[pipeline_runner] Report rendering stage failed: {e}")
                            _update(task_id, stage="render", progress=99,
                                    message=f"Report rendering stage failed (md reports still available): {e}")
                except Exception as e:
                    print(f"[pipeline_runner] Report generation stage failed: {e}")
                    _update(task_id, stage="report", progress=99,
                            message=f"Report generation stage failed (subgraph output unaffected): {e}")
            else:
                _update(task_id, stage="report", progress=99,
                        message="Report generation skipped (not selected)")

            # ---------- Stage 4: aggregate output ----------
            _update(task_id, stage="summary", progress=99, message="Summarizing debug output...")
            patients = []
            for app_no in app_nos:
                app_no = str(app_no)
                info = per_patient.get(app_no, {})
                rec = record_map.get(app_no)
                entry = {
                    "app_no": app_no,
                    "pat_id": info.get("pat_id"),
                    "diseases": info.get("diseases", []),
                    "uncomputable": info.get("uncomputable", []),
                    "patient_risk_vector": vectors.get(app_no, {}),
                    "retrieval_status": (rec or {}).get("status", "NO_RISK_RESULT"),
                    "status_class": _status_class((rec or {}).get("status")),
                    "retrieval_message": (rec or {}).get("message", ""),
                    "hit_module": (rec or {}).get("hit_module"),
                    "core_mechanism": (rec or {}).get("core_mechanism"),
                    "suggested_interpretation": (rec or {}).get("suggested_interpretation"),
                    "subgraph_nodes": (rec or {}).get("subgraph_nodes"),
                    "subgraph_edges": (rec or {}).get("subgraph_edges"),
                    "subgraph_files": (rec or {}).get("subgraph_files", {}),
                    "compression": (rec or {}).get("compression"),
                    "decision_status": (rec or {}).get("decision_status"),
                    "decision_model": (rec or {}).get("decision_model"),
                    "report_status": (rec or {}).get("report_status"),
                    "report_files": (rec or {}).get("report_files"),
                    "report_meta": (rec or {}).get("report_meta"),
                }
                patients.append(entry)

            success_cnt = sum(1 for p in patients if p["retrieval_status"] == "SUCCESS")
            summary = {
                "task_id": task_id,
                "source": source,
                "enable_decide": enable_decide,
                "enable_report": enable_report,
                "created_at": task.get("created_at"),
                "finished_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "total_requested": len(app_nos),
                "total_with_vector": len(vectors),
                "total_retrieval_success": success_cnt,
                "output_dir": str(out_dir),
                "patients": patients,
            }

            result_file = out_dir / f"pipeline_result_{task_id}.json"
            with open(result_file, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)

            # Debug CSV: one (app_no, disease) per row
            csv_file = out_dir / f"stacking_summary_{task_id}.csv"
            with open(csv_file, "w", encoding="utf-8-sig") as f:
                f.write("app_no,pat_id,disease_name,english_name,unified_tier,rwr_weight,tier_source,"
                        "valid_votes,support_votes,input_query,risk_5y(reference_only),system_risk_level,stacking_model_count\n")
                for p in patients:
                    for d in p["diseases"]:
                        f.write("{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(
                            p["app_no"], p["pat_id"] or "",
                            d.get("disease_name", ""), d.get("english_name") or "",
                            _TIER_LABEL.get(d.get("risk_tier"), "-"),
                            "" if d.get("risk_weight") is None else d.get("risk_weight"),
                            _SOURCE_LABEL.get(d.get("tier_source"), "-"),
                            d.get("vote_valid") or 0,
                            d.get("vote_support") or 0,
                            "Yes" if d.get("selected") else "No",
                            "" if d.get("risk_5y") is None else d.get("risk_5y"),
                            d.get("risk_level") or "",
                            d.get("stacking_models") or 0,
                        ))

            summary["result_file"] = str(result_file)
            summary["summary_csv"] = str(csv_file)
            _update(task_id, status="done", stage="done", progress=100,
                    message=f"All done: {success_cnt}/{len(app_nos)} patients' subgraph search succeeded; see {result_file}",
                    result=summary)

        except Exception as e:
            err = f"Pipeline execution failed: {e}\n{traceback.format_exc()}"
            print(err)
            _update(task_id, status="error", stage="error", message=f"Execution failed: {e}", error=err)


def start_pipeline(app_nos, source="search", enable_decide=False, enable_report=True):
    """Create a task, start a background thread and return task_id."""
    task = create_task(app_nos, source, enable_decide=enable_decide, enable_report=enable_report)
    t = threading.Thread(target=run_pipeline, args=(task["task_id"],), daemon=True)
    t.start()
    return task["task_id"]
