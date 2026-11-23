#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Patient subgraph: structure-preserving evidence compression

Goal
----
1. Keep the top-level and edge_details structure of the original patient subgraph JSON unchanged;
2. Every disease in patient_risk_vector is treated as a target/high-risk disease for this patient;
3. Do not delete edges or change their source/target/relation/all_types;
4. Mainly compress duplicated, low-information document_sources under each edge;
5. Retain enough to support LLM report generation:
   - Relation type and direction
   - aggregated_properties (total documents, evidence strength, consensus direction, conflicts)
   - representative text_summary (not truncated by default)
   - Meaningful quantitative indicators
   - recommendation_class / evidence_level
   - target_population / special_populations / contraindications
   - Normalized strength labels etc.
6. Automatically drop exact duplicates between evidence_context and the outer context, plus low-information fields such as null / not_reported.

Default Windows path
----------------
Input: D:\\Desktop\\subgraph_tj
Output: D:\\Desktop\\subgraph_2

Run directly:
    D:\\anaconda\\python.exe subgraph_structure_preserving_compress.py

Or customize:
    python subgraph_structure_preserving_compress.py "D:\\Desktop\\subgraph_tj" --output-dir "D:\\Desktop\\subgraph_2"

Output
----
- Each input JSON maps to a same-named compressed JSON in the output directory (relative structure preserved);
- compression_summary.csv: batch compression statistics, not part of LLM input.

Note
----
subgraph_edge_data.document_count and aggregated_properties.total_documents keep their original values,
Total evidence for this relation in the original KG; document_sources is only the representative subset expanded for the LLM.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import re
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


DEFAULT_INPUT_DIR = Path(r"D:\Desktop\subgraph_tj_extend")
DEFAULT_OUTPUT_DIR = Path(r"D:\Desktop\subgraph_extend")

# These are task-level relation types, not a disease/factor dictionary.
REPORT_RELEVANT_RELATIONS = {
    "Risk_Factor",
    "Protective_Factor",
    "Preventive",
    "Screening",
    "Surveillance",
    "Diagnostic",
    "Therapeutic",
    "Contraindicated",
    "Causal",
    "Prognostic",
    "Associational",
    "Positive_Correlation",
    "Negative_Correlation",
}

# Quantitative keywords most useful for report recommendations; used only to prioritize evidence, never to rewrite indicators.
ACTIONABLE_METRIC_KEYWORDS = (
    "threshold",
    "screening",
    "frequency",
    "interval",
    "recommend",
    "dose",
    "dosage",
    "duration",
    "target",
    "cutoff",
    "cut-off",
    "risk reduction",
    "hazard ratio",
    "odds ratio",
    "relative risk",
    "hr",
    "rr",
    "or",
)

EMPTY_STRINGS = {
    "",
    "null",
    "none",
    "not_reported",
    "not reported",
    "not specified",
    "unknown",
    "n/a",
    "na",
}


# ---------------------------------------------------------------------------
# Basic utilities
# ---------------------------------------------------------------------------

def is_empty_value(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, str):
        return v.strip().lower() in EMPTY_STRINGS
    if isinstance(v, (list, tuple, set, dict)):
        return len(v) == 0
    return False


def prune_empty(obj: Any) -> Any:
    """Recursively drop low-information fields such as null / not_reported / empty lists."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            pv = prune_empty(v)
            if not is_empty_value(pv):
                out[k] = pv
        return out
    if isinstance(obj, list):
        out = []
        for x in obj:
            px = prune_empty(x)
            if not is_empty_value(px):
                out.append(px)
        return out
    return obj


def normalize_text(s: Any) -> str:
    if s is None:
        return ""
    return re.sub(r"\s+", " ", str(s)).strip().lower()


def meaningful_metrics(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    rd = doc.get("relation_detail") or {}
    sq = rd.get("strength_quantitative") or {}
    metrics = sq.get("metrics") or []
    out: List[Dict[str, Any]] = []
    for m in metrics:
        if not isinstance(m, dict):
            continue
        # metric_type present alone with no value is usually just a "Qualitative" placeholder.
        useful = False
        for k, v in m.items():
            if k == "metric_type":
                continue
            if not is_empty_value(v):
                useful = True
                break
        if useful:
            out.append(m)
    return out


def actionable_metric_category(metric_type: Any) -> Optional[str]:
    """Group genuinely actionable/interpretable indicators into a few task-level categories to avoid OR-matching false positives such as "reported"."""
    mt = normalize_text(metric_type)
    if not mt:
        return None

    # Short abbreviations must match exactly.
    if mt in {"hr", "hazard ratio"}:
        return "effect_size_hr"
    if mt in {"rr", "relative risk", "risk ratio", "rrr"}:
        return "effect_size_rr"
    if mt in {"or", "odds ratio"}:
        return "effect_size_or"

    if any(x in mt for x in ("screening frequency", "screening interval", "follow-up", "follow up", "surveillance interval", "monitoring frequency")):
        return "screening_or_followup_frequency"
    if any(x in mt for x in ("threshold", "cutoff", "cut-off", "diagnostic threshold", "initiation threshold")):
        return "threshold"
    if any(x in mt for x in ("dose", "dosage")):
        return "dose"
    if "duration" in mt:
        return "duration"
    if any(x in mt for x in ("target", "target_value", "target value", "target reduction", "target percentage", "bp_target")):
        return "target"
    if any(x in mt for x in ("recommendation", "recommendation class", "recommendation grade")):
        return "recommendation"
    if "risk reduction" in mt:
        return "risk_reduction"
    return None


def actionable_metric_types(doc: Dict[str, Any]) -> Set[str]:
    """Backward-compatible with the old stat name: return the set of actionable-metric categories."""
    found: Set[str] = set()
    for m in meaningful_metrics(doc):
        cat = actionable_metric_category(m.get("metric_type"))
        if cat:
            found.add(cat)
    return found


def evidence_context(doc: Dict[str, Any]) -> Dict[str, Any]:
    rd = doc.get("relation_detail") or {}
    ctx = rd.get("evidence_context")
    return ctx if isinstance(ctx, dict) else {}


def relation_type_of_doc(doc: Dict[str, Any]) -> str:
    rd = doc.get("relation_detail") or {}
    return str(doc.get("relation_type") or rd.get("relation_type") or "")


def relation_subtype_of_doc(doc: Dict[str, Any]) -> str:
    rd = doc.get("relation_detail") or {}
    return str(rd.get("relation_subtype") or "")


def year_of_doc(doc: Dict[str, Any]) -> int:
    ctx = evidence_context(doc)
    vals = [ctx.get("guideline_year"), ctx.get("publication_year")]
    years = []
    for v in vals:
        try:
            if v is not None:
                years.append(int(v))
        except (ValueError, TypeError):
            pass
    return max(years) if years else 0


def nonempty_ctx(doc: Dict[str, Any], key: str) -> bool:
    return not is_empty_value(evidence_context(doc).get(key))


# ---------------------------------------------------------------------------
# Graph distance only controls how much evidence is expanded, not whether an edge is kept
# ---------------------------------------------------------------------------

def build_adjacency(edge_details: Sequence[Dict[str, Any]]) -> Dict[str, Set[str]]:
    adj: Dict[str, Set[str]] = defaultdict(set)
    for e in edge_details:
        s = e.get("source_term")
        t = e.get("target_term")
        if s is None or t is None:
            continue
        s, t = str(s), str(t)
        adj[s].add(t)
        adj[t].add(s)
    return adj


def distances_from_targets(
    edge_details: Sequence[Dict[str, Any]], targets: Sequence[str]
) -> Dict[str, int]:
    adj = build_adjacency(edge_details)
    dist: Dict[str, int] = {}
    q: deque[str] = deque()
    for target in targets:
        target = str(target)
        dist[target] = 0
        q.append(target)
    while q:
        u = q.popleft()
        for v in adj.get(u, ()):
            if v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist


def edge_distance(edge: Dict[str, Any], dist: Dict[str, int]) -> int:
    inf = 10**9
    s = str(edge.get("source_term", ""))
    t = str(edge.get("target_term", ""))
    return min(dist.get(s, inf), dist.get(t, inf))


# ---------------------------------------------------------------------------
# Representative evidence selection
# ---------------------------------------------------------------------------

def qualitative_strength_score(s: Any) -> float:
    x = normalize_text(s)
    if x in {"strong", "high", "very strong"}:
        return 4.0
    if x in {"moderate", "medium"}:
        return 2.5
    if x in {"weak", "low"}:
        return 1.0
    if "qualitative" in x:
        return 0.5
    return 0.0


def study_design_score(v: Any) -> float:
    x = normalize_text(v)
    score = 0.0
    if "guideline" in x:
        score = max(score, 5.0)
    if "meta-analysis" in x or "meta analysis" in x:
        score = max(score, 4.8)
    if "systematic review" in x:
        score = max(score, 4.5)
    if "randomized" in x or "randomised" in x or "rct" in x:
        score = max(score, 4.2)
    if "consensus" in x:
        score = max(score, 3.6)
    if "cohort" in x:
        score = max(score, 2.6)
    if "case-control" in x or "case control" in x:
        score = max(score, 2.2)
    if "cross-sectional" in x:
        score = max(score, 1.5)
    return score


def doc_score(doc: Dict[str, Any]) -> float:
    rd = doc.get("relation_detail") or {}
    ctx = evidence_context(doc)
    score = 0.0

    score += qualitative_strength_score(rd.get("strength_qualitative"))
    score += study_design_score(ctx.get("study_design"))

    metrics = meaningful_metrics(doc)
    if metrics:
        score += 2.2 + min(1.5, 0.25 * len(metrics))
    if actionable_metric_types(doc):
        score += 2.0

    if nonempty_ctx(doc, "recommendation_class"):
        score += 1.8
    if nonempty_ctx(doc, "evidence_level_letter"):
        score += 1.3
    if nonempty_ctx(doc, "contraindications"):
        score += 2.5
    if nonempty_ctx(doc, "target_population"):
        score += 0.8
    if nonempty_ctx(doc, "special_populations"):
        score += 0.8

    tags = rd.get("standardized_strength_tags") or []
    if tags:
        score += 0.8

    # Newer guidelines/studies act only as a weak tie-breaker; newer is not necessarily better.
    y = year_of_doc(doc)
    if y:
        score += min(0.5, max(0, y - 2015) * 0.03)

    summary = rd.get("text_summary")
    if isinstance(summary, str) and summary.strip():
        score += 0.5

    return score


def conflict_relation_types(edge: Dict[str, Any]) -> Set[str]:
    out: Set[str] = set()
    agg = (edge.get("full_kg_edge_data") or {}).get("aggregated_properties") or {}
    for c in agg.get("conflicts") or []:
        if not isinstance(c, dict):
            continue
        details = c.get("details") or []
        for d in details:
            if isinstance(d, str) and d:
                out.add(d)
    return out


def choose_representative_documents(
    edge: Dict[str, Any],
    base_quota: int,
    max_extra_critical: int,
) -> List[str]:
    docs_dict = (edge.get("full_kg_edge_data") or {}).get("document_sources") or {}
    if not isinstance(docs_dict, dict) or not docs_dict:
        return []

    items = list(docs_dict.items())
    if len(items) <= base_quota:
        return [k for k, _ in items]

    # Exactly duplicated summaries: keep only the higher-scoring one when filling.
    scored = sorted(items, key=lambda kv: (doc_score(kv[1]), kv[0]), reverse=True)
    score_map = {k: doc_score(v) for k, v in items}

    selected: List[str] = []
    selected_set: Set[str] = set()

    def add(k: Optional[str]) -> None:
        if k is not None and k not in selected_set:
            selected.append(k)
            selected_set.add(k)

    def best_where(pred) -> Optional[str]:
        candidates = [(k, d) for k, d in items if pred(k, d)]
        if not candidates:
            return None
        return max(candidates, key=lambda kv: (score_map[kv[0]], kv[0]))[0]

    # 1) Keep at least the single highest-quality document.
    add(scored[0][0])

    # 2) Cover at least one document per document-level relation_type, especially to preserve semantic differences within an edge.
    doc_relation_types = sorted({relation_type_of_doc(d) for _, d in items if relation_type_of_doc(d)})
    for rt in doc_relation_types:
        add(best_where(lambda _k, d, rt=rt: relation_type_of_doc(d) == rt))

    # 3) Relation types involved in conflicts should have representative readable evidence.
    for rt in sorted(conflict_relation_types(edge)):
        add(best_where(lambda _k, d, rt=rt: relation_type_of_doc(d) == rt))

    # 4) Safety / contraindications must not be suppressed due to few evidence items.
    add(best_where(lambda _k, d: nonempty_ctx(d, "contraindications")))

    # 5) Actionable metrics such as screening frequency, thresholds, dosage, risk effect size:
    #    Keep at least one representative document per task-level category where possible.
    metric_categories = sorted({
        cat
        for _k, d in items
        for cat in actionable_metric_types(d)
    })
    for cat in metric_categories:
        add(best_where(lambda _k, d, cat=cat: cat in actionable_metric_types(d)))

    # 6) Prefer keeping one guideline with an explicit recommendation/evidence level.
    add(best_where(lambda _k, d: nonempty_ctx(d, "recommendation_class") or nonempty_ctx(d, "evidence_level_letter")))

    # Critical evidence may exceed base_quota, but a soft cap prevents a single edge from growing unbounded.
    hard_cap = max(1, min(len(items), base_quota + max_extra_critical))
    if len(selected) > hard_cap:
        # Prioritize relation-type coverage, then overall score: keep the top score first, then cover relation_type as much as possible.
        reduced: List[str] = []
        reduced_set: Set[str] = set()

        def radd(k: str) -> None:
            if k not in reduced_set and len(reduced) < hard_cap:
                reduced.append(k)
                reduced_set.add(k)

        radd(scored[0][0])
        for rt in doc_relation_types:
            k = best_where(lambda _k, d, rt=rt: relation_type_of_doc(d) == rt)
            if k:
                radd(k)
        # Safety first.
        k = best_where(lambda _k, d: nonempty_ctx(d, "contraindications"))
        if k:
            radd(k)
        # Cover each category of actionable metric where possible.
        for cat in metric_categories:
            k = best_where(lambda _k, d, cat=cat: cat in actionable_metric_types(d))
            if k:
                radd(k)
        # Recommendation grade / evidence level.
        k = best_where(lambda _k, d: nonempty_ctx(d, "recommendation_class") or nonempty_ctx(d, "evidence_level_letter"))
        if k:
            radd(k)
        for k, _ in scored:
            radd(k)
        selected = reduced
        selected_set = reduced_set

    # 7) Fill base_quota so evidence is not too sparse just because no "special fields" exist; skip fully duplicated summaries.
    seen_summaries: Set[str] = set()
    for k in selected:
        rd = docs_dict[k].get("relation_detail") or {}
        s = normalize_text(rd.get("text_summary"))
        if s:
            seen_summaries.add(s)

    target_n = min(len(items), max(base_quota, len(selected)))
    for k, d in scored:
        if len(selected) >= target_n:
            break
        if k in selected_set:
            continue
        summary = normalize_text((d.get("relation_detail") or {}).get("text_summary"))
        if summary and summary in seen_summaries:
            continue
        add(k)
        if summary:
            seen_summaries.add(summary)

    # Preserve the original document_sources order for comparison and readability.
    return [k for k, _ in items if k in selected_set]


# ---------------------------------------------------------------------------
# Per-document evidence compression
# ---------------------------------------------------------------------------

def compress_metric(m: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    pm = prune_empty(copy.deepcopy(m))
    if not isinstance(pm, dict) or not pm:
        return None
    # If only metric_type remains with no quantitative/descriptive content, it is not worth the main context.
    if set(pm.keys()) <= {"metric_type"}:
        return None
    return pm


def compress_document(doc: Dict[str, Any], max_summary_chars: int = 0) -> Dict[str, Any]:
    d = copy.deepcopy(doc)

    # extraction_timestamp has no clinical meaning for report generation.
    d.pop("extraction_timestamp", None)

    # Drop entity_context when empty; keep it when non-empty.
    if is_empty_value(d.get("entity_context")):
        d.pop("entity_context", None)

    rd = d.get("relation_detail")
    if not isinstance(rd, dict):
        rd = {}
        d["relation_detail"] = rd

    # metrics: keep informative quantitative items, drop null/not_reported placeholders.
    sq = rd.get("strength_quantitative")
    if isinstance(sq, dict):
        metrics = sq.get("metrics") or []
        new_metrics = []
        for m in metrics:
            if isinstance(m, dict):
                cm = compress_metric(m)
                if cm:
                    new_metrics.append(cm)
        if new_metrics:
            sq["metrics"] = new_metrics
            rd["strength_quantitative"] = prune_empty(sq)
        else:
            rd.pop("strength_quantitative", None)

    # evidence_context: drop all low-information placeholder fields but keep actual values.
    if isinstance(rd.get("evidence_context"), dict):
        ctx = prune_empty(rd["evidence_context"])
        if ctx:
            rd["evidence_context"] = ctx
        else:
            rd.pop("evidence_context", None)

    # Do not truncate summaries by default, to preserve semantics such as the same concept acting differently across diseases.
    if max_summary_chars and max_summary_chars > 0:
        s = rd.get("text_summary")
        if isinstance(s, str) and len(s) > max_summary_chars:
            rd["text_summary"] = s[: max_summary_chars - 1].rstrip() + "…"

    rd = prune_empty(rd)
    d["relation_detail"] = rd

    # In raw data the outer context often duplicates relation_detail.evidence_context exactly.
    # Delete only exact duplicates; when they differ, keep the non-empty information to avoid semantic loss.
    outer_ctx = d.get("context")
    inner_ctx = rd.get("evidence_context")
    if isinstance(outer_ctx, dict):
        pruned_outer = prune_empty(copy.deepcopy(outer_ctx))
        if pruned_outer == inner_ctx or not pruned_outer:
            d.pop("context", None)
        else:
            d["context"] = pruned_outer

    return prune_empty(d)


# ---------------------------------------------------------------------------
# Per-patient compression + validation
# ---------------------------------------------------------------------------

def edge_relation_signature(edge: Dict[str, Any]) -> Tuple[Any, ...]:
    sg = edge.get("subgraph_edge_data") or {}
    return (
        edge.get("source_term"),
        edge.get("target_term"),
        sg.get("type"),
        tuple(sg.get("all_types") or []),
    )


def count_docs(data: Dict[str, Any]) -> int:
    n = 0
    for e in data.get("edge_details") or []:
        ds = (e.get("full_kg_edge_data") or {}).get("document_sources") or {}
        if isinstance(ds, dict):
            n += len(ds)
    return n


def count_metrics(data: Dict[str, Any]) -> int:
    n = 0
    for e in data.get("edge_details") or []:
        ds = (e.get("full_kg_edge_data") or {}).get("document_sources") or {}
        if not isinstance(ds, dict):
            continue
        for d in ds.values():
            n += len(meaningful_metrics(d))
    return n


def collect_actionable_metric_types(data: Dict[str, Any]) -> Set[str]:
    out: Set[str] = set()
    for e in data.get("edge_details") or []:
        ds = (e.get("full_kg_edge_data") or {}).get("document_sources") or {}
        if not isinstance(ds, dict):
            continue
        for d in ds.values():
            out.update(actionable_metric_types(d))
    return out


def count_conflict_edges(data: Dict[str, Any]) -> int:
    n = 0
    for e in data.get("edge_details") or []:
        agg = (e.get("full_kg_edge_data") or {}).get("aggregated_properties") or {}
        if agg.get("conflicts"):
            n += 1
    return n


def compress_patient_subgraph(
    data: Dict[str, Any],
    direct_docs: int = 3,
    near_docs: int = 2,
    background_docs: int = 1,
    max_extra_critical: int = 3,
    max_summary_chars: int = 0,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    out = copy.deepcopy(data)
    edges = out.get("edge_details") or []
    targets = list((out.get("patient_risk_vector") or {}).keys())
    dist = distances_from_targets(edges, targets)

    original_docs = count_docs(data)
    original_metrics = count_metrics(data)
    original_actionable = collect_actionable_metric_types(data)

    kept_docs_total = 0
    direct_edges = 0
    near_edges = 0
    background_edges = 0

    for edge in edges:
        ed = edge_distance(edge, dist)
        sg = edge.get("subgraph_edge_data") or {}
        rel_types = set(sg.get("all_types") or [])
        if sg.get("type"):
            rel_types.add(sg.get("type"))

        if ed == 0:
            quota = max(1, direct_docs)
            direct_edges += 1
        elif ed == 1:
            quota = max(1, near_docs)
            near_edges += 1
        else:
            quota = max(1, background_docs)
            background_edges += 1

        # Keep at least 2 documents for report-relevant relations even if ranked low (when available).
        if rel_types & REPORT_RELEVANT_RELATIONS:
            quota = max(quota, 2 if ed > 0 else quota)

        full = edge.get("full_kg_edge_data")
        if not isinstance(full, dict):
            continue
        docs = full.get("document_sources")
        if not isinstance(docs, dict) or not docs:
            continue

        keep_keys = choose_representative_documents(
            edge=edge,
            base_quota=quota,
            max_extra_critical=max_extra_critical,
        )
        kept = {}
        for k in keep_keys:
            kept[k] = compress_document(docs[k], max_summary_chars=max_summary_chars)
        full["document_sources"] = kept
        kept_docs_total += len(kept)

        # aggregated_properties kept as-is: total documents, evidence strength, direction and conflicts unchanged.

    # ---- Strict validation: graph structure and core relation semantics must not change ----
    original_edges = data.get("edge_details") or []
    compressed_edges = out.get("edge_details") or []
    if len(original_edges) != len(compressed_edges):
        raise AssertionError("edge_details count changed; structure-preservation violated.")

    if data.get("patient_risk_vector") != out.get("patient_risk_vector"):
        raise AssertionError("patient_risk_vector was modified.")

    for i, (a, b) in enumerate(zip(original_edges, compressed_edges)):
        if edge_relation_signature(a) != edge_relation_signature(b):
            raise AssertionError(f"edge {i}: source/target/relation/all_types was modified.")
        aagg = (a.get("full_kg_edge_data") or {}).get("aggregated_properties") or {}
        bagg = (b.get("full_kg_edge_data") or {}).get("aggregated_properties") or {}
        if aagg != bagg:
            raise AssertionError(f"edge {i}: aggregated_properties was modified.")

    compressed_metrics = count_metrics(out)
    compressed_actionable = collect_actionable_metric_types(out)
    original_conflicts = count_conflict_edges(data)
    compressed_conflicts = count_conflict_edges(out)

    stats = {
        "patient_id": str(data.get("eid", "")),
        "target_disease_count": len(targets),
        "edge_count": len(original_edges),
        "edge_structure_recall": 1.0,
        "original_document_sources": original_docs,
        "kept_document_sources": kept_docs_total,
        "document_expansion_ratio": (kept_docs_total / original_docs) if original_docs else 1.0,
        "original_meaningful_metrics": original_metrics,
        "kept_meaningful_metrics": compressed_metrics,
        "metric_direct_retention": (compressed_metrics / original_metrics) if original_metrics else 1.0,
        "actionable_metric_type_coverage": (
            len(original_actionable & compressed_actionable) / len(original_actionable)
            if original_actionable else 1.0
        ),
        "original_conflict_edges": original_conflicts,
        "kept_conflict_edges": compressed_conflicts,
        "conflict_edge_recall": (
            compressed_conflicts / original_conflicts if original_conflicts else 1.0
        ),
        "direct_target_edges": direct_edges,
        "near_target_edges": near_edges,
        "background_edges": background_edges,
    }
    return out, stats


# ---------------------------------------------------------------------------
# Files / batch processing
# ---------------------------------------------------------------------------

def json_bytes(obj: Any, pretty: bool = False) -> bytes:
    if pretty:
        s = json.dumps(obj, ensure_ascii=False, indent=2)
    else:
        s = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    return s.encode("utf-8")


def write_json(path: Path, obj: Any, pretty: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json_bytes(obj, pretty=pretty))


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        x = json.load(f)
    if not isinstance(x, dict):
        raise ValueError("top-level JSON must be an object.")
    return x


def iter_json_files(input_dir: Path, recursive: bool = True) -> List[Path]:
    pattern = "**/*.json" if recursive else "*.json"
    return sorted(p for p in input_dir.glob(pattern) if p.is_file())


def process_one(
    src: Path,
    dst: Path,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    data = load_json(src)
    original_raw_bytes = src.stat().st_size

    compressed, stats = compress_patient_subgraph(
        data,
        direct_docs=args.direct_docs,
        near_docs=args.near_docs,
        background_docs=args.background_docs,
        max_extra_critical=args.max_extra_critical,
        max_summary_chars=args.max_summary_chars,
    )
    out_bytes = json_bytes(compressed, pretty=args.pretty)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(out_bytes)

    stats["source_file"] = str(src)
    stats["output_file"] = str(dst)
    stats["original_bytes"] = original_raw_bytes
    stats["compressed_bytes"] = len(out_bytes)
    stats["compression_ratio"] = (len(out_bytes) / original_raw_bytes) if original_raw_bytes else 1.0
    stats["size_reduction_percent"] = (
        (1.0 - len(out_bytes) / original_raw_bytes) * 100.0 if original_raw_bytes else 0.0
    )
    stats["pass_structure_checks"] = (
        stats["edge_structure_recall"] == 1.0
        and stats["conflict_edge_recall"] == 1.0
        and stats["actionable_metric_type_coverage"] >= 0.999999
    )
    return stats


def write_summary_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    columns = [
        "patient_id",
        "source_file",
        "output_file",
        "target_disease_count",
        "edge_count",
        "edge_structure_recall",
        "original_document_sources",
        "kept_document_sources",
        "document_expansion_ratio",
        "original_meaningful_metrics",
        "kept_meaningful_metrics",
        "metric_direct_retention",
        "actionable_metric_type_coverage",
        "original_conflict_edges",
        "kept_conflict_edges",
        "conflict_edge_recall",
        "direct_target_edges",
        "near_target_edges",
        "background_edges",
        "original_bytes",
        "compressed_bytes",
        "compression_ratio",
        "size_reduction_percent",
        "pass_structure_checks",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in columns})


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Keep the patient subgraph JSON structure; compress only redundant evidence inside each edge."
    )
    p.add_argument(
        "input_dir",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=r"Input directory (default: D:\Desktop\subgraph_tj)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=r"Output directory (default: D:\Desktop\subgraph_2)",
    )
    p.add_argument(
        "--direct-docs",
        type=int,
        default=3,
        help="Edges directly connected to patient_risk_vector diseases; base number of representative evidence documents (default 3).",
    )
    p.add_argument(
        "--near-docs",
        type=int,
        default=2,
        help="Edges one hop from target diseases; base number of representative evidence documents (default 2).",
    )
    p.add_argument(
        "--background-docs",
        type=int,
        default=1,
        help="Base number of representative evidence documents for farther edges (default 1; at least 2 for report-relevant relations).",
    )
    p.add_argument(
        "--max-extra-critical",
        type=int,
        default=3,
        help="Extra evidence documents allowed for relation-type differences, contraindications, key indicators and recommendation grades (default 3).",
    )
    p.add_argument(
        "--max-summary-chars",
        type=int,
        default=0,
        help="Max characters per text_summary; 0 means no truncation (default, safest).",
    )
    p.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON for human reading; default is compact JSON, better for LLMs.",
    )
    p.add_argument(
        "--no-recursive",
        action="store_true",
        help="Process only top-level JSON in the input directory; do not recurse.",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir

    if args.direct_docs < 1 or args.near_docs < 1 or args.background_docs < 1:
        raise ValueError("direct/near/background docs must all be >= 1.")
    if args.max_extra_critical < 0:
        raise ValueError("--max-extra-critical must not be less than 0.")

    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"input directory not found: {input_dir}")

    files = iter_json_files(input_dir, recursive=not args.no_recursive)
    # Avoid treating old output as input when the output directory sits inside the input directory.
    try:
        output_resolved = output_dir.resolve()
        files = [p for p in files if output_resolved not in p.resolve().parents]
    except Exception:
        pass

    if not files:
        raise FileNotFoundError(f"no JSON found in input directory: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    errors: List[Tuple[str, str]] = []

    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"JSON files: {len(files)}")
    print("High-risk/target disease rule: every disease in patient_risk_vector.")
    print("Compression: keep all edges, compress only document_sources.")
    print("-")

    for i, src in enumerate(files, 1):
        rel = src.relative_to(input_dir)
        dst = output_dir / rel
        try:
            stats = process_one(src, dst, args)
            rows.append(stats)
            print(
                f"[{i}/{len(files)}] {src.name} | "
                f"edges={stats['edge_count']} | "
                f"docs={stats['original_document_sources']}->{stats['kept_document_sources']} | "
                f"size -{stats['size_reduction_percent']:.1f}% | "
                f"actionable_metric_coverage={stats['actionable_metric_type_coverage']:.3f} | "
                f"PASS={stats['pass_structure_checks']}"
            )
        except Exception as e:
            errors.append((str(src), repr(e)))
            print(f"[{i}/{len(files)}] ERROR {src}: {e}", file=sys.stderr)

    summary_path = output_dir / "compression_summary.csv"
    write_summary_csv(summary_path, rows)

    print("-")
    print(f"Done: {len(rows)} succeeded, {len(errors)} failed")
    print(f"Compressed output directory: {output_dir}")
    print(f"Batch summary: {summary_path}")
    print("For LLM use: the same-named compressed JSONs in the output directory.")

    if rows:
        total_orig = sum(int(r["original_bytes"]) for r in rows)
        total_comp = sum(int(r["compressed_bytes"]) for r in rows)
        reduction = (1 - total_comp / total_orig) * 100 if total_orig else 0.0
        print(f"Overall size reduction: {reduction:.2f}%")
        print(
            "Structure check passed: "
            f"{sum(bool(r['pass_structure_checks']) for r in rows)}/{len(rows)}"
        )

    if errors:
        err_path = output_dir / "compression_errors.txt"
        with err_path.open("w", encoding="utf-8") as f:
            for fn, msg in errors:
                f.write(fn + "\t" + msg + "\n")
        print(f"Error log: {err_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
