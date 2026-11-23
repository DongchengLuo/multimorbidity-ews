# -*- coding: utf-8 -*-
"""
Downstream report-generation concurrency test (16 patients by default)
================================================================================
Usage:
    python _test_report_concurrency.py
    python _test_report_concurrency.py --patients 16 --concurrency 16

Test coverage:
  1. Pick N patients with existing subgraphs (edges_json) from historical debug_output tasks,
     Copy edges into a separate test directory to avoid overwriting the serial baseline;
  2. Use the real chain: BatchProcessor risk scoring -> build_patient_vectors ->
     report_generator.generate_reports (concurrent);
  3. Validate: report status for all patients, md section completeness, json-render spec validity,
     whether output was truncated by max_tokens and whether progress increased monotonically to N/N;
  4. Compare with the serial baseline: section heading sets for the same patient (same prompt version),
     Whether key values are retained; patients from different prompt versions are compared structurally only.
"""
import argparse
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Serial baseline (same prompt version, 3 patients)
BASELINE_TASK = "task_20261023_191939_b970a6"
# Add patient source (legacy prompt version, structural comparison only)
EXTRA_TASK = "task_20261022_120952_2e3ea0"

DEBUG_DIR = PROJECT_ROOT / "debug_output"


def _load_result(task_id):
    f = DEBUG_DIR / task_id / f"pipeline_result_{task_id}.json"
    return json.load(open(f, encoding="utf-8"))


def _collect_patients(n):
    """Prefer serial baseline patients (for content comparison); fill the rest from all historical tasks.

    Historical debug_output already has 100+ patients with edges_json, enough for high-concurrency load testing.
    Each app_no is taken once; baseline patients first, ensuring a few can be compared under the same prompt.
    """
    picked = []
    seen = set()

    def _add(p, kind):
        app_no = str(p.get("app_no") or "")
        edges = (p.get("subgraph_files") or {}).get("edges_json")
        if not app_no or app_no in seen or not edges or not os.path.exists(edges):
            return False
        seen.add(app_no)
        picked.append({"app_no": app_no, "edges": edges,
                       "baseline_dir": str(Path(edges).parent),
                       "baseline_kind": kind})
        return True

    # 1) Serial baseline patients (same prompt version, for content comparison)
    for p in _load_result(BASELINE_TASK)["patients"]:
        _add(p, "same_prompt")

    # 2) Legacy prompt-version tasks (structural comparison)
    for p in _load_result(EXTRA_TASK)["patients"]:
        if len(picked) >= n:
            break
        _add(p, "old_prompt")

    # 3) All remaining historical tasks, padded to n
    if len(picked) < n:
        for f in sorted(DEBUG_DIR.glob("*/pipeline_result_*.json")):
            if len(picked) >= n:
                break
            try:
                d = json.load(open(f, encoding="utf-8"))
            except Exception:
                continue
            for p in d.get("patients", []):
                if len(picked) >= n:
                    break
                _add(p, "old_prompt")
    return picked[:n]


def _prepare_workdir(patients, tag):
    work = DEBUG_DIR / tag
    work.mkdir(parents=True, exist_ok=True)
    for p in patients:
        d = work / p["app_no"]
        d.mkdir(parents=True, exist_ok=True)
        dst = d / f"eid_{p['app_no']}_edges.json"
        shutil.copyfile(p["edges"], dst)
        p["work_edges"] = str(dst)
    return work


def _sections(md):
    return [m.strip() for m in re.findall(r"^##\s+(.+)$", md, flags=re.M)]


def _numbers(text):
    """Extract key numeric tokens from the report (percentages/decimals/integers) to compare information retention."""
    return set(re.findall(r"\d+(?:\.\d+)?%?", text))


def _validate_one(app_no, work_edges, baseline_dir, baseline_kind):
    import jsonrender_catalog as catalog

    d = Path(work_edges).parent
    md_path = d / f"report_{app_no}.md"
    spec_path = d / f"report_{app_no}.json"
    meta_path = d / f"report_{app_no}_meta.json"
    out = {"app_no": app_no, "baseline_kind": baseline_kind, "problems": []}

    if not md_path.exists():
        out["problems"].append("missing md report")
        return out
    md = md_path.read_text(encoding="utf-8")
    out["md_chars"] = len(md)
    out["sections"] = _sections(md)
    if len(out["sections"]) < 5:
        out["problems"].append(f"too few sections: {len(out['sections'])}")

    if not spec_path.exists():
        out["problems"].append("missing json-render spec")
        return out
    spec = json.load(open(spec_path, encoding="utf-8"))
    errors = catalog.validate_spec(spec)
    out["spec_elements"] = len(spec.get("elements") or {})
    out["spec_errors"] = errors
    if errors:
        out["problems"].append(f"{len(errors)} spec validation errors")
    if out["spec_elements"] < 20:
        out["problems"].append(f"too few spec elements: {out['spec_elements']}")

    meta = json.load(open(meta_path, encoding="utf-8")) if meta_path.exists() else {}
    out["meta"] = {k: meta.get(k) for k in
                   ("md_chars", "spec_elements", "knowledge_chars", "corpus_chars",
                    "md_finish_reason", "spec_finish_reason", "truncated",
                    "elapsed_seconds", "validation_errors", "consistency_warnings")}
    if meta.get("truncated"):
        out["problems"].append("LLM response truncated by max_tokens")

    # ---- Comparison with serial baseline ----
    b_md_path = Path(baseline_dir) / f"report_{app_no}.md"
    b_spec_path = Path(baseline_dir) / f"report_{app_no}.json"
    if b_md_path.exists():
        b_md = b_md_path.read_text(encoding="utf-8")
        b_secs = _sections(b_md)
        out["baseline_sections"] = b_secs
        missing = [s for s in b_secs if s not in out["sections"]]
        out["missing_sections"] = missing
        # Only the same prompt version counts missing sections as problems; older versions are recorded for reference
        if missing and baseline_kind == "same_prompt":
            out["problems"].append(f"missing baseline sections: {missing}")

        b_nums, n_nums = _numbers(b_md), _numbers(md)
        # Measure only the share of baseline key values retained (new reports may add values)
        keep = len(b_nums & n_nums) / max(len(b_nums), 1)
        out["number_retention"] = round(keep, 3)
        if baseline_kind == "same_prompt" and keep < 0.75:
            out["problems"].append(f"key-number retention too low: {keep:.0%}")

        out["md_chars_ratio"] = round(len(md) / max(len(b_md), 1), 2)
    if b_spec_path.exists():
        b_spec = json.load(open(b_spec_path, encoding="utf-8"))
        b_el = len(b_spec.get("elements") or {})
        out["baseline_spec_elements"] = b_el
        out["spec_elements_ratio"] = round(out["spec_elements"] / max(b_el, 1), 2)
        # Whether to inject the dashboard / abnormal-findings table
        types = {e.get("type") for e in (spec.get("elements") or {}).values()}
        out["has_dashboard"] = "Dashboard" in types or any(
            "dashboard" in str(e.get("props", {}).get("title", "")).lower()
            for e in (spec.get("elements") or {}).values()
        )
        if not out["has_dashboard"]:
            out["problems"].append("missing risk dashboard")
    return out



# ===================== Offline mode: no database dependency =====================
def _latest_batch_dir():
    """Find the most recent fully fetched CSV directory (containing batch_results.json)."""
    dirs = sorted(Path("batches").glob("batch_*"),
                  key=lambda d: d.stat().st_mtime, reverse=True)
    for d in dirs:
        if (d / "batch_results.json").exists() and (d / "PES_RESULT.csv").exists():
            return d
    return None


def _read_csv(path):
    import pandas as pd
    if not path.exists():
        return None
    return pd.read_csv(path, dtype=str, encoding="utf-8-sig").fillna("")


def _build_archive_from_csv(batch_dir, app_no):
    """Rebuild the patient record from fetched CSVs (same structure as the DB fetch_patient_archive)."""
    apply = _read_csv(batch_dir / "PES_APPLY.csv")
    patient = _read_csv(batch_dir / "PES_PATIENT.csv")
    result = _read_csv(batch_dir / "PES_RESULT.csv")
    desc = _read_csv(batch_dir / "PES_RESULT_DESC.csv")
    app_no = str(app_no).strip()
    rec = {"app_no": app_no}
    if apply is not None and not apply.empty:
        row = apply[apply["app_no"] == app_no]
        if not row.empty:
            r = row.iloc[0]
            rec.update(pat_id=r.get("pat_id", ""),
                       app_pat_name=r.get("app_pat_name", ""),
                       app_pat_age=r.get("app_pat_age", ""))
            if patient is not None and not patient.empty:
                p = patient[patient["pat_id"] == r.get("pat_id", "")]
                if not p.empty:
                    pr = p.iloc[0]
                    rec.update(pat_name=pr.get("pat_name", ""),
                               pat_sex=pr.get("pat_sex", ""),
                               pat_nation=pr.get("pat_nation", ""),
                               pat_address=pr.get("pat_address", ""))
    lab = []
    if result is not None and not result.empty:
        for _, r in result[result["app_no"] == app_no].iterrows():
            name = r.get("res_itm_name", "")
            lab.append({
                "detail_id": r.get("detail_id", ""),
                "res_itm_name": name,
                "fieldName": name,
                "det_com_name": "",
                "det_dep_name": "",
                "res_value": r.get("res_value", ""),
                "res_unit": r.get("res_unit", ""),
                "res_exp": r.get("res_exp", ""),
                "res_refl": r.get("res_refl", ""),
                "res_refu": r.get("res_refu", ""),
                "res_sum_flag": r.get("res_sum_flag", ""),
            })
    ds = []
    if desc is not None and not desc.empty:
        for _, r in desc[desc["app_no"] == app_no].iterrows():
            text = str(r.get("res_result", "") or "").strip()
            if text:
                ds.append({"detail_id": r.get("detail_id", ""),
                           "det_com_name": "", "det_dep_name": "",
                           "res_result": text})
    rec["lab_items"] = lab
    rec["desc_items"] = ds
    return rec


def _db_reachable(timeout=5):
    try:
        import pymssql
        c = pymssql.connect(
            server=os.getenv("MEDICAL_DB_SERVER", ""),
            user=os.getenv("MEDICAL_DB_USER", ""),
            password=os.getenv("MEDICAL_DB_PASSWORD", ""),
            database=os.getenv("MEDICAL_DB_NAME", ""),
            charset="utf8", login_timeout=timeout, timeout=timeout,
        )
        c.close()
        return True
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patients", type=int, default=16)
    ap.add_argument("--inflight", type=int, default=128,
                    help="Global cap on in-flight tasks (default 128)")
    ap.add_argument("--concurrency", type=int, default=0,
                    help="Legacy alias for --inflight")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    inflight = args.concurrency or args.inflight
    os.environ["REPORT_LLM_MAX_INFLIGHT"] = str(inflight)
    os.environ["REPORT_LLM_MAX_CONCURRENCY"] = str(inflight)

    import report_generator as RG
    import pipeline_runner as PR

    print(f"[test] LLM endpoint={RG.LLM_URL} model={RG.LLM_MODEL}")
    print(f"[test] global task concurrency={RG.REPORT_LLM_MAX_INFLIGHT} "
          f"wait cap={RG.REPORT_LLM_WAIT_TIMEOUT:.0f}s "
          f"per-request timeout={RG.LLM_TIMEOUT:.0f}s "
          f"task retries={RG.REPORT_TASK_RETRIES}")

    patients = _collect_patients(args.patients)
    if len(patients) < args.patients:
        print(f"[test] WARNING: only {len(patients)} usable patients found")
    app_nos = [p["app_no"] for p in patients]
    tag = args.tag or f"task_{time.strftime('%Y%m%d_%H%M%S')}_conc{args.concurrency}"
    work = _prepare_workdir(patients, tag)
    print(f"[test] test directory: {work}")

    # ---- Risk-scoring chain: prefer the database, fall back to cache when unreachable ----
    t0 = time.time()
    batch_dir = _latest_batch_dir()
    vectors = {}
    per_patient = {}
    if _db_reachable():
        try:
            bp = PR._get_batch_processor()
            batch_results = bp.process_batch(app_nos)
            vectors, per_patient = PR.build_patient_vectors(batch_results)
            print(f"[test] risk scoring complete: {len(batch_results)} records, "
                  f"elapsed {time.time()-t0:.1f}s")
        except Exception as e:
            print(f"[test] risk scoring failed; using cache: {e}")
            batch_results = []
    else:
        print("[test] Database unreachable; entering offline mode (cached CSV / batch_results)")
        batch_results = []

    if not batch_results:
        assert batch_dir is not None, "no usable cached batch"
        cached = json.load(open(batch_dir / "batch_results.json", encoding="utf-8"))
        by_no = {str(r["app_no"]): r for r in cached}
        batch_results = [by_no[a] for a in app_nos if a in by_no]
        vectors, per_patient = PR.build_patient_vectors(batch_results)
        print(f"[test] loaded {len(batch_results)} risk results from cache: {batch_dir}")
        # Restore the record from cached CSVs too, bypassing the database entirely
        import jev27b_decider as J
        J.fetch_patient_archive = lambda app_no, _d=batch_dir: _build_archive_from_csv(_d, app_no)

    records = [{
        "eid": p["app_no"],
        "status": "SUCCESS",
        "subgraph_files": {"edges_json": p["work_edges"]},
    } for p in patients]

    # ---- Concurrent report generation ----
    progress = []

    def cb(done, total, app_no, status):
        progress.append((done, total, app_no, status))
        print(f"[progress] {done}/{total} {app_no} -> {status}")

    t1 = time.time()
    records = RG.generate_reports(records, batch_results, per_patient=per_patient,
                                  progress_cb=cb)
    elapsed = time.time() - t1
    print(f"[test] report generation complete in {elapsed:.1f}s")

    # ---- Validation ----
    results = []
    for p, rec in zip(patients, records):
        r = _validate_one(p["app_no"], p["work_edges"], p["baseline_dir"],
                          p["baseline_kind"])
        r["report_status"] = rec.get("report_status")
        if not str(rec.get("report_status") or "").startswith("SUCCESS"):
            r["problems"].append(f"abnormal report status: {rec.get('report_status')}")
        results.append(r)

    # Progress-callback check
    idxs = [x[0] for x in progress]
    progress_ok = (
        len(progress) == len(patients)
        and idxs == sorted(idxs)
        and idxs == list(range(1, len(patients) + 1))
        and all(x[1] == len(patients) for x in progress)
    )

    ok = [r for r in results if not r["problems"]]
    summary = {
        "tag": tag,
        "inflight": inflight,
        "patients": len(patients),
        "elapsed_seconds": round(elapsed, 1),
        "avg_seconds_per_patient": round(elapsed / max(len(patients), 1), 1),
        "success": len(ok),
        "failed": len(results) - len(ok),
        "progress_callback_ok": progress_ok,
        "progress_events": len(progress),
        "results": results,
    }
    out_file = work / "concurrency_test_summary.json"
    out_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                        encoding="utf-8")

    print("\n================ TEST RESULTS ================")
    print(f"task concurrency cap: {inflight}  patients: {len(patients)}")
    print(f"total time: {elapsed:.1f}s  per patient: {elapsed/max(len(patients),1):.1f}s")
    print(f"succeeded: {len(ok)}/{len(results)}  progress callback OK: {progress_ok}")
    for r in results:
        flag = "OK " if not r["problems"] else "BAD"
        print(f"  [{flag}] {r['app_no']} status={r.get('report_status')} "
              f"md={r.get('md_chars')} spec={r.get('spec_elements')} "
              f"number retention={r.get('number_retention')} "
              f"{'problems: ' + '; '.join(r['problems']) if r['problems'] else ''}")
    print(f"details: {out_file}")
    return 0 if (len(ok) == len(results) and progress_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
