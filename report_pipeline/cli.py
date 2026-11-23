"""Command-line entry point.

Usage::

    python -m report_pipeline --app-no 90000001
    python -m report_pipeline --app-no 90000001 --batch batches/batch_200_xxx --pdf
    python -m report_pipeline --app-no 90000001 --score        # run scoring first, then the report
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def find_batch(explicit=None, app_no=None):
    if explicit:
        return explicit if os.path.isabs(explicit) else os.path.join(ROOT, explicit)
    cands = sorted(glob.glob(os.path.join(ROOT, "batches", "batch_*")),
                   key=os.path.getmtime, reverse=True)
    cands = [c for c in cands if os.path.exists(os.path.join(c, "batch_results.json"))]
    if app_no:
        key = str(app_no).lstrip("0")
        for c in cands:
            try:
                data = json.load(open(os.path.join(c, "batch_results.json"), encoding="utf-8"))
            except Exception:
                continue
            if any(str(p.get("app_no", "")).lstrip("0") == key for p in data):
                return c
    if cands:
        return cands[0]
    raise SystemExit("batch directory not found")


def find_md(app_no, explicit=None):
    if explicit:
        return explicit if os.path.isabs(explicit) else os.path.join(ROOT, explicit)
    key = str(app_no).lstrip("0")
    pats = [
        os.path.join(ROOT, "debug_output", "task_*", f"{app_no}", f"report_{app_no}.md"),
        os.path.join(ROOT, "debug_output", "task_*", f"{key}", f"report_{key}.md"),
    ]
    hits = []
    for p in pats:
        hits += glob.glob(p)
    if not hits:
        return None
    return max(hits, key=os.path.getmtime)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="report_pipeline")
    ap.add_argument("--app-no", required=True)
    ap.add_argument("--batch", default=None)
    ap.add_argument("--md", default=None)
    ap.add_argument("--out", default=None, help="Output directory (default: project root)")
    ap.add_argument("--pdf", action="store_true")
    ap.add_argument("--score", action="store_true", help="Run the scoring chain first")
    ap.add_argument("--check", action="store_true", help="Run the information-completeness check")
    args = ap.parse_args(argv)

    app_no = str(args.app_no).zfill(8)

    if args.score:
        sys.path.insert(0, ROOT)
        from batch_processor import BatchProcessor

        bp = BatchProcessor()
        bp.process_batch([app_no])
        batch_dir = bp.last_batch_dir
        print(f"[score] batch directory {batch_dir}")
    else:
        batch_dir = find_batch(args.batch, app_no)

    md_path = find_md(app_no, args.md)
    if not md_path:
        raise SystemExit(f"md report for {app_no} not found; specify it with --md")

    from .cohort import Cohort
    from .render import build_html

    cohort = Cohort(batch_dir)
    patient = cohort.patient(app_no)
    md_text = open(md_path, encoding="utf-8").read()
    html = build_html(patient, cohort, md_text)

    out_dir = args.out or ROOT
    os.makedirs(out_dir, exist_ok=True)
    html_path = os.path.join(out_dir, f"report_{app_no}.html")
    open(html_path, "w", encoding="utf-8").write(html)
    print(f"[html] {html_path} ({len(html)} bytes)")

    if args.check:
        from .check import check

        ok = check(md_text, html)
        print("[check]", "passed" if ok else "failed")

    if args.pdf:
        from .pdf import html_to_pdf

        pdf_path = os.path.join(out_dir, f"report_{app_no}.pdf")
        html_to_pdf(html, pdf_path)
        print(f"[pdf] {pdf_path} ({os.path.getsize(pdf_path)} bytes)")

    return html_path


if __name__ == "__main__":
    main()
