# -*- coding: utf-8 -*-
"""Scheduler unit test: no real LLM; verifies the task queue/dependencies/wrap-up/progress callbacks."""
import json, os, tempfile, time, threading
from pathlib import Path
import report_generator as RG

# ---- Stub: replace all LLM calls with a short sleep ----
def fake_filter(key_info, corpus, part, total):
    time.sleep(0.05)
    return f"K{part}"
def fake_md(key_info, corpus):
    time.sleep(0.05)
    return "## 1. Individual Risk Profile\n## 2. Recommendations\n## 3. Follow-up\n## 4. Medication\n## 5. Lifestyle"
def fake_spec(md):
    time.sleep(0.05)
    return {"root": "page", "elements": {"page": {"type": "Page", "children": []}}}, []
RG._filter_one = fake_filter
RG.generate_markdown = fake_md
RG.convert_to_jsonrender = fake_spec
RG.inject_page_meta = lambda spec, patient=None: spec
RG.inject_dashboard = lambda spec, items: spec
RG.inject_findings_table = lambda spec, findings: spec
RG.check_negative_consistency = lambda md, rec: []
RG.build_dashboard_items = lambda d, a=None: []

# ---- Stub prepare: skip the database, build the corpus and chunk it directly ----
def fake_prepare(self, job):
    job.archive = {"pat_name": "Test", "pat_sex": "1", "app_pat_age": 30}
    job.key_info = "KEY"
    job.corpus = "### Evidence " + ("x" * 300)
    job.chunks = ["### Evidence " + ("x" * 300)] * 3   # force 3 chunks
    job.knowledge_parts = [None] * 3
    job.pending_chunks = 3
    for i in range(3):
        self.submit("knowledge", job, i)
RG._ReportScheduler._do_prepare = fake_prepare

tmp = Path(tempfile.mkdtemp())
records = []
for i in range(16):
    d = tmp / f"p{i}"
    d.mkdir()
    edges = d / f"eid_{i}_edges.json"
    edges.write_text("{}", encoding="utf-8")
    records.append({"eid": str(i), "status": "SUCCESS",
                    "subgraph_files": {"edges_json": str(edges)}})

progress = []
def cb(done, total, app_no, status):
    progress.append((done, total, app_no, status))

t0 = time.time()
out = RG.generate_reports(records, [], per_patient={}, progress_cb=cb)
elapsed = time.time() - t0
statuses = [r.get("report_status") for r in out]
print("elapsed", round(elapsed, 2))
print("statuses", statuses)
print("progress events", len(progress), "monotonic", [p[0] for p in progress] == sorted(p[0] for p in progress))
print("all success", all(s == "SUCCESS" for s in statuses))
print("meta files", len(list(tmp.glob("*/report_*_meta.json"))))
print("md files", len(list(tmp.glob("*/report_*.md"))))
print("json files", len(list(tmp.glob("*/report_*.json"))))
assert all(s == "SUCCESS" for s in statuses), statuses
assert len(progress) == 16
assert [p[0] for p in progress] == list(range(1, 17))
print("unit test passed")
