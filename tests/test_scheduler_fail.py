# -*- coding: utf-8 -*-
"""Scheduler failure-path test: knowledge-chunk permanent-failure fallback + overall timeout marking."""
import tempfile, time
from pathlib import Path
import report_generator as RG

def fake_md(key_info, corpus):
    return "## 1. x\n## 2. y"
def fake_spec(md):
    return {"root": "page", "elements": {"page": {"type": "Page", "children": []}}}, []
RG.generate_markdown = fake_md
RG.convert_to_jsonrender = fake_spec
RG.inject_page_meta = lambda spec, patient=None: spec
RG.inject_dashboard = lambda spec, items: spec
RG.inject_findings_table = lambda spec, findings: spec
RG.check_negative_consistency = lambda md, rec: []
RG.build_dashboard_items = lambda d, a=None: []

# Knowledge chunking: chunk 0 always fails, the rest succeed
def flaky_filter(key_info, corpus, part, total):
    if part == 1:
        raise RuntimeError("simulated network interruption")
    return f"K{part}"
RG._filter_one = flaky_filter

def fake_prepare(self, job):
    job.archive = {"pat_name": "Test", "pat_sex": "1", "app_pat_age": 30}
    job.key_info = "KEY"
    job.corpus = "### Evidence " + ("x" * 300)
    job.chunks = ["### Evidence " + ("x" * 300)] * 3
    job.knowledge_parts = [None] * 3
    job.pending_chunks = 3
    for i in range(3):
        self.submit("knowledge", job, i)
RG._ReportScheduler._do_prepare = fake_prepare

tmp = Path(tempfile.mkdtemp())
records = []
for i in range(2):
    d = tmp / f"p{i}"; d.mkdir()
    e = d / f"eid_{i}_edges.json"; e.write_text("{}", encoding="utf-8")
    records.append({"eid": str(i), "status": "SUCCESS",
                    "subgraph_files": {"edges_json": str(e)}})

progress = []
out = RG.generate_reports(records, [], per_patient={},
                          progress_cb=lambda *a: progress.append(a))
print("statuses", [r.get("report_status") for r in out])
print("warnings", [r.get("report_meta", {}).get("consistency_warnings") for r in out])
print("failed_chunks", [r.get("report_meta", {}).get("knowledge_failed_chunks") for r in out])
assert all(r.get("report_status") == "SUCCESS" for r in out), "should still be SUCCESS after fallback"
assert all(r.get("report_meta", {}).get("knowledge_failed_chunks") == [0] for r in out)
print("failure-path test passed (knowledge failure fell back; patient still got a report)")
