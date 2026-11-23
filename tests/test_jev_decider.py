# -*- coding: utf-8 -*-
"""jev27b_decider logic self-test (does not call the real decision service)"""
import json
import shutil
import tempfile
from pathlib import Path

import jev27b_decider as J

# ---------- 1. Negative denoising ----------
record = {
    "app_no": "90000001", "pat_id": "P1", "app_pat_name": "Zhang San", "app_pat_age": 52,
    "pat_name": "Zhang San", "pat_sex": "male", "pat_nation": "Han", "pat_address": "Some City",
    "lab_items": [
        {"fieldName": "height", "res_value": "163.5", "res_unit": "cm", "res_exp": "", "det_dep_name": "Internal Medicine", "det_com_name": "General Exam"},
        {"fieldName": "weight", "res_value": "45.7", "res_unit": "kg", "res_exp": "", "det_dep_name": "Internal Medicine", "det_com_name": "General Exam"},
        {"fieldName": "BMI", "res_value": "17.1", "res_unit": "kg/m2", "res_exp": "18.5-23.9", "det_dep_name": "Internal Medicine", "det_com_name": "General Exam"},
        {"fieldName": "fasting glucose", "res_value": "6.70", "res_unit": "mmol/L", "res_exp": "3.90-6.10", "det_dep_name": "Laboratory", "det_com_name": "Biochemistry"},
        {"fieldName": "HbA1c", "res_value": "10.5", "res_unit": "%", "res_exp": "4.0-6.0", "det_dep_name": "Laboratory", "det_com_name": "Biochemistry"},
        {"fieldName": "WBC", "res_value": "6.2", "res_unit": "×10~9/L", "res_exp": "3.50-9.50", "det_dep_name": "Laboratory", "det_com_name": "Blood Routine"},
        {"fieldName": "hemoglobin", "res_value": "138", "res_unit": "g/L", "res_exp": "115-150", "det_dep_name": "Laboratory", "det_com_name": "Blood Routine"},
        {"fieldName": "urine glucose", "res_value": "4+", "res_unit": "", "res_exp": "negative", "det_dep_name": "Laboratory", "det_com_name": "Urinalysis"},
        {"fieldName": "urine protein", "res_value": "negative", "res_unit": "", "res_exp": "negative", "det_dep_name": "Laboratory", "det_com_name": "Urinalysis"},
        {"fieldName": "bone density T-score", "res_value": "-1.21", "res_unit": "", "res_exp": ">-1", "det_dep_name": "Orthopedics", "det_com_name": "Bone Density"},
        {"fieldName": "systolic BP", "res_value": "102", "res_unit": "mmHg", "res_exp": "90-140", "det_dep_name": "Internal Medicine", "det_com_name": "Blood Pressure"},
        {"fieldName": "fasting insulin", "res_value": "0.69", "res_unit": "μU/mL", "res_exp": "2.6-24.9", "det_dep_name": "Laboratory", "det_com_name": "Biochemistry"},
    ],
    "desc_items": [
        {"det_com_name": "Abdominal Ultrasound", "det_dep_name": "Ultrasound", "res_result": "Possible bilateral kidney stones (0.3cm), prostate cyst"},
        {"det_com_name": "Chest DR", "det_dep_name": "Radiology", "res_result": "No obvious abnormality in both lungs; heart shadow not enlarged"},
        {"det_com_name": "ENT", "det_dep_name": "ENT", "res_result": "Chronic pharyngitis"},
        {"det_com_name": "Ophthalmology", "det_dep_name": "Ophthalmology", "res_result": "Reduced vision with amblyopia"},
        {"det_com_name": "Echocardiography", "det_dep_name": "Ultrasound", "res_result": "Mild tricuspid regurgitation"},
        {"det_com_name": "Thyroid Ultrasound", "det_dep_name": "Ultrasound", "res_result": "No obvious thyroid abnormality"},
    ],
}

text, stats = J.build_archive_text(record)
print("=== denoising stats ===")
print(json.dumps(stats, ensure_ascii=False, indent=2))
print("\n=== record text ===")
print(text)

# ---------- 1.1 Denoising rule assertions ----------
# Exclude only normal items; never drop abnormal ones
kept_names = {it["fieldName"] for it in record["lab_items"] if not J.is_normal_lab_item(it)}
for must_keep in ["BMI", "fasting glucose", "HbA1c", "urine glucose", "bone density T-score", "fasting insulin"]:
    assert must_keep in kept_names, f"abnormal item wrongly removed: {must_keep}"
# Core safety exclusion band + clearly-normal characters: normal items must be removed
for must_drop in ["WBC", "hemoglobin", "systolic BP", "urine protein", "HBsAg"]:
    assert must_drop not in kept_names, f"normal item not removed: {must_drop}"
# Normal-word exclusion does not apply to descriptive exam items: desc_items are always kept
assert stats["desc_kept"] == stats["desc_total"] == 6, "descriptive items must all be retained"
assert "No obvious abnormality in both lungs" in text, "descriptive normal reports must be retained in the record"
assert "Mild tricuspid regurgitation" in text
print("\n[assertion passed] denoising rule: exclude normal only; descriptive items all retained")

# ---------- 2. Edge information text (all doc_sources + full-text summaries) ----------
src = "debug_output/task_20261019_153155_8be029/90000001/eid_90000001_edges.json"
data = json.load(open(src, encoding="utf-8"))
edge = data["edge_details"][0]
et = J.build_edge_text(edge)
n_docs = len(edge["full_kg_edge_data"]["document_sources"])
print("\n=== edge information text ===")
print(f"documents: {n_docs} | text length: {len(et)}")
print(et[:1200])
print("...")
print(et[-600:])

# ---------- 3. Embed scoring results (mock scores) ----------
tmp = Path(tempfile.mkdtemp())
shutil.copy(src, tmp / "eid_90000001_edges.json")
g = tmp / "eid_90000001.graphml"
shutil.copy("debug_output/task_20261019_153155_8be029/90000001/eid_90000001.graphml", g)

scored = {}
for e in data["edge_details"]:
    scored[(e["source_term"], e["target_term"])] = {
        "score": 4.0, "probabilities": {"0": 0.01, "5": 0.9}, "raw": {"score": 4.0},
    }
n = J._embed_into_edges_json(tmp / "eid_90000001_edges.json", scored, "90000001", stats)
print(f"\n=== embedded into edges.json: {n} edges ===")
back = json.load(open(tmp / "eid_90000001_edges.json", encoding="utf-8"))
print("top-level decision_model:", json.dumps(back["decision_model"], ensure_ascii=False)[:300])
print("first-edge attributes:", json.dumps(back["edge_details"][0]["subgraph_edge_data"], ensure_ascii=False)[:300])
ok = J._embed_into_graphml(str(g), scored)
print("graphml write-back:", ok)
import networkx as nx
gg = nx.read_graphml(str(g))
u, v = list(gg.edges())[0]
print("graphml first-edge attributes:", dict(gg.edges[u, v]))
