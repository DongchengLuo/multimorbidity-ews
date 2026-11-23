# -*- coding: utf-8 -*-
"""
Comorbidity subgraph construction module (integrated)
================================================================================
Role: accept the patient risk vector ({English standard disease name: risk value}) and call
Done by hybrid_retrieval_engine.MultimorbidityRetrievalEngine:
  1. Steady-state comorbidity module matching (Fast Lane, anchor-library hit)
  2. Dynamic graph retrieval and explanatory subgraph generation (Smart Lane, patient-weighted RWR)
  3. Persist full subgraph data (graphml / edges.json / png)

Upstream input: disease-level risk tiers from cli.py (BatchProcessor), mapped
assembled after mapping to English standard disease names via disease_name_mapping.json; multiple models per disease
First unify into one tier (majority vote + higher tier on ties + high-risk veto); only unified tiers at
Only medium-risk and above diseases enter the patient risk vector; the value is "tier base weight x vote support
coefficient" rather than stacking values (see pipeline_runner.build_patient_vectors).

Fast Lane (steady-state comorbidity module matching) is disabled: the score formula weight_sum/len(union) and the threshold
0.6 is structurally unreachable under tier weights, and the module library has only 2 large communities with insufficient overlap with patient vectors;
All patients use Smart Lane patient-weighted RWR dynamic retrieval.

KG path defaults to data/kg/knowledge_graph_normalized_with_weights.json inside the project,
Overridable via KG_PATH / KG_S2_PATH / KG_DIR (to reuse external large files).

Called as a module by pipeline_runner.py:
  from subgraph_construct import retrieve_for_patients
  records = retrieve_for_patients({app_no: {disease: risk}}, output_dir)

Can also run standalone (original behavior: retrieve per patient from the CSV risk matrix):
  python subgraph_construct.py [risk_matrix.csv]
================================================================================
"""
import json
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless environment; must be set before importing pyplot

PROJECT_ROOT = Path(__file__).resolve().parent

# ===================== Path configuration =====================
# The KG and comorbidity-combination tables default to data/kg/; override via env vars to reuse external large files.
#   KG_PATH  -> environment variable KG_PATH
#   S2_PATH  -> environment variable KG_S2_PATH
KG_DIR = Path(os.getenv("KG_DIR", str(PROJECT_ROOT / "data" / "kg")))
KG_PATH = os.getenv(
    "KG_PATH",
    str(KG_DIR / "knowledge_graph_normalized_with_weights.json"),
)
S2_PATH = os.getenv(
    "KG_S2_PATH",
    str(KG_DIR / "comorbidity_combinations_S2.csv"),
)
MAPPING_PATH = str(PROJECT_ROOT / 'mapping.csv')

# ===================== Engine singleton (73MB KG, loaded once) =====================
_ENGINE = None


def get_engine():
    """Lazily load and cache the retrieval engine (Fast Lane disabled; no steady-state module library)."""
    global _ENGINE
    if _ENGINE is None:
        from hybrid_retrieval_engine import MultimorbidityRetrievalEngine
        print("[subgraph_construct] initializing retrieval engine (loading KG; Fast Lane disabled, skipping module-library build)...")
        t0 = time.time()
        engine = MultimorbidityRetrievalEngine(KG_PATH, S2_PATH, MAPPING_PATH)
        if engine.ENABLE_FAST_LANE:
            engine.build_offline_module_library(top_percentile=0.90)
        else:
            print("[subgraph_construct] ENABLE_FAST_LANE=False; skipping steady-state module-library build")
        print(f"[subgraph_construct] engine initialized in {time.time() - t0:.1f}s")
        _ENGINE = engine
    return _ENGINE


def retrieve_for_patient(patient_risk_vector, output_dir=None, eid=None):
    """
    Retrieve for a single patient, optionally persisting the subgraph.

    Success shape returned by the engine:
      - "SUCCESS": smart lane; returns the explanatory subgraph from patient-weighted RWR dynamic retrieval.
      ("SUCCESS (FAST LANE)" only appears when the engine has ENABLE_FAST_LANE=True,
        currently disabled.)

    Parameters:
        patient_risk_vector: {English standard disease name: RWR weight}; weights derive from the risk tier
            mapped (medium 0.50 / high 0.75 / very high 1.00, then by vote
            refined within +/-20% by support rate), e.g. {"Lung Cancer": 0.75, "Hypertension": 0.50}
        output_dir:          subgraph output directory (None = do not persist)
        eid:                 patient identifier (app_no), used for file naming
    Returns:
        dict retrieval record (status, matched module, subgraph file paths, etc.)
    """
    engine = get_engine()
    result = engine.retrieve_for_patient(patient_risk_vector)
    status = result.get("status")

    record = {
        "eid": eid,
        "patient_risk_vector": patient_risk_vector,
        "status": status,
        "message": result.get("message", ""),
        "subgraph_files": {},
    }

    # ---- Fast lane: steady-state comorbidity module hit (currently disabled, code path retained) ----
    if status == "SUCCESS (FAST LANE)":
        record["hit_module"] = result.get("matched_module_id")
        record["core_mechanism"] = result.get("core_mechanism")
        record["suggested_interpretation"] = result.get("suggested_interpretation")
        return record

    # ---- Smart lane: dynamic graph retrieval, generate explanatory subgraph ----
    if status == "SUCCESS" and "subgraph" in result:
        subgraph = result["subgraph"]
        record["subgraph_nodes"] = subgraph.number_of_nodes()
        record["subgraph_edges"] = subgraph.number_of_edges()

        if output_dir is not None:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            try:
                edges_path = engine.save_subgraph_complete(
                    subgraph, patient_risk_vector, str(out), eid
                )
                # save_subgraph_complete returns the edges.json path; graphml / png sit alongside it
                base = Path(edges_path).name.replace("_edges.json", "")
                files = {}
                for suffix, key in [
                    (".graphml", "graphml"),
                    ("_edges.json", "edges_json"),
                    (".png", "png"),
                ]:
                    p = out / (base + suffix)
                    if p.exists():
                        files[key] = str(p)
                record["subgraph_files"] = files
            except Exception as e:
                record["subgraph_files"] = {"error": f"subgraph persist failed: {e}"}

    return record


def retrieve_for_patients(patient_vectors, output_dir, progress_cb=None):
    """
    Batch-run patient risk-vector retrieval.

    Parameters:
        patient_vectors: {app_no: {English standard disease name: risk value}}
        output_dir:     subgraph output root (one subdirectory per app_no)
        progress_cb:    optional callback fn(done, total, app_no, record)
    Returns:
        list[dict] retrieval record per patient
    """
    records = []
    items = list(patient_vectors.items())
    total = len(items)
    for i, (app_no, vector) in enumerate(items, 1):
        try:
            record = retrieve_for_patient(
                vector, output_dir=os.path.join(str(output_dir), str(app_no)), eid=app_no
            )
        except Exception as e:
            record = {
                "eid": app_no,
                "patient_risk_vector": vector,
                "status": "ERROR",
                "message": f"retrieval error: {e}",
                "subgraph_files": {},
            }
        records.append(record)
        if progress_cb:
            try:
                progress_cb(i, total, app_no, record)
            except Exception:
                pass
    return records


# ===================== Standalone entry point (original script behavior retained) =====================
if __name__ == "__main__":
    import pandas as pd

    csv_path = sys.argv[1] if len(sys.argv) > 1 else str(PROJECT_ROOT / "risk_matrix.csv")
    output_dir = sys.argv[2] if len(sys.argv) > 2 else str(PROJECT_ROOT / "debug_output" / "csv_mode")

    if not os.path.exists(csv_path):
        print(f"[subgraph_construct] risk-matrix CSV not found: {csv_path}")
        print("[subgraph_construct] run the full pipeline via pipeline_runner.py (app_no -> scoring -> subgraph)")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    print(f"[subgraph_construct] read {len(df)} rows of patient risk matrix: {csv_path}")

    vectors = {}
    for _, row in df.iterrows():
        eid = str(row.iloc[0])
        vec = {}
        for col in df.columns[1:]:
            risk_val = row[col]
            if pd.notna(risk_val) and float(risk_val) > 0:
                vec[col] = float(risk_val)
        if vec:
            vectors[eid] = vec

    print(f"[subgraph_construct] valid patient risk vectors: {len(vectors)}, output dir: {output_dir}")
    records = retrieve_for_patients(vectors, output_dir)
    ok = sum(1 for r in records if r.get("status") == "SUCCESS")
    print(f"[subgraph_construct] done: {ok} succeeded / {len(records)}")
