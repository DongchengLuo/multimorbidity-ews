# app.py - fix SQL Server text-type errors + support GBK-encoded search + automatic secondary-field parsing
import json
import os
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory, Response
from flask_cors import CORS
import time
from datetime import datetime
import glob
import re
from pathlib import Path

# Watch-directory config: defaults to watched_app_nos in the project; can be set via env var
WATCH_DIR = os.getenv(
    "APP_NO_WATCH_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "watched_app_nos"),
)
# Fixed paths for the patient basic-info file and processed-record manifest
LOCAL_BASE_CSV = os.path.join(WATCH_DIR, "synced_base_info.csv")
PROCESSED_IDS_FILE = os.path.join(WATCH_DIR, ".processed_app_nos.txt")
app = Flask(__name__)
CORS(app)

# ---------------------------------------------------------------- Snapshot mode
# With SNAPSHOT_MODE=1 (default), all queries use the local de-identified snapshot in data/snapshot/,
# Never connect to the database. Set to 0 to restore direct SQL Server access.
SNAPSHOT_MODE = os.getenv("SNAPSHOT_MODE", "1").strip().lower() not in (
    "0", "false", "no", "off", ""
)
if SNAPSHOT_MODE:
    import snapshot_store

# Read database configuration (unused in snapshot mode)
DB_SERVER = os.getenv("MEDICAL_DB_SERVER", "")
DB_NAME = os.getenv("MEDICAL_DB_NAME", "")
DB_USER = os.getenv("MEDICAL_DB_USER", "")
DB_PASSWORD = os.getenv("MEDICAL_DB_PASSWORD", "")

BASE_FIELDS = {
    "app_no": ("a.app_no", "text"),
    "pat_id": ("a.pat_id", "text"),
    "pat_name": ("p.pat_name", "text"),
    "app_pat_age": ("a.app_pat_age", "number"),
    "pat_sex": ("p.pat_sex", "text"),       # male='1', female='2'
    "pat_nation": ("p.pat_nation", "text"),
    "pat_address": ("p.pat_address", "text"),
}

# Cache the secondary-field dict in memory to avoid rescanning on every page refresh
CATALOG_CACHE = None


def fix_mojibake(text):
    """Reuse the mojibake fix from db_fetcher.py: restore text misdecoded as latin-1 back to GBK"""
    if isinstance(text, str):
        try:
            return text.encode("latin-1").decode("gbk")
        except Exception:
            return text
    return text


def to_db_mojibake(text):
    """Reverse encoding: send Chinese WHERE conditions to SQL Server columns with Latin-1 collation"""
    if isinstance(text, str):
        try:
            return text.encode("gbk").decode("latin-1")
        except Exception:
            return text
    return text


def get_db_connection():
    # Lazy import: pymssql is not needed in snapshot mode (especially convenient on Windows)
    try:
        import pymssql
    except ImportError as exc:
        raise RuntimeError(
            "Direct database access requires pymssql: pip install pymssql; "
            "to use only the local snapshot, keep SNAPSHOT_MODE=1"
        ) from exc
    return pymssql.connect(
        server=DB_SERVER,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        charset="utf8"
    )


def query_dataframe(sql: str, params=None) -> pd.DataFrame:
    """Query directly with a pymssql cursor and convert to DataFrame, avoiding the pandas SQLAlchemy UserWarning"""
    conn = get_db_connection()
    try:
        cursor = conn.cursor(as_dict=True)
        if params:
            cursor.execute(sql, tuple(params))
        else:
            cursor.execute(sql)
        rows = cursor.fetchall()
        df = pd.DataFrame(rows) if rows else pd.DataFrame()
        if not df.empty:
            for col in df.select_dtypes(include=["object", "string"]).columns:
                df[col] = (
                    df[col]
                    .fillna("")
                    .apply(fix_mojibake)
                    .astype(str)
                    .str.replace("\t", "", regex=False)
                    .str.replace("\x00", "", regex=False)
                    .str.strip()
                )
        return df
    finally:
        conn.close()


def attach_items_to_records(records: list, max_app_nos: int = 60) -> list:
    """Batch-attach lab details (lab_items) and descriptive reports (desc_items) to the patient records shown on the current page"""
    if not records:
        return records

    target_records = records[:max_app_nos]
    app_nos = [str(r["app_no"]).strip() for r in target_records if str(r.get("app_no", "")).strip()]
    if not app_nos:
        return records

    # Snapshot mode: read from local CSVs without a database (report generation uses this too)
    if SNAPSHOT_MODE:
        import snapshot_store

        tables = snapshot_store.fetch_tables(app_nos)
        df_item = tables.get("PES_apply_item.csv")
        df_lab = tables.get("PES_RESULT.csv")
        df_desc = tables.get("PES_RESULT_DESC.csv")

        # Item and department names live in PES_apply_item, joined by detail_id
        item_cols = None
        if df_item is not None and not df_item.empty:
            item_cols = df_item[["detail_id", "det_com_name", "det_dep_name"]].copy()
            item_cols["detail_id"] = item_cols["detail_id"].astype(str).str.strip()
            item_cols = item_cols.drop_duplicates("detail_id")

        def _merge_item(df):
            df = df.copy()
            df["detail_id"] = df["detail_id"].astype(str).str.strip()
            if item_cols is not None:
                df = df.merge(item_cols, on="detail_id", how="left")
            for c in ("det_com_name", "det_dep_name"):
                if c not in df.columns:
                    df[c] = ""
                df[c] = df[c].fillna("")
            return df

        lab_map = {}
        if df_lab is not None and not df_lab.empty:
            df_lab = _merge_item(df_lab)
            df_lab["app_no"] = df_lab["app_no"].astype(str).str.strip()
            df_lab["res_itm_name"] = df_lab.get("res_itm_name", "").fillna("")
            df_lab["fieldName"] = df_lab.apply(
                lambda x: x["res_itm_name"] if str(x["res_itm_name"]).strip() else x["det_com_name"],
                axis=1,
            )
            df_lab = df_lab[(df_lab["fieldName"].astype(str).str.strip() != "")
                            & (df_lab["res_value"].notna())
                            & (df_lab["res_value"].astype(str).str.strip() != "")]
            cols = ["detail_id", "det_dep_name", "det_com_name", "res_itm_name", "fieldName",
                    "res_value", "res_unit", "res_exp", "res_refl", "res_refu", "res_sum_flag"]
            for c in cols:
                if c not in df_lab.columns:
                    df_lab[c] = ""
            for app_no, grp in df_lab.groupby("app_no"):
                lab_map[str(app_no).strip()] = grp[cols].to_dict(orient="records")

        desc_map = {}
        if df_desc is not None and not df_desc.empty:
            df_desc = _merge_item(df_desc)
            df_desc["app_no"] = df_desc["app_no"].astype(str).str.strip()
            df_desc = df_desc[(df_desc["det_com_name"].astype(str).str.strip() != "")
                              & (df_desc["res_result"].notna())
                              & (df_desc["res_result"].astype(str).str.strip() != "")]
            cols = ["detail_id", "det_dep_name", "det_com_name", "res_result"]
            for c in cols:
                if c not in df_desc.columns:
                    df_desc[c] = ""
            for app_no, grp in df_desc.groupby("app_no"):
                desc_map[str(app_no).strip()] = grp[cols].to_dict(orient="records")

        for r in records:
            key = str(r.get("app_no", "")).strip()
            r["lab_items"] = lab_map.get(key, [])
            r["desc_items"] = desc_map.get(key, [])
        return records

    placeholders = ",".join(["%s"] * len(app_nos))

    # 1. Batch-fetch lab results for these app_no values (see result_itm.py)
    lab_sql = f"""
        SELECT
            i.app_no, i.detail_id, i.det_dep_name, i.det_com_name,
            r.res_itm_name, r.res_value, r.res_unit, r.res_exp,
            r.res_refl, r.res_refu, r.res_sum_flag
        FROM dbo.PES_apply_item i
        INNER JOIN dbo.PES_RESULT r
            ON i.app_no = r.app_no AND i.detail_id = r.detail_id
        WHERE i.app_no IN ({placeholders})
          AND r.res_value IS NOT NULL
    """
    df_lab = query_dataframe(lab_sql, app_nos)
    lab_map = {}
    if not df_lab.empty:
        # Prefer det_com_name, then res_itm_name
        df_lab["fieldName"] = df_lab.apply(
            lambda x: x["res_itm_name"] if x["res_itm_name"] else x["det_com_name"],
            axis=1,
        )
        df_lab = df_lab[(df_lab["fieldName"] != "") & (df_lab["res_value"] != "")]
        for app_no, grp in df_lab.groupby("app_no"):
            lab_map[str(app_no).strip()] = grp[
                ["detail_id", "det_dep_name", "det_com_name", "res_itm_name", "fieldName",
                 "res_value", "res_unit", "res_exp", "res_refl", "res_refu", "res_sum_flag"]
            ].to_dict(orient="records")

    # 2. Batch-fetch descriptive reports for these app_no values (see desc_item.py)
    desc_sql = f"""
        SELECT
            i.app_no, i.detail_id, i.det_dep_name, i.det_com_name,
            d.res_result
        FROM dbo.PES_apply_item i
        INNER JOIN dbo.PES_RESULT_DESC d
            ON i.app_no = d.app_no AND i.detail_id = d.detail_id
        WHERE i.app_no IN ({placeholders})
          AND d.res_result IS NOT NULL
    """
    df_desc = query_dataframe(desc_sql, app_nos)
    desc_map = {}
    if not df_desc.empty:
        df_desc = df_desc[(df_desc["det_com_name"] != "") & (df_desc["res_result"] != "")]
        for app_no, grp in df_desc.groupby("app_no"):
            desc_map[str(app_no).strip()] = grp[
                ["detail_id", "det_dep_name", "det_com_name", "res_result"]
            ].to_dict(orient="records")

    for r in records:
        key = str(r.get("app_no", "")).strip()
        r["lab_items"] = lab_map.get(key, [])
        r["desc_items"] = desc_map.get(key, [])

    return records


@app.route("/", methods=["GET"])
def index():
    """Serve the front-end page (web.html in the same directory as this service)."""
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), "web.html")


@app.route("/vue.global.prod.js", methods=["GET"])
def vue_js():
    """Local Vue runtime (works on intranet/offline)."""
    return send_from_directory(
        os.path.dirname(os.path.abspath(__file__)), "vue.global.prod.js"
    )


@app.route("/report_preview", methods=["GET"])
def report_preview():
    """Standalone report preview page (for screenshots/sharing)."""
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), "report_preview.html")


@app.route("/report_viewer.js", methods=["GET"])
def report_viewer_js():
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), "report_viewer.js")


@app.route("/report_viewer.css", methods=["GET"])
def report_viewer_css():
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), "report_viewer.css")


@app.route("/ui_icons/<path:filename>", methods=["GET"])
def ui_icons(filename):
    """Serve UI icon assets (ui_icons directory)."""
    return send_from_directory(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui_icons"), filename
    )


@app.route("/api/report/<task_id>/<app_no>", methods=["GET"])
def api_report_spec(task_id, app_no):
    """Return the json-render spec of a patient's report."""
    base = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "debug_output", task_id, app_no
    )
    path = os.path.join(base, f"report_{app_no}.json")
    if not os.path.exists(path):
        return jsonify({"error": "Report not found (it may not have been generated)"}), 404
    with open(path, "r", encoding="utf-8") as f:
        return jsonify(json.load(f))


@app.route("/api/reports", methods=["GET"])
def api_reports():
    """Report center: list all generated reports."""
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug_output")
    items = []
    for task_dir in sorted(Path(base).glob("task_*"), reverse=True):
        if not task_dir.is_dir():
            continue
        for app_dir in sorted(task_dir.iterdir()):
            if not app_dir.is_dir():
                continue
            app_no = app_dir.name
            spec = app_dir / f"report_{app_no}.json"
            if not spec.exists():
                continue
            meta = {}
            meta_path = app_dir / f"report_{app_no}_meta.json"
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                except Exception:
                    meta = {}
            md_path = app_dir / f"report_{app_no}.md"
            md_chars = meta.get("md_chars")
            if md_chars is None and md_path.exists():
                try:
                    md_chars = len(md_path.read_text(encoding="utf-8"))
                except Exception:
                    md_chars = 0
            spec_elements = meta.get("spec_elements")
            if spec_elements is None:
                try:
                    spec_elements = len(json.loads(spec.read_text(encoding="utf-8")).get("elements") or {})
                except Exception:
                    spec_elements = 0
            items.append({
                "task_id": task_dir.name,
                "app_no": app_no,
                "patient": meta.get("patient") or {},
                "generated_at": meta.get("generated_at") or datetime.fromtimestamp(
                    spec.stat().st_mtime
                ).strftime("%Y-%m-%d %H:%M:%S"),
                "md_chars": md_chars,
                "spec_elements": spec_elements,
                "has_markdown": md_path.exists(),
                "has_knowledge": (app_dir / f"report_{app_no}_knowledge.md").exists(),
            })
    return jsonify({"items": items, "count": len(items)})


@app.route("/api/tasks", methods=["GET"])
def api_tasks():
    """Export queue: list all historical export tasks."""
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug_output")
    items = []
    for f in sorted(Path(base).glob("task_*/pipeline_result_*.json"), reverse=True):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        patients = d.get("patients") or []
        app_nos = [p.get("app_no") for p in patients if p.get("app_no")]
        items.append({
            "task_id": d.get("task_id") or f.parent.name,
            "created_at": d.get("created_at"),
            "source": d.get("source"),
            "count": len(patients),
            "app_nos": app_nos,
            "app_no_range": (f"{min(app_nos)} ~ {max(app_nos)}" if app_nos else ""),
            "enable_decide": d.get("enable_decide"),
            "enable_report": d.get("enable_report"),
            "retrieval_success": sum(1 for p in patients if p.get("retrieval_status") == "SUCCESS"),
            "report_success": sum(1 for p in patients if p.get("report_status") == "SUCCESS"),
        })
    return jsonify({"items": items, "count": len(items)})


@app.route("/api/tasks/<task_id>/download", methods=["GET"])
def api_task_download(task_id):
    """Download the basic-information collection (CSV) of an export task."""
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug_output", task_id)
    files = glob.glob(os.path.join(base, "pipeline_result_*.json"))
    if not files:
        return jsonify({"error": "Task not found"}), 404
    with open(files[0], "r", encoding="utf-8") as f:
        d = json.load(f)

    rows = [
        ["task_id", d.get("task_id")],
        ["created_at", d.get("created_at")],
        ["source", d.get("source")],
        ["enable_decide", d.get("enable_decide")],
        ["enable_report", d.get("enable_report")],
        ["patient_count", len(d.get("patients") or [])],
    ]
    lines = ["field,value"] + [f"{k},{v}" for k, v in rows]
    lines.append("")
    lines.append("app_no,pat_id,search_status,decision_status,report_status,risk_disease_count,subgraph_file")
    for p in (d.get("patients") or []):
        lines.append(",".join([
            str(p.get("app_no") or ""),
            str(p.get("pat_id") or ""),
            str(p.get("retrieval_status") or ""),
            str(p.get("decision_status") or ""),
            str(p.get("report_status") or ""),
            str(len(p.get("diseases") or [])),
            str((p.get("subgraph_files") or {}).get("edges_json") or ""),
        ]))
    csv_text = "\ufeff" + "\n".join(lines)
    return Response(
        csv_text,
        mimetype="text/csv; charset=utf-8",
        headers={"Content-Disposition": f"attachment; filename={task_id}.csv"},
    )


@app.route("/api/report/<task_id>/<app_no>/markdown", methods=["GET"])
def api_report_markdown(task_id, app_no):
    """Return the raw Markdown of a patient's report."""
    base = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "debug_output", task_id, app_no
    )
    path = os.path.join(base, f"report_{app_no}.md")
    if not os.path.exists(path):
        return jsonify({"error": "Report not found"}), 404
    return send_from_directory(base, f"report_{app_no}.md", mimetype="text/markdown")


@app.route("/api/report/<task_id>/<app_no>/html", methods=["GET"])
def api_report_html(task_id, app_no):
    """Return the single-file HTML of a patient's report (rendered by report_pipeline)."""
    base = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "debug_output", task_id, app_no
    )
    name = f"report_{app_no}.html"
    if not os.path.exists(os.path.join(base, name)):
        return jsonify({"error": "HTML report not found; please confirm report rendering is enabled"}), 404
    return send_from_directory(base, name, mimetype="text/html; charset=utf-8")


@app.route("/api/report/<task_id>/<app_no>/pdf", methods=["GET"])
def api_report_pdf(task_id, app_no):
    """Download a patient's vector PDF report (rendered by report_pipeline)."""
    base = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "debug_output", task_id, app_no
    )
    name = f"report_{app_no}.pdf"
    if not os.path.exists(os.path.join(base, name)):
        return jsonify({"error": "PDF report not found; please confirm weasyprint is installed and PDF rendering is enabled"}), 404
    return send_from_directory(base, name, mimetype="application/pdf")


@app.route("/api/field_catalog", methods=["GET"])
def get_field_catalog():
    """
    Parse the distinct secondary-field dictionary and sample values from the database (fully compatible with SQL Server text)
    """
    global CATALOG_CACHE
    force_refresh = request.args.get("refresh") == "1"
    if CATALOG_CACHE is not None and not force_refresh:
        return jsonify(CATALOG_CACHE)

    if SNAPSHOT_MODE:
        CATALOG_CACHE = snapshot_store.field_catalog()
        return jsonify(CATALOG_CACHE)

    # 1. Sample lab/exam fields (use IS NOT NULL only, to avoid SQL error 402 on text columns)
    lab_sql = """
        SELECT TOP 300000
            i.det_com_name,
            r.res_itm_name,
            i.det_dep_name,
            r.res_value,
            r.res_unit,
            r.res_exp
        FROM dbo.PES_apply_item i
        INNER JOIN dbo.PES_RESULT r
            ON i.app_no = r.app_no AND i.detail_id = r.detail_id
        WHERE r.res_value IS NOT NULL
    """
    df_lab = query_dataframe(lab_sql)

    lab_fields = []
    if not df_lab.empty:
        df_lab = df_lab[df_lab["res_value"] != ""]
        # Prefer det_com_name, then res_itm_name
        # After the change:
        df_lab["fieldName"] = df_lab.apply(
            lambda row: row["res_itm_name"]
            if row["res_itm_name"]
            else row["det_com_name"],
            axis=1,
        )
        df_lab = df_lab[df_lab["fieldName"] != ""]

        for field_name, grp in df_lab.groupby("fieldName", sort=False):
            first_row = grp.iloc[0]
            source_col = (
                "res_itm_name" if first_row["res_itm_name"] else "det_com_name(fallback)"
            )
            unique_vals = grp[["res_value", "res_unit", "res_exp"]].drop_duplicates().head(4)

            def is_num(val_str):
                try:
                    float(val_str)
                    return True
                except Exception:
                    return False

            numeric_hits = sum(1 for v in unique_vals["res_value"] if is_num(v))
            val_type = "number" if numeric_hits >= max(1, len(unique_vals) / 2) else "text"

            samples = [
                {
                    "rawValue": r["res_value"],
                    "display": f"{r['res_value']} {r['res_unit']}".strip(),
                    "refRange": r["res_exp"]
                }
                for _, r in unique_vals.iterrows()
            ]
            lab_fields.append({
                "fieldName": field_name,
                "sourceCol": source_col,
                "dept": first_row["det_dep_name"],
                "valueType": val_type,
                "unit": first_row["res_unit"],
                "refRange": first_row["res_exp"],
                "samples": samples
            })

    # 2. Sample descriptive exam report fields (det_com_name -> res_result)
    desc_sql = """
        SELECT TOP 15000
            i.det_com_name,
            i.det_dep_name,
            d.res_result
        FROM dbo.PES_apply_item i
        INNER JOIN dbo.PES_RESULT_DESC d
            ON i.app_no = d.app_no AND i.detail_id = d.detail_id
        WHERE d.res_result IS NOT NULL
    """
    df_desc = query_dataframe(desc_sql)

    desc_fields = []
    if not df_desc.empty:
        df_desc = df_desc[(df_desc["det_com_name"] != "") & (df_desc["res_result"] != "")]
        for field_name, grp in df_desc.groupby("det_com_name", sort=False):
            first_row = grp.iloc[0]
            unique_res = grp["res_result"].drop_duplicates().head(3)
            samples = [
                {
                    "rawValue": txt[:10].strip(),
                    "display": f'"{txt[:45]}..."' if len(txt) > 45 else f'"{txt}"'
                }
                for txt in unique_res
            ]
            desc_fields.append({
                "fieldName": field_name,
                "sourceCol": "det_com_name",
                "dept": first_row["det_dep_name"],
                "valueType": "text",
                "unit": "",
                "samples": samples
            })

    CATALOG_CACHE = {
        "lab_fields": lab_fields,
        "desc_fields": desc_fields
    }
    print(f"[OK] Secondary field dictionary loaded: {len(lab_fields)} lab/exam items, {len(desc_fields)} descriptive exam items")
    return jsonify(CATALOG_CACHE)


@app.route("/api/preload", methods=["GET"])
def preload_records():
    """Preload 5000 patients' basic exam info on page init, attaching lab and description summaries for the first 40"""
    limit = int(request.args.get("limit", 5000))
    if SNAPSHOT_MODE:
        return jsonify(snapshot_store.preload(limit))
    sql = f"""
        SELECT TOP ({limit})
            a.app_no, a.pat_id, a.app_pat_name, a.app_pat_age,
            p.pat_name, p.pat_sex, p.pat_nation, p.pat_address
        FROM dbo.PES_APPLY a
        INNER JOIN dbo.PES_PATIENT p ON a.pat_id = p.pat_id
        WHERE a.app_pat_name = p.pat_name
        ORDER BY a.app_no DESC
    """
    df = query_dataframe(sql)
    records = df.to_dict(orient="records") if not df.empty else []
    records = attach_items_to_records(records, max_app_nos=40)

    return jsonify({
        "total": len(records),
        "records": records
    })


@app.route("/api/patient_detail/<app_no>", methods=["GET"])
def get_patient_detail(app_no):
    """When [Full Record] is clicked, query that app_no's full lab and descriptive reports in real time"""
    if SNAPSHOT_MODE:
        return jsonify(snapshot_store.patient_detail(app_no))
    dummy = [{"app_no": app_no}]
    enriched = attach_items_to_records(dummy, max_app_nos=1)
    return jsonify(enriched[0])


def build_text_match_sql(vcol: str, operator: str, val_str: str, params: list) -> str:
    """
    For string matching, support both normal UTF-8 and Latin1-GBK byte streams, casting text to VARCHAR(MAX) first
    """
    val_raw = str(val_str).strip()
    val_moji = to_db_mojibake(val_raw)
    candidates = [val_raw] if val_raw == val_moji else [val_raw, val_moji]

    if operator == "contains":
        sub_clauses = []
        for c in candidates:
            params.append(f"%{c}%")
            sub_clauses.append(f"{vcol} LIKE %s")
        return "(" + " OR ".join(sub_clauses) + ")"

    if operator == "not_contains":
        sub_clauses = []
        for c in candidates:
            params.append(f"%{c}%")
            sub_clauses.append(f"{vcol} NOT LIKE %s")
        return "(" + " AND ".join(sub_clauses) + ")"

    if operator == "starts_with":
        sub_clauses = []
        for c in candidates:
            params.append(f"{c}%")
            sub_clauses.append(f"{vcol} LIKE %s")
        return "(" + " OR ".join(sub_clauses) + ")"

    if operator == "ends_with":
        sub_clauses = []
        for c in candidates:
            params.append(f"%{c}")
            sub_clauses.append(f"{vcol} LIKE %s")
        return "(" + " OR ".join(sub_clauses) + ")"

    if operator == "eq":
        sub_clauses = []
        for c in candidates:
            params.append(c)
            sub_clauses.append(f"LTRIM(RTRIM({vcol})) = %s")
        return "(" + " OR ".join(sub_clauses) + ")"

    if operator == "ne":
        sub_clauses = []
        for c in candidates:
            params.append(c)
            sub_clauses.append(f"LTRIM(RTRIM({vcol})) <> %s")
        return "(" + " AND ".join(sub_clauses) + ")"

    return ""


def build_value_clause(col_expr: str, operator: str, val, params: list) -> str:
    """Safely cast any column (including text) to VARCHAR(MAX) before building SQL conditions"""
    vcol = f"CAST({col_expr} AS VARCHAR(MAX))"

    if operator == "is_null":
        return f"({col_expr} IS NULL OR LTRIM(RTRIM({vcol})) = '')"
    if operator == "is_not_null":
        return f"({col_expr} IS NOT NULL AND LTRIM(RTRIM({vcol})) <> '')"

    if operator in ("gt", "ge", "lt", "le"):
        op_map = {"gt": ">", "ge": ">=", "lt": "<", "le": "<="}
        try:
            num_val = float(str(val).strip())
        except (TypeError, ValueError):
            raise ValueError(
                f"Numeric comparison operator {operator} requires a number, got {val!r}"
            )
        params.append(num_val)
        return f"TRY_CAST(LTRIM(RTRIM(CAST({col_expr} AS VARCHAR(8000)))) AS FLOAT) {op_map[operator]} %s"

    return build_text_match_sql(vcol, operator, str(val), params)


def build_where_from_conditions(conditions: list):
    """
    Build the UI's multi-condition search into WHERE clauses and parameters.
    Shared by /api/search and the selection-related app_no endpoints so both behave identically.
    """
    where_clauses = []
    params = []

    for idx, cond in enumerate(conditions):
        category = cond.get("category")
        sub_field = str(cond.get("subField", "")).strip()
        operator = cond.get("operator", "eq")
        val = cond.get("value", "")
        logic = "AND" if idx == 0 else ("OR" if cond.get("logic") == "OR" else "AND")

        # Same semantics as the UI's activeSearchConditions(): incomplete conditions are ignored, not used as filters.
        # If no secondary field is selected, or a non-null operator has no comparison value, continuing would produce
        # build_value_clause called float() on '' and raised ValueError (previously caused a 500).
        if category in ("LAB_EXAM", "DESC_EXAM") and not sub_field:
            continue
        if operator not in ("is_null", "is_not_null") and str(val).strip() == "":
            continue

        clause = ""
        # 1. Secondary lab/exam fields: prefer res_itm_name, fall back to det_com_name (split into index-friendly SARGable conditions)
        if category == "LAB_EXAM" and sub_field:
            params.extend([sub_field, sub_field])
            val_clause = build_value_clause("r.res_value", operator, val, params)

            clause = f"""
                EXISTS (
                    SELECT 1 FROM dbo.PES_apply_item i WITH (NOLOCK)
                    INNER JOIN dbo.PES_RESULT r WITH (NOLOCK)
                        ON i.app_no = r.app_no AND i.detail_id = r.detail_id
                    WHERE i.app_no = a.app_no
                      AND (
                          r.res_itm_name = %s
                          OR ((r.res_itm_name IS NULL OR r.res_itm_name = '') AND i.det_com_name = %s)
                      )
                      AND {val_clause}
                )
            """
        # 2. Secondary descriptive exam fields
        elif category == "DESC_EXAM" and sub_field:
            params.append(sub_field)
            val_clause = build_value_clause("d.res_result", operator, val, params)

            clause = f"""
                EXISTS (
                    SELECT 1 FROM dbo.PES_apply_item i WITH (NOLOCK)
                    INNER JOIN dbo.PES_RESULT_DESC d WITH (NOLOCK)
                        ON i.app_no = d.app_no AND i.detail_id = d.detail_id
                    WHERE i.app_no = a.app_no
                      AND i.det_com_name = %s
                      AND {val_clause}
                )
            """
        # 3. Patient basic information fields
        elif category in BASE_FIELDS:
            col_name, _ = BASE_FIELDS[category]
            clause = build_value_clause(col_name, operator, val, params)

        if clause:
            prefix = "" if not where_clauses else f" {logic} "
            where_clauses.append(f"{prefix}({clause})")

    return where_clauses, params


@app.route("/api/search", methods=["POST"])
def search_records():
    """Optimized multi-condition query: paging without COUNT, SARGable index search and millisecond timing"""
    t_start = time.perf_counter()
    payload = request.get_json(force=True)
    conditions = payload.get("conditions", [])
    page = max(1, int(payload.get("page", 1)))
    page_size = max(1, min(500, int(payload.get("page_size", 20))))

    # Skip the expensive COUNT(1) when paging if the UI supplies a known total
    client_known_total = payload.get("known_total")

    if SNAPSHOT_MODE:
        return jsonify(snapshot_store.search(conditions, page, page_size))

    where_clauses, params = build_where_from_conditions(conditions)

    # Drop the expensive cross-table name comparison a.app_pat_name = p.pat_name and rely on the pat_id key
    full_where_sql = f"WHERE ({''.join(where_clauses)})" if where_clauses else ""
    offset = (page - 1) * page_size

    t_db_start = time.perf_counter()

    # Run a full COUNT only on page 1 or when the client has no usable count
    if page == 1 or client_known_total is None:
        count_sql = f"""
            SELECT COUNT(1) AS cnt
            FROM dbo.PES_APPLY a WITH (NOLOCK)
            INNER JOIN dbo.PES_PATIENT p WITH (NOLOCK) ON a.pat_id = p.pat_id
            {full_where_sql}
        """
        total_df = query_dataframe(count_sql, params)
        total_count = int(total_df.iloc[0]["cnt"]) if not total_df.empty else 0
    else:
        total_count = int(client_known_total)

    # Paged data retrieval
    data_sql = f"""
        SELECT
            a.app_no, a.pat_id, a.app_pat_name, a.app_pat_age,
            p.pat_name, p.pat_sex, p.pat_nation, p.pat_address
        FROM dbo.PES_APPLY a WITH (NOLOCK)
        INNER JOIN dbo.PES_PATIENT p WITH (NOLOCK) ON a.pat_id = p.pat_id
        {full_where_sql}
        ORDER BY a.app_no DESC
        OFFSET {offset} ROWS FETCH NEXT {page_size} ROWS ONLY
    """
    df = query_dataframe(data_sql, params)
    t_db_query_end = time.perf_counter()

    records = df.to_dict(orient="records") if not df.empty else []
    # Attach details only for the rows shown on the current page
    records = attach_items_to_records(records, max_app_nos=page_size)

    t_end = time.perf_counter()

    return jsonify({
        "total": total_count,
        "page": page,
        "page_size": page_size,
        "records": records,
        "timing": {
            "db_query_ms": round((t_db_query_end - t_db_start) * 1000, 1),
            "total_server_ms": round((t_end - t_start) * 1000, 1)
        }
    })


def parse_app_nos_from_txt_files(watch_dir: str) -> dict:
    """
    Scan all .txt files in the given directory, extract app_no and return {app_no: source txt filename}
    Support newline/comma/space/tab separators and strip the .0 suffix
    """
    os.makedirs(watch_dir, exist_ok=True)
    txt_files = sorted(glob.glob(os.path.join(watch_dir, "*.txt")))
    app_no_source_map = {}

    for file_path in txt_files:
        file_name = os.path.basename(file_path)
        if file_name.startswith("."):
            continue
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            # Split order numbers by whitespace, comma, semicolon
            tokens = re.split(r"[\s,;，；]+", content)
            for token in tokens:
                clean_no = token.strip()
                if clean_no.endswith(".0"):
                    clean_no = clean_no[:-2]
                if clean_no and clean_no not in app_no_source_map:
                    app_no_source_map[clean_no] = file_name
        except Exception as e:
            print(f"[WARN] Failed to read file {file_name}: {e}")

    return app_no_source_map, len(
        [f for f in txt_files if not os.path.basename(f).startswith(".")]
    )


def sync_watched_directory() -> dict:
    """
    Incremental sync core logic:
    1. Scan app_no in all txt files under WATCH_DIR
    2. Filter out app_no that have not been queried
    3. Query only new app_no basic info and append it to synced_base_info.csv at a fixed path
    """
    t0 = time.perf_counter()
    os.makedirs(WATCH_DIR, exist_ok=True)

    if SNAPSHOT_MODE:
        df = snapshot_store.watch_base_df()
        return {
            "df": df,
            "watch_dir": WATCH_DIR,
            "txt_file_count": 1,
            "total_txt_app_nos": len(df),
            "newly_synced_count": 0,
            "scan_ms": round((time.perf_counter() - t0) * 1000, 1),
            "last_scan_time": datetime.now().strftime("%H:%M:%S"),
        }

    app_no_map, txt_count = parse_app_nos_from_txt_files(WATCH_DIR)
    all_txt_app_nos = list(app_no_map.keys())

    # Read the existing local basic-information CSV
    base_cols = [
        "app_no",
        "pat_id",
        "pat_name",
        "app_pat_name",
        "app_pat_age",
        "pat_sex",
        "pat_nation",
        "pat_address",
        "source_txt",
        "sync_time",
    ]
    if os.path.exists(LOCAL_BASE_CSV):
        try:
            df_local = pd.read_csv(LOCAL_BASE_CSV, dtype=str).fillna("")
        except Exception:
            df_local = pd.DataFrame(columns=base_cols)
    else:
        df_local = pd.DataFrame(columns=base_cols)

    existing_app_nos = (
        set(df_local["app_no"].astype(str).str.strip())
        if not df_local.empty
        else set()
    )

    # Read validated app_no (avoid re-querying invalid numbers on every poll)
    processed_app_nos = set(existing_app_nos)
    if os.path.exists(PROCESSED_IDS_FILE):
        with open(PROCESSED_IDS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    processed_app_nos.add(line.strip())

    # Filter for new app_no that actually require a database lookup
    new_app_nos = [no for no in all_txt_app_nos if no not in processed_app_nos]
    newly_added_count = 0

    if new_app_nos:
        fetched_dfs = []
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Query in batches of 500 to avoid too many SQL IN parameters
        batch_size = 500
        for i in range(0, len(new_app_nos), batch_size):
            batch = new_app_nos[i : i + batch_size]
            placeholders = ",".join(["%s"] * len(batch))
            sql = f"""
                SELECT
                    a.app_no, a.pat_id, a.app_pat_name, a.app_pat_age,
                    p.pat_name, p.pat_sex, p.pat_nation, p.pat_address
                FROM dbo.PES_APPLY a WITH (NOLOCK)
                INNER JOIN dbo.PES_PATIENT p WITH (NOLOCK) ON a.pat_id = p.pat_id
                WHERE a.app_no IN ({placeholders})
            """
            df_batch = query_dataframe(sql, batch)
            if not df_batch.empty:
                df_batch["app_no"] = df_batch["app_no"].astype(str).str.strip()
                # Deduplicate to one basic record per app_no
                df_batch = df_batch.drop_duplicates(
                    subset=["app_no"], keep="first"
                )
                df_batch["source_txt"] = df_batch["app_no"].map(
                    lambda x: app_no_map.get(x, "")
                )
                df_batch["sync_time"] = now_str
                fetched_dfs.append(df_batch[base_cols])

        if fetched_dfs:
            df_new = pd.concat(fetched_dfs, ignore_index=True)
            newly_added_count = len(df_new)
            # Newly synced data goes first and is persisted to a CSV at a fixed path
            df_local = pd.concat([df_new, df_local], ignore_index=True)
            df_local = df_local.drop_duplicates(subset=["app_no"], keep="first")
            df_local.to_csv(LOCAL_BASE_CSV, index=False, encoding="utf-8-sig")

        # Record the app_no processed in this batch
        with open(PROCESSED_IDS_FILE, "a", encoding="utf-8") as f:
            for no in new_app_nos:
                f.write(f"{no}\n")

    scan_ms = round((time.perf_counter() - t0) * 1000, 1)
    return {
        "df": df_local,
        "watch_dir": WATCH_DIR,
        "txt_file_count": txt_count,
        "total_txt_app_nos": len(all_txt_app_nos),
        "newly_synced_count": newly_added_count,
        "scan_ms": scan_ms,
        "last_scan_time": datetime.now().strftime("%H:%M:%S"),
    }


@app.route("/api/watch_sync", methods=["GET"])
def api_watch_sync():
    """
    Module 2 main interface: auto-scan txt files in the watch path, sync incremental basic info and return paged results
    """
    page = max(1, int(request.args.get("page", 1)))
    page_size = max(1, min(500, int(request.args.get("page_size", 20))))
    keyword = str(request.args.get("keyword", "")).strip().lower()

    sync_res = sync_watched_directory()
    df = sync_res["df"]

    # Support quick filtering of stored local basic info by app_no or name
    if not df.empty and keyword:
        mask = (
            df["app_no"].astype(str).str.lower().str.contains(keyword, na=False)
            | df["pat_name"]
            .astype(str)
            .str.lower()
            .str.contains(keyword, na=False)
            | df["source_txt"]
            .astype(str)
            .str.lower()
            .str.contains(keyword, na=False)
        )
        df = df[mask]

    total_count = len(df)
    start = (page - 1) * page_size
    df_page = (
        df.iloc[start : start + page_size] if not df.empty else pd.DataFrame()
    )

    return jsonify({
        "total": total_count,
        "page": page,
        "page_size": page_size,
        "records": df_page.to_dict(orient="records")
        if not df_page.empty
        else [],
        "meta": {
            "watch_dir": sync_res["watch_dir"],
            "local_csv": LOCAL_BASE_CSV,
            "txt_file_count": sync_res["txt_file_count"],
            "total_txt_app_nos": sync_res["total_txt_app_nos"],
            "newly_synced_count": sync_res["newly_synced_count"],
            "scan_ms": sync_res["scan_ms"],
            "last_scan_time": sync_res["last_scan_time"],
        },
    })


@app.route("/api/watch_demo_drop", methods=["POST"])
def api_watch_demo_drop():
    """
    Test helper: randomly pick 15 real app_no from the database, write a new .txt into the watch directory,
    lets you verify in one click that "new txt file dropped -> auto-scan and sync basic info" works end to end
    """
    os.makedirs(WATCH_DIR, exist_ok=True)
    if SNAPSHOT_MODE:
        import random as _random
        pool = snapshot_store.load()["base"]["app_no"].tolist()
        app_nos = [str(x) for x in _random.sample(pool, min(15, len(pool)))]
    else:
        sql = """
            SELECT TOP 15 a.app_no
            FROM dbo.PES_APPLY a WITH (NOLOCK)
            INNER JOIN dbo.PES_PATIENT p WITH (NOLOCK) ON a.pat_id = p.pat_id
            ORDER BY NEWID()
        """
        df = query_dataframe(sql)
        app_nos = (
            [str(x).strip() for x in df["app_no"].tolist()]
            if not df.empty
            else [f"202600{i}" for i in range(10, 25)]
        )

    file_name = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    file_path = os.path.join(WATCH_DIR, file_name)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(app_nos))

    return jsonify(
        {"file_name": file_name, "count": len(app_nos), "app_nos": app_nos}
    )


# ===================== Selection and export pipeline API =====================
# Used by the UI for "select app_no -> deselect/export":
#   /api/search_app_nos        Collect all app_no matching the current search (select all)
#   /api/search_page_app_nos   Collect app_no by page range (select range)
#   /api/watch_all_app_nos     All app_no in the watch directory (select all)
#   /api/watch_page_app_nos    Watch-directory app_no by page range (select range)
#   /api/pipeline/export       Export selected app_no and start the background pipeline
#   /api/pipeline/status       Query pipeline task progress and results

MAX_EXPORT_APP_NOS = 200      # per-export cap (kept in sync with pipeline_runner)
MAX_SELECT_PAGES = 200        # per-request page-range selection cap

# Shared FROM clause for app_no collection endpoints; must match /api/search:
# Clauses generated by build_where_from_conditions reference both the a. and p. aliases,
# Therefore PES_APPLY must join PES_PATIENT, otherwise p.* columns cannot be resolved.
_SEARCH_FROM_SQL = (
    "FROM dbo.PES_APPLY a WITH (NOLOCK) "
    "INNER JOIN dbo.PES_PATIENT p WITH (NOLOCK) ON a.pat_id = p.pat_id "
)


def _where_sql_from_conditions(conditions: list):
    """Uses the same WHERE assembly as /api/search and returns (where_sql, params)."""
    where_clauses, params = build_where_from_conditions(conditions)
    # Note: clauses already carry " AND "/" OR " prefixes; concatenate with an empty string, then wrap in parentheses,
    # Cannot join(" AND ") again, otherwise "(c1) AND  AND (c2)" causes a syntax error.
    where_sql = f"({''.join(where_clauses)})" if where_clauses else "1=1"
    return where_sql, params


def _load_watch_df(keyword: str):
    """Sync the watch directory and filter by keyword; returns (df, sync_res)."""
    sync_res = sync_watched_directory()
    df = sync_res["df"]
    if not df.empty and keyword:
        mask = (
            df["app_no"].astype(str).str.lower().str.contains(keyword, na=False)
            | df["pat_name"].astype(str).str.lower().str.contains(keyword, na=False)
            | df["pat_id"].astype(str).str.lower().str.contains(keyword, na=False)
        )
        df = df[mask]
    return df, sync_res


@app.route("/api/search_app_nos", methods=["POST"])
def api_search_app_nos():
    """Return all app_no matching the current search (for "select all")."""
    payload = request.get_json() or {}
    conditions = payload.get("conditions", [])
    limit = max(1, min(50000, int(payload.get("limit", 20000))))

    if SNAPSHOT_MODE:
        try:
            return jsonify(snapshot_store.search_app_nos(conditions, limit))
        except ValueError as e:
            return jsonify({"error": str(e)}), 400

    try:
        where_sql, params = _where_sql_from_conditions(conditions)
    except ValueError as e:
        # The search condition itself is invalid (e.g. a numeric operator received a non-number); this is a client input problem
        return jsonify({"error": str(e)}), 400

    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(f"SELECT COUNT(*) {_SEARCH_FROM_SQL}WHERE {where_sql}", tuple(params))
        total = cursor.fetchone()[0]
        cursor.execute(
            f"SELECT TOP {limit} a.[app_no] {_SEARCH_FROM_SQL}WHERE {where_sql} ORDER BY a.app_no DESC",
            tuple(params),
        )
        app_nos = [str(r[0]) for r in cursor.fetchall()]
    except Exception as e:
        return jsonify({"error": f"Failed to collect app_no: {e}"}), 500
    finally:
        conn.close()

    return jsonify({
        "app_nos": app_nos,
        "total": total,
        "truncated": total > len(app_nos),
    })


@app.route("/api/search_page_app_nos", methods=["POST"])
def api_search_page_app_nos():
    """Return app_no within the given page range (for "select page range")."""
    payload = request.get_json() or {}
    conditions = payload.get("conditions", [])
    from_page = max(1, int(payload.get("from_page", 1)))
    to_page = max(from_page, int(payload.get("to_page", from_page)))
    page_size = max(1, min(500, int(payload.get("page_size", 20))))

    if to_page - from_page + 1 > MAX_SELECT_PAGES:
        return jsonify({"error": f"At most {MAX_SELECT_PAGES} pages can be selected per batch; please narrow the page range"}), 400

    if SNAPSHOT_MODE:
        try:
            return jsonify(snapshot_store.search_page_app_nos(
                conditions, from_page, to_page, page_size))
        except ValueError as e:
            return jsonify({"error": str(e)}), 400

    try:
        where_sql, params = _where_sql_from_conditions(conditions)
    except ValueError as e:
        # The search condition itself is invalid (e.g. a numeric operator received a non-number); this is a client input problem
        return jsonify({"error": str(e)}), 400

    offset = (from_page - 1) * page_size
    fetch = (to_page - from_page + 1) * page_size

    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            f"SELECT a.[app_no] {_SEARCH_FROM_SQL}WHERE {where_sql} "
            f"ORDER BY a.app_no DESC OFFSET %s ROWS FETCH NEXT %s ROWS ONLY",
            tuple(params) + (offset, fetch),
        )
        app_nos = [str(r[0]) for r in cursor.fetchall()]
    except Exception as e:
        return jsonify({"error": f"Failed to collect app_no: {e}"}), 500
    finally:
        conn.close()

    return jsonify({
        "app_nos": app_nos,
        "from_page": from_page,
        "to_page": to_page,
        "page_size": page_size,
    })


@app.route("/api/watch_all_app_nos", methods=["GET"])
def api_watch_all_app_nos():
    """Return all app_no in the watch directory (for "select all")."""
    keyword = str(request.args.get("keyword", "")).strip().lower()
    limit = max(1, min(50000, int(request.args.get("limit", 20000))))

    df, _ = _load_watch_df(keyword)
    all_nos = [str(x) for x in df["app_no"].tolist()]
    app_nos = all_nos[:limit]
    return jsonify({
        "app_nos": app_nos,
        "total": len(all_nos),
        "truncated": len(all_nos) > len(app_nos),
    })


@app.route("/api/watch_page_app_nos", methods=["GET"])
def api_watch_page_app_nos():
    """Return watch-directory app_no within a page range (for "select page range")."""
    keyword = str(request.args.get("keyword", "")).strip().lower()
    from_page = max(1, int(request.args.get("from_page", 1)))
    to_page = max(from_page, int(request.args.get("to_page", from_page)))
    page_size = max(1, min(500, int(request.args.get("page_size", 20))))

    if to_page - from_page + 1 > MAX_SELECT_PAGES:
        return jsonify({"error": f"At most {MAX_SELECT_PAGES} pages can be selected per batch; please narrow the page range"}), 400

    df, _ = _load_watch_df(keyword)
    all_nos = [str(x) for x in df["app_no"].tolist()]
    start = (from_page - 1) * page_size
    end = min(to_page * page_size, len(all_nos))
    app_nos = all_nos[start:end] if start < len(all_nos) else []
    return jsonify({
        "app_nos": app_nos,
        "from_page": from_page,
        "to_page": to_page,
        "page_size": page_size,
        "total": len(all_nos),
    })


@app.route("/api/pipeline/export", methods=["POST"])
def api_pipeline_export():
    """
    Accept app_no selected in the UI and start the background pipeline:
    cli.py (BatchProcessor) risk scoring -> disease-level tier assembly (stacking values deprecated) ->
    subgraph_construct + hybrid_retrieval_engine subgraph retrieval -> persisted to debug_output.
    """
    payload = request.get_json() or {}
    app_nos = payload.get("app_nos") or []
    source = payload.get("source", "search")
    # Whether to enable JEV-27B decision-model scoring (UI checkbox; off by default, not run in the normal flow)
    enable_decide = bool(payload.get("enable_decide", False))
    # Whether to generate the visual report (UI checkbox; on by default)
    enable_report = bool(payload.get("enable_report", True))

    seen = set()
    unique = []
    for a in app_nos:
        a = str(a).strip()
        if a and a not in seen:
            seen.add(a)
            unique.append(a)

    if not unique:
        return jsonify({"error": "No app_no selected"}), 400
    if len(unique) > MAX_EXPORT_APP_NOS:
        return jsonify({
            "error": f"At most {MAX_EXPORT_APP_NOS} app_no can be exported per batch; currently {len(unique)} selected"
        }), 400

    import pipeline_runner
    task_id = pipeline_runner.start_pipeline(
        unique, source, enable_decide=enable_decide, enable_report=enable_report
    )
    return jsonify({
        "task_id": task_id,
        "count": len(unique),
        "enable_decide": enable_decide,
        "enable_report": enable_report,
    })


@app.route("/api/pipeline/status/<task_id>", methods=["GET"])
def api_pipeline_status(task_id):
    """Query pipeline task progress and results."""
    import pipeline_runner
    task = pipeline_runner.get_task(task_id)
    if not task:
        return jsonify({"error": "Task not found or expired"}), 404
    return jsonify(task)


@app.route("/api/pipeline/tasks", methods=["GET"])
def api_pipeline_tasks():
    """List of recent pipeline tasks."""
    import pipeline_runner
    return jsonify({"tasks": pipeline_runner.list_tasks()})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)