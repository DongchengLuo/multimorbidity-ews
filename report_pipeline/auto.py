"""Automatic report rendering: md -> single-file HTML -> vector PDF.

Called by ``pipeline_runner`` at the end of the report-generation stage to turn ``report_generator``
The produced md renders to a single-file HTML / PDF identical to ``python -m report_pipeline``.

Design notes
--------
* Rendering makes no LLM calls; the same input always yields the same output (same code path and layout as the CLI).
* ``Cohort`` is read-only, built once and reused globally; rendering locks per patient and is idempotent/reentrant.
* A single patient's render failure does not affect the main pipeline; the error is written into that patient's ``report_files``.
* Switches: ``REPORT_RENDER_HTML`` (default 1), ``REPORT_RENDER_PDF`` (default 1).
  PDF depends on ``weasyprint``; when missing, degrade to HTML only and record the reason.
"""
from __future__ import annotations

import os
import threading
import time
import traceback


def _env_flag(name, default=True):
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).strip().lower() not in ("0", "false", "no", "off", "")


RENDER_HTML = _env_flag("REPORT_RENDER_HTML", True)
RENDER_PDF = _env_flag("REPORT_RENDER_PDF", True)


def _pdf_available():
    try:
        import weasyprint  # noqa: F401
        return True
    except Exception:
        return False


class ReportRenderer:
    """Batch-render a batch's md reports into HTML / PDF.

    Usage::

        r = ReportRenderer(batch_dir)
        r.render_records(records, progress_cb=cb)
    """

    def __init__(self, batch_dir, do_html=None, do_pdf=None):
        self.batch_dir = batch_dir
        self.do_html = RENDER_HTML if do_html is None else bool(do_html)
        self.do_pdf = RENDER_PDF if do_pdf is None else bool(do_pdf)
        self._cohort = None
        self._cohort_lock = threading.Lock()
        self._done = set()
        self._lock = threading.Lock()
        self._pdf_ok = None

    # ---------- Internal ----------
    def _get_cohort(self):
        if self._cohort is None:
            with self._cohort_lock:
                if self._cohort is None:
                    from .cohort import Cohort

                    self._cohort = Cohort(self.batch_dir)
        return self._cohort

    def _pdf_ready(self):
        if self._pdf_ok is None:
            self._pdf_ok = _pdf_available()
            if not self._pdf_ok:
                print("[report_pipeline.auto] weasyprint not installed; HTML only this run")
        return self._pdf_ok

    # ---------- Single patient ----------
    def render_record(self, rec):
        """Render one patient record, update ``rec['report_files']`` in place and return that dict.

        Idempotent: repeated calls for the same app_no return the previous result.
        """
        app_no = str(rec.get("eid") or rec.get("app_no") or "")
        files = dict(rec.get("report_files") or {})

        if not self.do_html:
            return files

        with self._lock:
            if app_no in self._done:
                return files
            self._done.add(app_no)

        md_path = files.get("markdown")
        if not md_path or not os.path.exists(md_path):
            files["render_error"] = "no md report; skipping render"
            rec["report_files"] = files
            return files

        try:
            t0 = time.time()
            from .check import check as check_report
            from .render import build_html

            md_text = open(md_path, encoding="utf-8").read()
            cohort = self._get_cohort()
            patient = cohort.patient(app_no)
            html = build_html(patient, cohort, md_text)

            out_dir = os.path.dirname(md_path)
            html_path = os.path.join(out_dir, f"report_{app_no}.html")
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(html)
            files["html"] = html_path

            try:
                files["check_passed"] = bool(check_report(md_text, html, verbose=False))
            except Exception as e:
                files["check_passed"] = None
                files["check_error"] = str(e)

            if self.do_pdf:
                if self._pdf_ready():
                    try:
                        from .pdf import html_to_pdf

                        pdf_path = os.path.join(out_dir, f"report_{app_no}.pdf")
                        html_to_pdf(html, pdf_path)
                        files["pdf"] = pdf_path
                    except Exception as e:
                        files["pdf_error"] = str(e)
                        print(f"[report_pipeline.auto] {app_no} PDF failed: {e}")
                else:
                    files["pdf_error"] = "weasyprint not installed"

            files["render_seconds"] = round(time.time() - t0, 2)
            print(
                f"[report_pipeline.auto] {app_no} render complete "
                f"html {len(html)}B"
                + (f" / pdf {os.path.getsize(files['pdf'])}B" if files.get("pdf") else "")
                + f" / check {'passed' if files.get('check_passed') else 'failed'}"
                + f" / {files['render_seconds']}s"
            )
        except Exception as e:
            files["render_error"] = str(e)
            print(f"[report_pipeline.auto] {app_no} render failed: {e}")
            print(traceback.format_exc())

        rec["report_files"] = files
        return files

    # ---------- Batch ----------
    def render_records(self, records, progress_cb=None):
        """Render all patients whose report succeeded; returns ``{app_no: files}``."""
        targets = [
            r for r in (records or [])
            if r.get("report_status") == "SUCCESS" and r.get("report_files")
        ]
        total = len(targets)
        out = {}
        for i, rec in enumerate(targets, 1):
            app_no = str(rec.get("eid") or rec.get("app_no") or "")
            out[app_no] = self.render_record(rec)
            if progress_cb:
                try:
                    progress_cb(i, total, app_no, out[app_no])
                except Exception:
                    pass
        return out


def render_batch(batch_dir, records, progress_cb=None, do_html=None, do_pdf=None):
    """Convenience function: render all reports of a batch in one call."""
    return ReportRenderer(batch_dir, do_html=do_html, do_pdf=do_pdf).render_records(
        records, progress_cb=progress_cb
    )
