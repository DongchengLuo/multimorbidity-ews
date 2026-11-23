"""HTML assembly."""
from __future__ import annotations

from . import pages
from .mdparse import Doc
from .style import CSS


def build_html(patient, cohort, md_text):
    doc = Doc(md_text)
    body = [
        pages.cover(patient, doc.preamble),
        '<section class="dash">',
        pages.dashboard(patient, cohort),
        "</section>",
        '<section class="body">',
        pages.render_doc(doc, patient, cohort),
        "</section>",
    ]
    return (
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<title>Personal Multimorbidity Risk Interpretation and Health Management Report</title>"
        f"<style>{CSS}</style></head><body>{''.join(body)}</body></html>"
    )
