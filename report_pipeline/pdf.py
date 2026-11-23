"""HTML -> PDF (WeasyPrint, vector text, no browser dependency)."""
from __future__ import annotations

import os


def html_to_pdf(html, out_path):
    from weasyprint import HTML

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    HTML(string=html, base_url=os.getcwd()).write_pdf(out_path)
    return out_path
