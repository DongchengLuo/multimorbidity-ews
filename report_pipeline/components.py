"""HTML components: icons, dashboard, disease cards, generic block rendering."""
from __future__ import annotations

import base64
import json
import os
import re

from . import gauge
from .mdparse import inline

# The icon library ships with the project under ui_icons/; override with the ICON_ROOT env var.
ICON_ROOT = os.getenv(
    "ICON_ROOT",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ui_icons"),
)
_INDEX = None
_CACHE = {}

CHIP = {3: "c-high", 2: "c-mid", 1: "c-low", 0: "c-none"}
DZ = {3: "dz-high", 2: "dz-mid", 1: "dz-low", 0: "dz-none"}


def _index():
    global _INDEX
    if _INDEX is None:
        p = os.path.join(ICON_ROOT, "icon_index.json")
        _INDEX = json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}
    return _INDEX


def _b64(path):
    if path in _CACHE:
        return _CACHE[path]
    full = os.path.join(ICON_ROOT, path)
    if not os.path.exists(full):
        _CACHE[path] = ""
        return ""
    ext = os.path.splitext(full)[1].lower()
    mime = "image/svg+xml" if ext == ".svg" else "image/png"
    data = base64.b64encode(open(full, "rb").read()).decode()
    _CACHE[path] = f"data:{mime};base64,{data}"
    return _CACHE[path]


def icon(name, tone="orange", cls="ic"):
    """Get an asset-library icon by name (1 exam / 3 non-exam)."""
    if not name:
        return ""
    cands = [name]
    m = re.search(r"[（(]([A-Za-z0-9\-]+)[)）]", name)
    if m:
        cands.append(m.group(1))
    for cand in cands:
        for key, val in _index().items():
            if cand in key:
                path = val.get(tone) or val.get("orange") or next(iter(val.values()))
                src = _b64(path)
                if src:
                    return f'<img class="{cls}" src="{src}" alt="">'
    return ""


def ui_icon(name, cls="uic"):
    """Get the SVG line-art icon under report_ui."""
    for sub in ("report_ui/", ""):
        for ext in (".svg", ".png"):
            src = _b64(f"{sub}{name}{ext}")
            if src:
                return f'<img class="{cls}" src="{src}" alt="">'
    return ""


def png_icon(rel, size=34, cls="pic"):
    """Get a PNG icon by asset-library relative path (1 exam / 3 non-exam)."""
    src = _b64(rel)
    if not src:
        return ""
    return f'<img class="{cls}" width="{size}" height="{size}" src="{src}" alt="">'


# ---------------------------------------------------------------- Dashboard
def _score(v):
    try:
        return float(str(v).replace("%", ""))
    except (TypeError, ValueError):
        return 0.0


def gauge_block(label, bands, source, score, level, rank, compact=False, cohort_n=0, unit="",
                chip=True):
    neutral = source == "cohort" and rank == 0
    value_text = None
    if source == "level":
        score = max(0.0, rank - 0.5)
        value_text = level or ""
    chip_html = f'<span class="chip {CHIP[rank]}">{level or "Not stratified"}</span>' if chip else ""
    head = f'<div class="gcard-head"><span class="gcard-name">{inline(label)}</span>{chip_html}</div>'
    if not bands:
        return (
            f'<div class="gcard na">{head}'
            '<div class="gcard-na">Insufficient sample size; tier ticks not drawn</div></div>'
        )
    parsed = {"bands": bands, "unit": unit}
    width = 132 if compact else 230
    svg = gauge.gauge_svg(
        score, parsed, width=width, show_ticks=not compact,
        value_font=15 if compact else 19, value_text=value_text, neutral=neutral,
    )
    note = ""
    if source == "cohort":
        note = f'<div class="gcard-note">Thresholds from same-batch population percentiles (n={cohort_n})</div>'
    return f'<div class="gcard">{head}{svg}{note}</div>'


def legend_panel(bands, source, score, cohort_n=0, unit="", rank=0):
    if not bands:
        return ""
    parsed = {"bands": bands, "unit": unit}
    idx0, _ = gauge.locate(score, bands)
    rows = "".join(
        f'<div class="gl-row{" on" if i == idx0 else ""}">'
        f'<i style="background:{c}"></i>{lab}</div>'
        for i, (c, lab) in enumerate(gauge.band_rows(parsed, neutral=(source == "cohort" and rank == 0)))
    )
    idx, _ = gauge.locate(score, bands)
    src = {
        "cohort": f"Threshold source: same-batch population percentiles (n={cohort_n})",
        "level": "Threshold source: model risk tiers",
    }.get(source, "Threshold source: model original tiers")
    return (
        '<div class="glegend">'
        f'<div class="glegend-h">Risk bands</div>{rows}'
        f'<div class="glegend-note">{src} · pointer in band {idx + 1}</div>'
        "</div>"
    )


# ---------------------------------------------------------------- Disease cards
def visible_models(disease):
    """Show medium/high only: hide low-risk models when medium or high exists. Returns (kept, hidden)."""
    models = sorted(disease["models"], key=lambda m: -m["rank"])
    if any(m["rank"] >= 2 for m in models):
        kept = [m for m in models if m["rank"] >= 2]
    else:
        kept = models
    return kept, len(models) - len(kept)


def disease_card(disease, cohort, icon_html="", compact=False):
    from .cohort import resolve_bands

    kept, hidden = visible_models(disease)
    rank = max([m["rank"] for m in kept], default=0)
    chip = f'<span class="chip {CHIP[rank]}">{disease["level"]}</span>'
    n_tier = sum(1 for m in kept if m["rank"] > 0)
    meta = f'{n_tier}/{len(kept)} models with tier thresholds'
    if any(m.get("derived") for m in kept):
        meta += " · includes population-percentile estimates"
    if hidden:
        meta += f' · {hidden} low-risk model(s) hidden'
    cards = []
    for m in kept:
        bands, source, _ = resolve_bands(m["model"], cohort, _score(m["score"]), m["rank"])
        n = len(cohort.dist.get(m["model"], {}).get("scores", []))
        cards.append(
            gauge_block(m["label"], bands, source, _score(m["score"]),
                        m["level"], m["rank"], compact=compact, cohort_n=n,
                        unit=m.get("unit", ""))
        )
    legend = ""
    if len(kept) == 1:
        bands, source, _ = resolve_bands(kept[0]["model"], cohort, _score(kept[0]["score"]), kept[0]["rank"])
        n = len(cohort.dist.get(kept[0]["model"], {}).get("scores", []))
        legend = legend_panel(bands, source, _score(kept[0]["score"]), n,
                              kept[0].get("unit", ""), kept[0]["rank"])
    return (
        f'<div class="dz {DZ[rank]}">'
        f'<div class="dz-head">{icon_html}<span class="dz-name">{inline(disease["name"])}</span>'
        f'{chip}<span class="dz-vote">{meta}</span></div>'
        f'<div class="dz-body{" solo" if not legend else ""}">'
        f'<div class="dz-gauges">{"".join(cards)}</div>{legend}</div>'
        "</div>"
    )


# ---------------------------------------------------------------- Common blocks
# Paragraph-level callout detection: only paragraphs explicitly marked "note / priority note" in the md; no medical wording is altered.
CALLOUT_PATTERNS = [
    (r"^\*\*(Note|Notes|Remark)\*\*", "info"),
    (r"^(Note|Notes)[:]", "info"),
    (r"^\*\*(What these negative[^*]*|Significance of these negative[^*]*)\*\*", "ok"),
    (r"^\*\*(Important note|Please note)[^*]*\*\*", ""),
]


def callout_cls(text):
    """Return the callout modifier class; None when not a callout."""
    for pat, cls in CALLOUT_PATTERNS:
        if re.match(pat, text, re.IGNORECASE):
            return cls
    return None


def render_blocks(blist, cls=""):
    """Fallback renderer: any md block without a dedicated handler still renders completely."""
    out = []
    for kind, val in blist:
        if kind == "p":
            tone = callout_cls(val)
            if tone is not None:
                extra = f" {tone}" if tone else ""
                out.append(f'<div class="callout{extra}">{inline(val)}</div>')
            else:
                out.append(f'<p class="p">{inline(val)}</p>')
        elif kind == "quote":
            tone = callout_cls(val)
            extra = f" {tone}" if tone else ""
            out.append(f'<div class="callout{extra}">{inline(val)}</div>')
        elif kind == "ul":
            out.append('<ul class="ul">' + "".join(f"<li>{inline(x)}</li>" for x in val) + "</ul>")
        elif kind == "ol":
            out.append('<ol class="ol">' + "".join(f"<li>{inline(x)}</li>" for _, x in val) + "</ol>")
        elif kind == "table" and val:
            head = "".join(f"<th>{inline(c)}</th>" for c in val[0])
            body = "".join(
                "<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>" for r in val[1:]
            )
            out.append(f'<table class="tbl"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>')
    return f'<div class="{cls}">{"".join(out)}</div>'
