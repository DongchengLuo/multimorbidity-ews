# -*- coding: utf-8 -*-
"""180-degree half-ring gauge: threshold parsing + geometry + SVG generation.

Design notes
--------
1. Arc length is split evenly by risk band, not linearly by value.
   Rationale: these models' thresholds are already banded (0-5%|5-10%|10-100%),
   Bands are the semantic unit; linear splitting would squeeze 2.93% into the leftmost 3%,
   readers cannot tell. Equal bands + linear within band preserve band semantics
   while preserving numeric precision within the band.
2. Pointer position = (band index + relative position within band) / total bands.
3. Label each band boundary with the real threshold so readers can verify the pointer position.
"""
import math
import re

# ---------------------------------------------------------------- Threshold parsing

_NUM = r"-?\d+(?:\.\d+)?"


def _to_float(s):
    try:
        return float(str(s).replace("%", "").strip())
    except (TypeError, ValueError):
        return None


def parse_bands(ref):
    """Parse gauge_reference into [{lo, hi, label}].

    Supported forms (all seen in this project):
        0-0.22|0.22-1
        0%-5%|5%-10%|10%-100%
        -5.0 - -0.11 | -0.11 - 0.58 | 0.58 - 5.0
        -10-23|23-60
        <8|≥8
        0-2|3-4|5-6|7-8|9+
        0-300.6|300.6 and above (nomogram total)
        Q1:<6.71|Q2:6.71-7.08|Q3:7.09-7.67|Q4:>7.67
        -1|0|1|2|3-4|5
    """
    if not ref:
        return None
    ref = str(ref).strip()
    if ref in ("-", "", "nan", "None"):
        return None

    unit = "%" if "%" in ref else ""
    parts = [p.strip() for p in ref.split("|") if p.strip()]
    if not parts:
        return None

    bands = []
    for p in parts:
        raw = p
        # Strip prefixes such as Q1:/Q2:
        p = re.sub(r"^Q\d+\s*[:：]\s*", "", p)
        # Strip parenthetical notes such as "300.6 and above (nomogram total)"
        p = re.sub(r"[（(].*?[)）]", "", p).strip()
        p = p.replace("～", "~").replace("—", "-").replace("–", "-").replace("－", "-")
        p = p.replace("＜", "<").replace("＞", ">").replace("≤", "<=").replace("≥", ">=")
        p = p.replace("%", "").replace("％", "").strip()

        lo = hi = None
        m = re.match(rf"^({_NUM})\s*[-~]\s*({_NUM})$", p)
        if m:
            lo, hi = float(m.group(1)), float(m.group(2))
        else:
            m = re.match(rf"^<\s*=?\s*({_NUM})$", p)
            if m:
                hi = float(m.group(1))
            else:
                m = re.match(rf"^>\s*=?\s*({_NUM})$", p)
                if m:
                    lo = float(m.group(1))
                else:
                    m = re.match(rf"^({_NUM})\s*\+$", p)
                    if m:
                        lo = float(m.group(1))
                    else:
                        m = re.match(rf"^({_NUM})\s*(?:and above|or above|\+)$", p, re.I)
                        if m:
                            lo = float(m.group(1))
                        else:
                            m = re.match(rf"^({_NUM})$", p)
                            if m:
                                lo = hi = float(m.group(1))
        if lo is None and hi is None:
            return None  # unparseable; let the caller degrade
        bands.append({"lo": lo, "hi": hi, "raw": raw})

    if not bands:
        return None

    # Direction: open upper bound at the end (>x / x+) -> ascending; open lower bound (<x) -> descending
    descending = False
    if len(bands) >= 2:
        last, first = bands[-1], bands[0]
        if last["hi"] is None and last["lo"] is not None:
            descending = False
        elif last["lo"] is None and last["hi"] is not None:
            descending = True
        else:
            a = first["lo"] if first["lo"] is not None else first["hi"]
            b = last["lo"] if last["lo"] is not None else last["hi"]
            descending = a is not None and b is not None and a > b
    if descending:
        bands = list(reversed(bands))

    # Fill in open-interval endpoints (±inf means unbounded)
    for i, b in enumerate(bands):
        if b["lo"] is None:
            b["lo"] = float("-inf") if i == 0 else bands[i - 1]["hi"]
        if b["hi"] is None:
            b["hi"] = float("inf") if i == len(bands) - 1 else bands[i + 1]["lo"]
    for b in bands:
        if b["hi"] <= b["lo"]:
            b["hi"] = b["lo"] + 1e-9

    return {"unit": unit, "bands": bands}


def locate(score, bands):
    """Returns (band index, relative position within band 0-1)."""
    n = len(bands)
    for i, b in enumerate(bands):
        lo, hi = b["lo"], b["hi"]
        if score < lo:
            return 0, 0.0
        if lo <= score <= hi:
            if math.isinf(lo) or math.isinf(hi):
                # Open interval: an unbounded side cannot be mapped linearly, so the pointer sits mid-band
                return i, 0.5
            span = hi - lo
            intra = 0.0 if span <= 0 else (score - lo) / span
            return i, max(0.0, min(1.0, intra))
    return n - 1, 1.0


# ---------------------------------------------------------------- Color palette

_RAMP = [
    (0.00, (0x2f, 0x9e, 0x6e)),   # green
    (0.42, (0x8f, 0xbf, 0x4d)),   # yellow-green
    (0.62, (0xe0, 0xa9, 0x2e)),   # amber
    (0.80, (0xe0, 0x7b, 0x39)),   # orange
    (1.00, (0xd6, 0x45, 0x45)),   # red
]


def ramp_color(pos):
    pos = max(0.0, min(1.0, pos))
    for i in range(len(_RAMP) - 1):
        p0, c0 = _RAMP[i]
        p1, c1 = _RAMP[i + 1]
        if p0 <= pos <= p1:
            k = 0.0 if p1 == p0 else (pos - p0) / (p1 - p0)
            rgb = tuple(int(round(c0[j] + (c1[j] - c0[j]) * k)) for j in range(3))
            return "#%02x%02x%02x" % rgb
    return "#d64545"


NEUTRAL = ["#c9d6e4", "#a8bdd4", "#8aa3bf", "#6f8aa8", "#587491", "#45607a"]


def band_colors(n, neutral=False):
    if n == 1:
        return ["#8a99ab"]
    if neutral:
        return [NEUTRAL[min(int(i * len(NEUTRAL) / n), len(NEUTRAL) - 1)] for i in range(n)]
    return [ramp_color(i / (n - 1)) for i in range(n)]


# ---------------------------------------------------------------- Geometry

def _pt(t, r, cx, cy):
    ang = math.pi * (1.0 - t)
    return cx + r * math.cos(ang), cy - r * math.sin(ang)


def _arc(t0, t1, r, cx, cy):
    x0, y0 = _pt(t0, r, cx, cy)
    x1, y1 = _pt(t1, r, cx, cy)
    return f"M {x0:.2f} {y0:.2f} A {r} {r} 0 0 1 {x1:.2f} {y1:.2f}"


def fmt_num(v):
    if v is None:
        return ""
    if isinstance(v, float) and math.isinf(v):
        return "…"
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    s = f"{v:.4f}".rstrip("0").rstrip(".")
    return s


def fmt_score(score, unit):
    if score is None:
        return "—"
    s = fmt_num(score)
    return s + unit


# ---------------------------------------------------------------- SVG

def gauge_svg(score, parsed, width=200, show_ticks=True, show_value=True,
              value_text=None, unit_override=None, value_font=19, neutral=False):
    """Generate the 180-degree half-ring gauge SVG string.

    parsed: return value of parse_bands()
    """
    bands = parsed["bands"]
    unit = unit_override if unit_override is not None else parsed["unit"]
    n = len(bands)
    colors = band_colors(n, neutral)

    cx, cy, r, sw = 100.0, 102.0, 76.0, 13.0
    h = 138.0

    idx, intra = locate(score, bands)
    t = (idx + intra) / n

    out = []
    out.append(
        f'<svg viewBox="0 0 200 {h}" width="{width}" height="{width * h / 200:.1f}" '
        f'xmlns="http://www.w3.org/2000/svg" class="gauge">'
    )
    # Base track
    out.append(
        f'<path d="{_arc(0, 1, r, cx, cy)}" fill="none" stroke="#eef1f6" '
        f'stroke-width="{sw}" stroke-linecap="butt"/>'
    )
    # Band arc
    for i, b in enumerate(bands):
        t0, t1 = i / n, (i + 1) / n
        gap = 0.004 if n > 1 else 0
        out.append(
            f'<path d="{_arc(t0 + gap, t1 - gap, r, cx, cy)}" fill="none" '
            f'stroke="{colors[i]}" stroke-width="{sw}" stroke-linecap="butt"/>'
        )
    # Band boundary ticks + threshold labels
    if show_ticks and n > 1:
        for i in range(1, n):
            tt = i / n
            x0, y0 = _pt(tt, r - sw / 2 - 1, cx, cy)
            x1, y1 = _pt(tt, r + sw / 2 + 1, cx, cy)
            out.append(
                f'<line x1="{x0:.2f}" y1="{y0:.2f}" x2="{x1:.2f}" y2="{y1:.2f}" '
                f'stroke="#ffffff" stroke-width="1.8"/>'
            )
        for i in range(1, n):
            tt = i / n
            lx, ly = _pt(tt, r + sw / 2 + 11, cx, cy)
            anchor = "middle"
            if tt < 0.16:
                anchor = "start"
            elif tt > 0.84:
                anchor = "end"
            out.append(
                f'<text x="{lx:.2f}" y="{ly + 3.2:.2f}" text-anchor="{anchor}" '
                f'class="gauge-tick" style="font-size:8.5px;fill:#8a97a8">'
                f'{bands[i].get("label") or fmt_num(bands[i]["lo"])}'
                f'{"" if bands[i].get("label") else unit}</text>'
            )
    # Pointer (tapered)
    tip_r = r - 15
    tx, ty = _pt(t, tip_r, cx, cy)
    dx, dy = tx - cx, ty - cy
    ln = math.hypot(dx, dy) or 1.0
    px, py = -dy / ln, dx / ln
    bw = 2.7
    out.append(
        f'<polygon points="{cx + px * bw:.2f},{cy + py * bw:.2f} '
        f'{cx - px * bw:.2f},{cy - py * bw:.2f} {tx:.2f},{ty:.2f}" fill="#1a2430"/>'
    )
    out.append(f'<circle cx="{cx}" cy="{cy}" r="5" fill="#1a2430"/>')
    out.append(f'<circle cx="{cx}" cy="{cy}" r="2" fill="#ffffff"/>')
    # Value (below the arc, clear of the pointer)
    if show_value:
        txt = value_text if value_text is not None else fmt_score(score, unit)
        # Text labels (e.g. "High risk") shrink with length to avoid overflowing the dial
        if value_text is not None and len(str(txt)) > 6:
            value_font = max(9, int(value_font * 6 / len(str(txt))))
        out.append(
            f'<text x="{cx}" y="{cy + 30}" text-anchor="middle" '
            f'class="gauge-value" style="font-size:{value_font}px">{txt}</text>'
        )
    out.append("</svg>")
    return "".join(out)


def band_legend(parsed, unit_override=None, neutral=False):
    """Band legend: color swatch + band text."""
    bands = parsed["bands"]
    unit = unit_override if unit_override is not None else parsed["unit"]
    colors = band_colors(len(bands), neutral)
    items = []
    for i, b in enumerate(bands):
        lo, hi = b["lo"], b["hi"]
        if i == 0 and b["raw"].startswith("<"):
            label = f"<{fmt_num(hi)}{unit}"
        elif i == len(bands) - 1 and ("+" in b["raw"] or "and above" in b["raw"].lower() or b["raw"].startswith(">")):
            label = f"≥{fmt_num(lo)}{unit}"
        else:
            label = f"{fmt_num(lo)}–{fmt_num(hi)}{unit}"
        items.append(
            f'<span class="lg-item"><i style="background:{colors[i]}"></i>{label}</span>'
        )
    return '<div class="lg">' + "".join(items) + "</div>"


def band_rows(parsed, unit_override=None, neutral=False):
    """Band details: [(color, band text)] for the sidebar legend."""
    bands = parsed["bands"]
    unit = unit_override if unit_override is not None else parsed["unit"]
    colors = band_colors(len(bands), neutral)
    rows = []
    for i, b in enumerate(bands):
        lo, hi = b["lo"], b["hi"]
        raw = b.get("raw", "")
        if b.get("label"):
            label = b["label"]
        elif i == 0 and raw.startswith("<"):
            label = f"< {fmt_num(hi)}{unit}"
        elif i == len(bands) - 1 and ("+" in raw or "and above" in raw.lower() or raw.startswith(">")):
            label = f"≥ {fmt_num(lo)}{unit}"
        elif math.isinf(lo):
            label = f"< {fmt_num(hi)}{unit}"
        elif math.isinf(hi):
            label = f"≥ {fmt_num(lo)}{unit}"
        else:
            label = f"{fmt_num(lo)} – {fmt_num(hi)}{unit}"
        rows.append((colors[i], label))
    return rows
