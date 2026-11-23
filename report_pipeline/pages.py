"""Page construction: cover, dashboard, body chapters."""
from __future__ import annotations

import re

from . import components as C
from .mdparse import blocks, inline, parse_table

DISEASE_ICON = [
    # English keywords (English report)
    ("liver", "liver"), ("hepat", "liver"), ("cirrhos", "liver"),
    ("cardiac", "heart"), ("coronary", "heart"), ("cardiovascular", "heart"),
    ("heart", "heart"), ("stroke", "brain"), ("cerebral", "brain"),
    ("brain", "brain"), ("kidney", "kidney"), ("renal", "kidney"),
    ("lung", "lung"), ("pulmonary", "lung"), ("copd", "lung"),
    ("asthma", "lung"), ("breast", "breast"), ("colorect", "colon"),
    ("colon", "colon"), ("gastric", "colon"), ("esophageal", "colon"),
    ("pancrea", "colon"), ("prostate", "prostate"), ("osteopor", "bone"),
    ("bone", "bone"), ("thyroid", "screening"), ("diabet", "lifestyle"),
    ("metabolic", "lifestyle"), ("gout", "lifestyle"), ("thromb", "heart"),
    ("venous", "heart"), ("hypertens", "heart"), ("lipid", "lifestyle"),
    ("dementia", "brain"), ("alzheimer", "brain"),
]


SYNONYMS = {
    # English disease-name synonyms
    "fatty liver": ["fatty liver", "NAFLD", "MAFLD", "hepatic steatosis"],
    "gout": ["gout", "hyperuricemia", "uric acid"],
    "copd": ["COPD", "chronic obstructive"],
    "vte": ["VTE", "venous thromboembolism", "deep vein thrombosis"],
    "stroke": ["stroke", "cerebral infarction", "ischaemic stroke"],
    "ckd": ["CKD", "chronic kidney disease", "renal"],
    "heart failure": ["heart failure", "cardiac failure"],
    "colorectal": ["colorectal", "colon cancer", "rectal cancer"],
    "asthma": ["asthma"],
    "ascvd": ["ASCVD", "atherosclerotic"],
    "thyroid": ["thyroid"],
    "diabetes": ["diabetes", "T2DM", "type 2 diabetes"],
    "hypertension": ["hypertension", "high blood pressure"],
    "osteoporosis": ["osteoporosis", "bone mineral density"],
}


_STOP = {"disease", "cancer", "chronic", "syndrome", "risk", "the", "and", "of"}


def match_disease(title, diseases):
    """Match md subsections to diseases by name/synonym."""
    tl = title.lower()
    for d in diseases:
        if d["name"] and d["name"] in title:
            return d
    for d in diseases:
        for kw in SYNONYMS.get(d["name"], []):
            if kw.lower() in tl:
                return d
    # English fallback: a salient word from the disease name in the title counts as a match
    for d in diseases:
        name = (d["name"] or "").lower()
        for w in re.split(r"[^a-z0-9]+", name):
            if len(w) >= 4 and w not in _STOP and w in tl:
                return d
    return None


def disease_icon(name, cls="dz-ic"):
    low = (name or "").lower()
    for kw, ic in DISEASE_ICON:
        if kw.lower() in low:
            return C.ui_icon(ic, cls)
    return C.ui_icon("risk", cls)


# Abnormal item -> asset-library icon. Order matters: ratio items must precede ALT/AST.
# This is a name-to-asset map with no patient values, so it is safe across patients.
FINDING_ICON = [
    (("AST/ALT", "transaminase ratio", "ratio"), "png",
     "1_jian_cha_lei/1_jian_cha_lei_-2_gan_dan_gong_neng_hua_yan/3_zhuan_an_mei_bi_zhi_AST_ALT_yan_sheng_bi_zhi_-_hong.png"),
    (("HbA1c", "glycated hemoglobin"), "png",
     "1_jian_cha_lei/1_jian_cha_lei_-3_xue_zhi_xue_tang_yu_dai_xie/7_tang_hua_xue_hong_dan_bai_-_bai_fen_bi_-_hong.png"),
    (("ALT", "alanine aminotransferase"), "png",
     "1_jian_cha_lei/1_jian_cha_lei_-2_gan_dan_gong_neng_hua_yan/2_gu_bing_zhuan_an_mei_ALT_-_sheng_hua_nong_du_-_hong.png"),
    (("AST", "aspartate aminotransferase"), "png",
     "1_jian_cha_lei/1_jian_cha_lei_-2_gan_dan_gong_neng_hua_yan/1_gu_cao_zhuan_an_mei_AST_-_sheng_hua_nong_du_-_hong.png"),
    (("uric acid", "UA"), "png",
     "1_jian_cha_lei/1_jian_cha_lei_-4_shen_gong_neng_yu_dian_jie_zhi/3_xue_niao_suan_-_sheng_hua_nong_du_-_hong.png"),
    (("LDL", "low-density lipoprotein"), "png",
     "1_jian_cha_lei/1_jian_cha_lei_-3_xue_zhi_xue_tang_yu_dai_xie/4_di_mi_du_zhi_dan_bai_dan_gu_chun_-_sheng_hua_nong_du_-_cheng.png"),
    (("HDL", "high-density lipoprotein"), "png",
     "1_jian_cha_lei/1_jian_cha_lei_-3_xue_zhi_xue_tang_yu_dai_xie/3_gao_mi_du_zhi_dan_bai_dan_gu_chun_-_sheng_hua_nong_du_-_hong.png"),
    (("total cholesterol", "TC"), "png",
     "1_jian_cha_lei/1_jian_cha_lei_-3_xue_zhi_xue_tang_yu_dai_xie/1_zong_dan_gu_chun_-_sheng_hua_nong_du_-_hong.png"),
    (("triglyceride", "TG"), "png",
     "1_jian_cha_lei/1_jian_cha_lei_-3_xue_zhi_xue_tang_yu_dai_xie/2_gan_you_san_zhi_-_sheng_hua_nong_du_-_hong.png"),
    (("glucose", "GLU"), "png",
     "1_jian_cha_lei/1_jian_cha_lei_-3_xue_zhi_xue_tang_yu_dai_xie/5_xue_tang_han_can_hou_-_sheng_hua_nong_du_-_hong.png"),
    (("BMI", "body mass index"), "png",
     "1_jian_cha_lei/1_jian_cha_lei_-12_ji_chu_ti_zheng_yu_ce_liang/3_ti_zhong_zhi_shu_BMI_-_zhi_shu_-_hong.png"),
    (("weight", "waist"), "png",
     "1_jian_cha_lei/1_jian_cha_lei_-12_ji_chu_ti_zheng_yu_ce_liang/2_ti_zhong_-_xing_tai_-_hong.png"),
    (("creatinine", "eGFR", "glomerular filtration"), "png",
     "1_jian_cha_lei/1_jian_cha_lei_-4_shen_gong_neng_yu_dian_jie_zhi/2_xue_ji_gan_-_sheng_hua_nong_du_-_hong.png"),
    (("hemoglobin", "Hb"), "png",
     "1_jian_cha_lei/1_jian_cha_lei_-1_xue_ye_chang_gui_yu_ning_xue/3_xue_hong_dan_bai_-_sheng_hua_nong_du_-_hong.png"),
    (("WBC", "white blood cell"), "png",
     "1_jian_cha_lei/1_jian_cha_lei_-1_xue_ye_chang_gui_yu_ning_xue/1_bai_xi_bao_-_sheng_hua_nong_du_-_hong.png"),
    (("platelet", "PLT"), "png",
     "1_jian_cha_lei/1_jian_cha_lei_-1_xue_ye_chang_gui_yu_ning_xue/4_xue_xiao_ban_ji_shu_-_sheng_hua_nong_du_-_hong.png"),
    (("urine protein", "urine glucose", "urine ketone", "urine occult blood"), "png",
     "1_jian_cha_lei/1_jian_cha_lei_-10_guo_min_yuan_jian_ce_11_niao_ye_jian_cha/11-1_niao_dan_bai_-_sheng_hua_nong_du_-_hong.png"),
    (("AFP", "alpha-fetoprotein"), "png",
     "1_jian_cha_lei/1_jian_cha_lei_-6_zhong_liu_biao_zhi_wu/4_jia_tai_dan_bai_AFP_-_sheng_hua_nong_du_-_hong.png"),
    (("CEA", "carcinoembryonic antigen"), "png",
     "1_jian_cha_lei/1_jian_cha_lei_-6_zhong_liu_biao_zhi_wu/3_ai_pei_kang_yuan_CEA_-_sheng_hua_nong_du_-_hong.png"),
    (("PSA", "prostate-specific antigen"), "png",
     "1_jian_cha_lei/1_jian_cha_lei_-6_zhong_liu_biao_zhi_wu/1_zong_qian_lie_xian_te_yi_xing_kang_yuan_PSA_-_sheng_hua_nong_du_-_hong.png"),
    (("blood pressure", "systolic", "diastolic"), "svg", "heart"),
    (("tooth", "dental", "oral"), "svg", "screening"),
    (("chest", "lung", "respiratory"), "svg", "lung"),
    (("ultrasound", "imaging", "CT", "X-ray", "liver"), "svg", "liver"),
]


def finding_icon(name):
    """Get a semantic icon by abnormal-item name; fall back to the generic warning icon."""
    low = (name or "").lower()
    for kws, kind, src in FINDING_ICON:
        if any(k.lower() in low for k in kws):
            html = C.png_icon(src, 34) if kind == "png" else C.ui_icon(src, "pic")
            if html:
                return html
    return C.ui_icon("warning", "pic")


# ---------------------------------------------------------------- Primitives
def sec_head(num, title, icon_name="report"):
    n = f'<span class="sec-num">{num}</span>' if num else ""
    return (
        f'<div class="sec-head">{n}<span class="sec-ic">{C.ui_icon(icon_name, "uic")}</span>'
        f"<h2>{inline(title)}</h2></div>"
    )


def sub_head(title):
    return f'<div class="sub-head">{inline(title)}</div>'


def ps(text):
    return f'<p class="t">{inline(text)}</p>'


def ul(items):
    return '<ul class="t">' + "".join(f"<li>{inline(x)}</li>" for x in items) + "</ul>"


def ol(items):
    return '<ol class="t">' + "".join(f"<li>{inline(x)}</li>" for x in items) + "</ol>"


def table(rows):
    if not rows:
        return ""
    head = "".join(f"<th>{inline(c)}</th>" for c in rows[0])
    body = "".join(
        "<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>" for r in rows[1:]
    )
    return f'<table class="hub"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


def labeled_blocks(body):
    """Split a `**label**: content` paragraph into (label, content)."""
    out, cur = [], None
    for ln in body.splitlines():
        m = re.match(r"^\s*\*\*(.+?)\*\*\s*[：:]\s*(.*)$", ln)
        if m:
            cur = [m.group(1).strip(), [m.group(2).strip()] if m.group(2).strip() else []]
            out.append(cur)
        elif cur is not None:
            cur[1].append(ln)
    return [(k, "\n".join(v).strip()) for k, v in out]


def render_block_content(body):
    return C.render_blocks(blocks(body))


def walk(body, handler):
    """Render blocks in order: the handler returns HTML; None falls back to the default renderer.

    Ensure no block (paragraph/list/table/quote) is lost when no dedicated renderer recognizes it.
    """
    out = []
    for kind, val in blocks(body):
        html = handler(kind, val)
        out.append(html if html is not None else C.render_blocks([(kind, val)]))
    return "".join(out)


def split_lead(body):
    """Split out body text before the first `**label**:` so leading paragraphs are not swallowed by labeled_blocks."""
    lines = body.splitlines()
    for i, ln in enumerate(lines):
        if re.match(r"^\s*\*\*(.+?)\*\*\s*[：:]", ln):
            return "\n".join(lines[:i]).strip(), "\n".join(lines[i:])
    return body.strip(), ""


def two_col(cards, cls):
    """Lay out card lists as a two-column table: row heights auto-align, avoiding inline-block misalignment."""
    rows = []
    for i in range(0, len(cards), 2):
        pair = cards[i:i + 2]
        cells = "".join(f"<td>{c}</td>" for c in pair)
        if len(pair) == 1:
            cells += "<td></td>"
        rows.append(f"<tr>{cells}</tr>")
    return f'<table class="{cls}"><tbody>{"".join(rows)}</tbody></table>'


# ---------------------------------------------------------------- Cover
def cover(p, preamble=""):
    d = p.demo
    chips = []
    if d["age"]:
        chips.append(f'Age {d["age"]}')
    if d["sex"]:
        chips.append(d["sex"])
    if d["height"]:
        chips.append(f'Height {d["height"]:g} cm')
    if d["weight"]:
        chips.append(f'Weight {d["weight"]:g} kg')
    if d["bmi"]:
        chips.append(f'BMI {d["bmi"]:g} kg/m²')
    if d["waist"]:
        chips.append(f'Waist {d["waist"]:g} cm')
    if d["sbp"]:
        chips.append(f'BP {d["sbp"]}/{d["dbp"]} mmHg')
    chip_html = "".join(f'<span class="chip c-none">{c}</span>' for c in chips)
    if preamble:
        chip_html = f'<div class="cover-meta">{inline(preamble)}</div>'
    else:
        chip_html = f'<div class="chips">{chip_html}</div>'

    # The cover no longer shows the four-cell "abnormal indicators" grid: section 1.2 already lists measured values item by item,
    # Repeating it here would conflict with later text (e.g. LDL-C 2.81 is in the low-risk band; flagging it red would mislead).
    hi = sum(1 for x in p.diseases if x["rank"] == 3)
    mid = sum(1 for x in p.diseases if x["rank"] == 2)
    lo = sum(1 for x in p.diseases if x["rank"] == 1)
    na = sum(1 for x in p.diseases if x["rank"] == 0)
    summary = (
        '<div class="dash-sum">'
        f'<div class="sum-item hi"><div class="sum-n">{hi}</div><div class="sum-l">High-risk conditions</div></div>'
        f'<div class="sum-item mid"><div class="sum-n">{mid}</div><div class="sum-l">Medium-risk conditions</div></div>'
        f'<div class="sum-item low"><div class="sum-n">{lo}</div><div class="sum-l">Low-risk conditions</div></div>'
        f'<div class="sum-item none"><div class="sum-n">{na}</div><div class="sum-l">Not stratified</div></div>'
        "</div>"
    )
    return (
        '<section class="cover">'
        '<div class="cover-t">Personal Multimorbidity Risk Interpretation and Health Management Report</div>'
        '<div class="cover-sub">Primary Prevention and Health Management</div>'
        f'{chip_html}'
        f"{summary}"
        "</section>"
    )


# ---------------------------------------------------------------- Dashboard
def dashboard(p, cohort):
    hi = [d for d in p.diseases if d["rank"] == 3]
    mid = [d for d in p.diseases if d["rank"] == 2]
    lo = [d for d in p.diseases if d["rank"] == 1]
    na = [d for d in p.diseases if d["rank"] == 0]

    out = [sec_head("01", "Individual Risk Profile", "dashboard")]
    if hi:
        out.append('<div class="lowsec-h">High-risk conditions</div>')
        out += [C.disease_card(d, cohort, disease_icon(d["name"])) for d in hi]
    if mid:
        out.append('<div class="lowsec-h">Medium-risk conditions</div>')
        out += [C.disease_card(d, cohort, disease_icon(d["name"])) for d in mid]
    if lo:
        out.append('<div class="lowsec-h">Low-risk conditions</div>')
        out.append(two_col([low_cell(d, cohort) for d in lo], "low-tbl"))
    if na:
        out.append('<div class="lowsec-h">Not uniformly stratified</div>')
        out.append(two_col([low_cell(d, cohort) for d in na], "low-tbl"))
    out.append(
        '<div class="dash-note">Each model contributing to a condition occupies one half-ring; the arc is '
        "divided evenly across that model's own risk bands, ticks mark the original thresholds, and the "
        "pointer is linearly mapped to the actual value within its band. A condition's tier is the highest "
        "risk among its models, so the same condition never appears as both high and low risk.</div>"
    )
    return "".join(out)


def low_cell(d, cohort):
    """Low-risk / untiered diseases: draw half-rings for every visible model under that disease."""
    from .cohort import resolve_bands

    kept, hidden = C.visible_models(d)
    gauges = []
    for m in kept:
        bands, source, _ = resolve_bands(m["model"], cohort, C._score(m["score"]), m["rank"])
        n = len(cohort.dist.get(m["model"], {}).get("scores", []))
        gauges.append(
            C.gauge_block(m["label"], bands, source, C._score(m["score"]),
                          m["level"], m["rank"], compact=True, cohort_n=n,
                          unit=m.get("unit", ""), chip=False)
        )
    more = f'<div class="lowcell-more">{hidden} low-risk model(s) hidden</div>' if hidden else ""
    return (
        f'<div class="lowcell"><div class="lowcell-head">{disease_icon(d["name"], "ic")}'
        f'<span class="lowcell-name">{inline(d["name"])}</span>'
        f'<span class="chip {C.CHIP[d["rank"]]}">{d["level"]}</span></div>'
        f'<div class="lowcell-gauges">{"".join(gauges)}</div>{more}</div>'
    )


# ---------------------------------------------------------------- Body

def split_table(body):
    """Split the body into (rows of the first table, all remaining text) so special renderers do not swallow content."""
    lines = body.splitlines()
    rows, rest, i, got = [], [], 0, False
    while i < len(lines):
        if lines[i].strip().startswith("|"):
            j = i
            while j < len(lines) and lines[j].strip().startswith("|"):
                j += 1
            if not got:
                rows = parse_table("\n".join(lines[i:j]))
                got = True
            else:
                rest.extend(lines[i:j])
            i = j
            continue
        rest.append(lines[i])
        i += 1
    return rows, "\n".join(rest).strip()


def findings_cards(sec):
    rows, rest = split_table(sec["body"])
    if len(rows) < 2:
        return render_block_content(sec["body"])
    head = rows[0]

    def col(*words):
        for i, h in enumerate(head):
            hl = h.lower()
            if any(w in h or w.lower() in hl for w in words):
                return i
        return None

    ci_name = col("abnormal item", "item", "indicator", "name", "exam",
                  "item", "finding", "indicator", "name", "test")
    ci_val = col("measured", "result", "value",
                 "result", "value", "measured")
    ci_ref = col("reference", "ref.")
    if ci_name is None:
        ci_name = 1 if len(head) > 2 else 0
    if ci_val is None:
        ci_val = 2 if len(head) > 2 else 1
    skip = {ci_name, ci_val, ci_ref}
    # The card layout presents item / measured value / reference range as three columns, so header text disappears;
    # Add a field-description row so table semantics are preserved.
    legend_cols = [head[j] for j in (ci_name, ci_val, ci_ref)
                   if j is not None and j < len(head) and head[j].strip()]
    legend = (f'<div class="fcard-legend">{" · ".join(inline(c) for c in legend_cols)}</div>'
              if legend_cols else "")
    cards = []
    for i, r in enumerate(rows[1:], 1):
        if len(r) <= max(ci_name, ci_val):
            continue
        name = r[ci_name]
        val = r[ci_val].replace("**", "").strip()
        ic = finding_icon(name)
        has_val = val not in ("—", "-", "–", "")
        val_html = (f'<span class="fcard-val">{inline(val)}</span>' if has_val
                    else '<span class="fcard-val norm">Imaging / sign</span>')
        ref = r[ci_ref].replace("**", "").strip() if ci_ref is not None and ci_ref < len(r) else ""
        ref_label = head[ci_ref] if ci_ref is not None and ci_ref < len(head) else "Reference"
        ref_html = (f'<div class="fcard-ref">{inline(ref_label)} {inline(ref)}</div>'
                    if ref not in ("—", "-", "–", "") else "")
        rows_html = "".join(
            f'<div class="fcard-row"><b>{inline(head[j])}</b>　{inline(r[j])}</div>'
            for j in range(len(r))
            if j not in skip and j < len(head) and r[j].strip()
            and "no." not in head[j].lower()
        )
        cards.append(
            f'<div class="fcard"><div class="fcard-ic">{ic}</div>'
            f'<div class="fcard-main"><div class="fcard-top">'
            f'<span class="fcard-idx">{i}</span><span class="fcard-name">{inline(name)}</span>'
            f'{val_html}</div>{ref_html}{rows_html}</div></div>'
        )
    return legend + "".join(cards) + (render_block_content(rest) if rest else "")


CARD_RE = re.compile(r"^\*\*((?:Strategy|Step|Action)\s*[^*]*)\*\*\s*$",
                     re.IGNORECASE)


def _kv_list(items):
    """Turn a `**label**: content` list into kv rows; non-label items join the previous row's sublist."""
    groups, cur = [], None
    for x in items:
        m = re.match(r"^\*\*(.+?)\*\*\s*[：:]\s*(.*)$", x)
        if m:
            cur = [m.group(1).strip(), [m.group(2).strip()] if m.group(2).strip() else []]
            groups.append(cur)
        elif cur is not None:
            cur[1].append(x)
        else:
            groups.append([None, [x]])
    out = []
    for label, lines in groups:
        if label is None:
            out.append(f'<p class="p">{inline(lines[0])}</p>')
            continue
        if len(lines) > 1:
            content = '<ul class="ul">' + "".join(f"<li>{inline(t)}</li>" for t in lines) + "</ul>"
        else:
            content = inline(lines[0]) if lines else ""
        out.append(
            f'<div class="kv"><div class="kv-k">{inline(label)}:</div>'
            f'<div class="kv-v">{content}</div></div>'
        )
    return "".join(out)


def strategy_cards(sec):
    """Strategy cards: `**Strategy X: ...**` or `**(N) ...**` opens a card, lists become kv rows, other blocks kept as-is."""
    cards, cur = [], None
    for kind, val in blocks(sec["body"]):
        if kind == "p":
            m = CARD_RE.match(val)
            if m:
                cur = {"title": m.group(1).strip(), "blocks": []}
                cards.append(cur)
                continue
        if cur is None:
            cards.append({"title": None, "blocks": [(kind, val)]})
        else:
            cur["blocks"].append((kind, val))
    if not any(c["title"] for c in cards):
        return render_block_content(sec["body"])
    out = []
    for c in cards:
        if c["title"] is None:
            out.append(C.render_blocks(c["blocks"]))
            continue
        m = re.match(r"^(Strategy\s*[0-9IVX]+)[:]\s*(.*)$", c["title"], re.I)
        m_en = re.match(r"^(Strategy\s*\d+)[:.]\s*(.*)$", c["title"], re.IGNORECASE)
        if m:
            head = f'<span class="stg-num">{inline(m.group(1))}:</span>{inline(m.group(2))}'
        elif m_en:
            head = f'<span class="stg-num">{inline(m_en.group(1))}:</span>{inline(m_en.group(2))}'
        else:
            head = inline(c["title"])
        inner = []
        for kind, val in c["blocks"]:
            inner.append(_kv_list(val) if kind == "ul" else C.render_blocks([(kind, val)]))
        out.append(f'<div class="stg"><div class="stg-title">{head}</div>{"".join(inner)}</div>')
    return "".join(out)


def _act_table(rows, gi):
    head = rows[0]
    has_pri = bool(re.search(r"priority|no\.|level|tier", head[0], re.IGNORECASE))
    ti, wi, mf = (1, 2, 3) if has_pri else (0, 1, 2)
    out = []
    for r in rows[1:]:
        pri = re.sub(r"\D", "", r[0]) if has_pri else str(gi)
        lvl = min(int(pri or gi), 5)
        title = r[ti] if len(r) > ti else ""
        why = r[wi] if len(r) > wi else ""
        meta = "".join(
            f'<span><b>{inline(head[j])}</b> {inline(r[j])}</span>'
            for j in range(mf, min(len(r), len(head)))
        )
        badge = r[0] if has_pri else str(gi)
        out.append(
            f'<div class="act"><div class="act-pri p{lvl}">{inline(badge)}</div>'
            f'<div class="act-main"><div class="act-title">{inline(title)}</div>'
            f'<div class="act-why">{inline(why)}</div>'
            f'<div class="act-meta">{meta}</div></div></div>'
        )
    return "".join(out)


def action_items(sec):
    """Action list: table -> priority cards, `**group heading**` -> group header, other blocks kept as-is."""
    state = {"gi": 0}

    def h(kind, val):
        if kind == "p":
            m = re.match(r"^\*\*(.+?)\*\*\s*$", val)
            if m:
                return f'<div class="act-group">{inline(m.group(1))}</div>'
            return None
        if kind == "table" and val and len(val) >= 2:
            state["gi"] += 1
            return _act_table(val, state["gi"])
        return None

    return walk(sec["body"], h)


def goal_cards(sec):
    """Quantitative targets: list -> target cards; other blocks kept as-is."""
    def h(kind, val):
        if kind == "ul":
            cells = []
            for it in val:
                m = re.match(r"^\*\*(.+?)\*\*\s*[：:]\s*(.*)$", it)
                if m:
                    cells.append(
                        f'<div class="goal"><div class="goal-k">{inline(m.group(1))}:</div>'
                        f'<div class="goal-v">{inline(m.group(2))}</div></div>'
                    )
                else:
                    cells.append(f'<div class="goal"><div class="goal-v">{inline(it)}</div></div>')
            return '<div class="goal-grid">' + "".join(cells) + "</div>"
        return None

    return walk(sec["body"], h)


def disease_detail(sec, p, cohort):
    title = sec["title"]
    name = re.sub(r"[（(].*?[)）]", "", title).strip()
    dis = match_disease(title, p.diseases)
    left = ""
    if dis:
        kept, hidden = C.visible_models(dis)
        rows = []
        for m in kept:
            rows.append(
                f'<div class="kv"><div class="kv-k">{inline(m["label"])}</div>'
                f'<div class="kv-v">{inline(str(m["score"]))} '
                f'<span class="chip {C.CHIP[m["rank"]]}">{m["level"] or "Not stratified"}</span></div></div>'
            )
        left = "".join(rows)
        if hidden:
            left += f'<div class="glegend-note">{hidden} low-risk model(s) hidden</div>'
    lead, rest = split_lead(sec["body"])
    kv = labeled_blocks(rest) if rest else []
    right = "".join(
        f'<div class="kv"><div class="kv-k">{inline(k)}:</div>'
        f'<div class="kv-v">{render_block_content(v)}</div></div>'
        for k, v in kv
    )
    if not kv:
        right = render_block_content(sec["body"])
    elif lead:
        right = render_block_content(lead) + right
    sub = re.search(r"[（(](.*?)[)）]", title)
    return (
        f'<div class="dz5"><div class="dz5-head">{disease_icon(name)}'
        f'<span class="dz5-name">{inline(name)}</span>'
        f'<span class="dz5-sub">{inline(sub.group(1) if sub else "")}</span></div>'
        f'<div class="dz5-cols"><div class="dz5-col">'
        f'<div class="dz5-col-h">Model tiers</div>{left}</div>'
        f'<div class="dz5-col"><div class="dz5-col-h">Recommendations</div>{right}</div></div></div>'
    )


# ---------------------------------------------------------------- 1.3 Normal items already tested
def negatives(sec):
    """Tested-normal items: lists -> two-column green cards, closing paragraph -> callout; other blocks kept as-is."""
    def h(kind, val):
        if kind == "ul":
            cards = []
            for t in val:
                m = re.match(r"^\*\*(.+?)\*\*[：:]\s*(.*)$", t)
                if m:
                    cards.append(
                        f'<div class="neg"><b>{inline(m.group(1))}:</b>{inline(m.group(2))}</div>'
                    )
                else:
                    cards.append(f'<div class="neg">{inline(t)}</div>')
            return two_col(cards, "neg-tbl")
        if kind == "p" and (val.startswith("**Significance of these negative results**")
                             or val.startswith("**Meaning of negative results**")
                             or val.lower().startswith("**what these negative")
                             or val.lower().startswith("**significance of these negative")):
            return f'<div class="callout ok">{inline(val)}</div>'
        return None

    return walk(sec["body"], h)


# ---------------------------------------------------------------- 2. Cross-specialty
MECH_RE = re.compile(
    r"^\*\*(?:（(\d+)）|\((\d+)\)|(\d+)[.)])\s*(.+?)\*\*[：:]\s*(.*)$"
)


def mech_cards(sec):
    """Cross-specialty integrated interpretation: `**(N) title**: body` -> numbered mechanism card; other blocks kept as-is."""
    def h(kind, val):
        if kind == "p":
            m = MECH_RE.match(val)
            if m:
                num = m.group(1) or m.group(2) or m.group(3)
                return (
                    f'<div class="mech"><div class="mech-chain">'
                    f'<span>({num}) {inline(m.group(4))}:</span></div>'
                    f'<div class="mech-body">{inline(m.group(5))}</div></div>'
                )
        return None

    return walk(sec["body"], h)


# ---------------------------------------------------------------- 3. Shared risk factors
def hub_table(sec):
    """Shared risk factors: promote the table to hub style; keep other blocks as-is."""
    def h(kind, val):
        if kind == "table" and val:
            head = "".join(f"<th>{inline(c)}</th>" for c in val[0])
            trs = "".join(
                "<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>"
                for r in val[1:]
            )
            return f'<table class="hub"><thead><tr>{head}</tr></thead><tbody>{trs}</tbody></table>'
        return None

    return walk(sec["body"], h)


def _has(t, *keys):
    """Title keyword matching (Chinese and English)."""
    tl = t.lower()
    return any(k in t or k.lower() in tl for k in keys)


def render_chapter(ch, p, cohort):
    """Chapter-level (##) heading dispatch: unrecognized chapters use the fallback renderer."""
    t = ch["title"]
    if _has(t, "cross-specialty", "cross-organ", "cross-disciplinary"):
        return mech_cards(ch)
    if _has(t, "shared risk factor", "risk hub", "common risk factor"):
        return hub_table(ch)
    return render_block_content(ch["body"])


def render_section(sec, p, cohort):
    t = sec["title"]
    if _has(t, "abnormal finding", "key abnormal"):
        return findings_cards(sec)
    if _has(t, "normal finding", "negative finding", "within normal"):
        return negatives(sec)
    if _has(t, "common strateg", "specific strateg", "coordinat",
            "common strateg", "shared strateg", "specific strateg",
            "condition-specific", "coordination", "reconcil"):
        return strategy_cards(sec)
    if _has(t, "action plan", "action checklist", "actionable plan"):
        return action_items(sec)
    if _has(t, "quantitative target", "target value", "treatment target"):
        return goal_cards(sec)
    if match_disease(t, p.diseases):
        return disease_detail(sec, p, cohort)
    return render_block_content(sec["body"])


def render_doc(doc, p, cohort):
    """Render the body in the md's original section order; unrecognized sections use the fallback renderer so nothing is lost."""
    out = []
    chapters = [s for s in doc.secs if s["level"] == 2]
    for ch in chapters:
        subs, inside = [], False
        for s in doc.secs:
            if s is ch:
                inside = True
                continue
            if inside:
                if s["level"] <= 2:
                    break
                subs.append(s)
        if not subs and not ch["body"]:
            continue
        out.append('<div class="chapter">')
        # The dashboard already uses "01 Individual Risk Profile" as its heading; the body does not repeat it
        if ch["title"] not in ("Individual Risk Profile",):
            out.append(sec_head("", ch["raw_title"], "report"))
        if ch["body"]:
            out.append(render_chapter(ch, p, cohort))
        for s in subs:
            out.append('<div class="sec">')
            out.append(sub_head(s["raw_title"]))
            out.append(render_section(s, p, cohort))
            out.append("</div>")
        out.append("</div>")
    return "".join(out)
