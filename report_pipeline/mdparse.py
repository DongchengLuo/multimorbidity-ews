"""Markdown report parsing: split the LLM-produced md into structured blocks.

Only "text transport + structural rearrangement"; no medical wording is rewritten.
"""
from __future__ import annotations

import re

HEAD_RE = re.compile(r"^(#{1,4})\s+(.*)$")
NUM_RE = re.compile(r"^\s*(?:\d+(?:\.\d+)*[.)]?)\s*")


def inline(text):
    """Inline markdown -> HTML (bold and escaping only)."""
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"`(.+?)`", r"<code>\1</code>", text)
    return text


def clean_title(t):
    return NUM_RE.sub("", t).strip()


def split_sections(md):
    """Returns [{'level','title','raw_title','num','body','chapter'}]."""
    lines = md.splitlines()
    secs, cur = [], None
    chapter = ""
    for ln in lines:
        m = HEAD_RE.match(ln)
        if m:
            level = len(m.group(1))
            title = m.group(2).strip()
            if level <= 2:
                chapter = clean_title(title)
            num = ""
            mm = re.match(r"^\s*(\d+(?:\.\d+)*)", title)
            if mm:
                num = mm.group(1)
            cur = {
                "level": level,
                "raw_title": title,
                "title": clean_title(title),
                "num": num,
                "chapter": chapter,
                "lines": [],
            }
            secs.append(cur)
        elif cur is not None:
            cur["lines"].append(ln)
    for s in secs:
        s["body"] = "\n".join(s["lines"]).strip()
    return secs


# ---------------------------------------------------------------- Semantic index
KEYWORDS = [
    ("profile", ["overall risk profile", "risk profile",
                 "individual risk profile", "overall risk profile", "risk profile"]),
    ("evidence", ["requires prompt confirmation", "evidence of established disease", "diagnosed",
                  "established disease", "evidence of disease", "already diagnosed"]),
    ("findings", ["key abnormal findings", "abnormal findings"]),
    ("negatives", ["tested normal", "normal results", "negative",
                   "normal findings", "negative findings", "within normal"]),
    ("tiers", ["risk stratification", "risk tiers", "tier definitions"]),
    ("common", ["common strategies", "shared strategies", "universal strategies"]),
    ("specific", ["specific strategies", "individualized strategies",
                  "specific strategies", "condition-specific", "disease-specific"]),
    ("coord", ["coordination", "follow-up cadence", "reconcil"]),
    ("lowrisk", ["low-risk conditions", "low risk conditions"]),
    ("actions", ["action plan", "action checklist", "actionable plan"]),
    ("goals", ["quantitative targets", "target values", "treatment targets"]),
    ("warning", ["seek care", "warning signs", "red flag",
                 "seek medical attention", "red flag", "when to seek"]),
    ("disclaimer", ["disclaimer", "notice"]),
]


class Doc:
    def __init__(self, md):
        self.md = md
        self.secs = split_sections(md)
        self.preamble = ""
        for s in self.secs:
            if s["level"] == 1:
                self.preamble = s["body"]
                break
        self.by_key = {}
        for key, words in KEYWORDS:
            for s in self.secs:
                tl = s["title"].lower()
                if key == "negatives" and "abnormal" in tl:
                    continue
                if any(w in s["title"] or w.lower() in tl for w in words):
                    self.by_key.setdefault(key, s)
                    break
        # Chapter 5 (disease details): take all level-3 headings after "5."
        self.disease_secs = []
        started = False
        for s in self.secs:
            if s["level"] <= 2:
                t = s["title"].strip()
                tl = t.lower()
                started = (
                    ("disease" in t.lower())
                    or t.startswith("5")
                    or t.startswith("5")
                    or "disease-specific" in tl
                    or "condition-specific" in tl
                    or "specific risk factors" in tl
                )
                continue
            if started and s["level"] == 3:
                self.disease_secs.append(s)

    def get(self, key):
        return self.by_key.get(key)

    def body(self, key):
        s = self.by_key.get(key)
        return s["body"] if s else ""

    def title(self, key):
        s = self.by_key.get(key)
        return s["title"] if s else ""

    def by_num(self, num):
        for s in self.secs:
            if s["num"] == num:
                return s
        return None

    def subsections(self, key):
        """Return all subheadings under a semantic block (e.g. strategy cards under 4.1)."""
        s = self.by_key.get(key)
        if not s:
            return []
        out, inside = [], False
        for x in self.secs:
            if x is s:
                inside = True
                continue
            if inside:
                if x["level"] <= s["level"]:
                    break
                out.append(x)
        return out


# ---------------------------------------------------------------- Block parsing
def parse_table(body):
    rows = []
    for ln in body.splitlines():
        ln = ln.strip()
        if not ln.startswith("|"):
            continue
        if re.match(r"^\|[\s:\-|]+\|$", ln):
            continue
        cells = [c.strip() for c in ln.strip("|").split("|")]
        rows.append(cells)
    return rows


def parse_list(body):
    out = []
    for ln in body.splitlines():
        m = re.match(r"^\s*[-*+]\s+(.*)$", ln)
        if m:
            out.append(m.group(1).strip())
    return out


def parse_ordered(body):
    out = []
    for ln in body.splitlines():
        m = re.match(r"^\s*(\d+)[.、)]\s+(.*)$", ln)
        if m:
            out.append((m.group(1), m.group(2).strip()))
    return out


def paragraphs(body):
    """Split on blank lines, keeping body text outside lists/tables."""
    out, buf = [], []
    for ln in body.splitlines():
        if not ln.strip():
            if buf:
                out.append(" ".join(buf).strip())
                buf = []
            continue
        if re.match(r"^\s*[-*+]\s+", ln) or ln.strip().startswith("|"):
            if buf:
                out.append(" ".join(buf).strip())
                buf = []
            continue
        buf.append(ln.strip())
    if buf:
        out.append(" ".join(buf).strip())
    return [p for p in out if p]


def blocks(body):
    """Split body text into ordered blocks: ('p',text) / ('ul',[..]) / ('ol',[(n,t)]) / ('table',rows)."""
    out = []
    lines = body.splitlines()
    i = 0
    while i < len(lines):
        ln = lines[i]
        s = ln.strip()
        if not s:
            i += 1
            continue
        if s.startswith("|"):
            j = i
            while j < len(lines) and lines[j].strip().startswith("|"):
                j += 1
            out.append(("table", parse_table("\n".join(lines[i:j]))))
            i = j
            continue
        if re.match(r"^\s*[-*+]\s+", ln):
            j = i
            while j < len(lines) and re.match(r"^\s*[-*+]\s+", lines[j]):
                j += 1
            out.append(("ul", parse_list("\n".join(lines[i:j]))))
            i = j
            continue
        if re.match(r"^\s*\d+[.、)]\s+", ln):
            j = i
            while j < len(lines) and re.match(r"^\s*\d+[.、)]\s+", lines[j]):
                j += 1
            out.append(("ol", parse_ordered("\n".join(lines[i:j]))))
            i = j
            continue
        if ln.startswith(">"):
            j = i
            buf = []
            while j < len(lines) and lines[j].strip().startswith(">"):
                buf.append(re.sub(r"^\s*>\s?", "", lines[j].strip()))
                j += 1
            out.append(("quote", " ".join(buf)))
            i = j
            continue
        j = i
        buf = []
        while j < len(lines) and lines[j].strip() and not lines[j].strip().startswith("|") \
                and not re.match(r"^\s*[-*+]\s+", lines[j]) and not re.match(r"^\s*\d+[.、)]\s+", lines[j]):
            buf.append(re.sub(r"^\s*>\s?", "", lines[j].strip()))
            j += 1
        out.append(("p", " ".join(buf)))
        i = j
    return out
