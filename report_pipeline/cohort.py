"""Data layer: read patients, model results, threshold references and population distributions from the batch directory.

Design principles
--------
* All patient data is read from batch artifacts; **no patient values are hard-coded**.
* Prefer the model's own ``gauge_reference``; infer from the same-batch distribution when absent
  (see :func:`resolve_bands`), noting "by population percentile" in the report.
"""
from __future__ import annotations

import csv
import glob
import json
import os
import re
import statistics
from collections import Counter, defaultdict

LEVEL_ORDER = {"high": 3, "medium": 2, "low": 1,
               "high": 3, "medium": 2, "moderate": 2, "low": 1}
LEVEL_NAME = {3: "High risk", 2: "Medium risk", 1: "Low risk", 0: "Not stratified"}


def _num(x):
    if x is None:
        return None
    s = str(x).strip().replace("%", "").replace(",", "")
    if not s or s.lower() in ("nan", "none", "-", "--"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


_MODEL_NAME_EN = None


def _model_name_en():
    """Optional model-name display overrides (report_pipeline/model_name_en.json).

    Model names are already English in this edition, so the override table is
    optional and normally absent; the function returns an empty mapping.
    """
    global _MODEL_NAME_EN
    if _MODEL_NAME_EN is None:
        path = os.path.join(os.path.dirname(__file__), "model_name_en.json")
        try:
            _MODEL_NAME_EN = json.load(open(path, encoding="utf-8"))
        except Exception:
            _MODEL_NAME_EN = {}
    return _MODEL_NAME_EN


def _clean_label(s):
    s = str(s or "").strip()
    s = re.sub(r"[(][^)]*(?:undefined|not provided|no fixed)[^)]*[)]", "", s, flags=re.I)
    s = re.sub(r"[(]continuous risk[^)]*[)]", "", s, flags=re.I)
    s = s.strip()
    return _model_name_en().get(s, s)


def _unit_of(v):
    s = str(v or "").strip()
    if s.endswith("%"):
        return "%"
    return ""


CATEGORY_RULES = [
    # (regex, tier-mapping function)
    (r"C-?TR\s*([1-6])", lambda n: 1 if n <= 3 else (2 if n == 4 else 3)),
    (r"TI-?RADS\s*([1-6])", lambda n: 1 if n <= 3 else (2 if n == 4 else 3)),
    (r"BI-?RADS\s*([0-6])", lambda n: 1 if n <= 3 else (2 if n == 4 else 3)),
    (r"Lung-?RADS\s*([1-4])([A-Z]?)", lambda n, x="": 1 if n <= 2 else (2 if n == 3 else 3)),
    (r"LI-?RADS\s*([1-5])", lambda n: 1 if n <= 2 else (2 if n == 3 else 3)),
    (r"FIB-?4[^0-9]{0,6}([0-9.]+)", None),
]


def level_rank(level):
    """Map risk-level text to 3/2/1/0.

    Supports: high/medium/low risk, C-TR3, BI-RADS 4a, Lung-RADS 3 and similar categorical grades.
    """
    if not level:
        return 0
    s = str(level)
    sl = s.lower()
    for k, v in LEVEL_ORDER.items():
        if k in s or k in sl:
            return v
    for pat, fn in CATEGORY_RULES:
        m = re.search(pat, s, re.I)
        if m and fn:
            try:
                return fn(*[int(g) if g.isdigit() else g for g in m.groups() if g])
            except Exception:
                continue
    return 0


def is_continuous(level):
    """Whether the model gives only continuous values with no tiers."""
    s = str(level or "")
    return ("continuous" in s.lower()) or (not s) or s in ("-", "nan", "None")


def _parse_range(exp):
    """Parse a reference-range string; returns (lo, hi), or None when unparseable."""
    if not exp or str(exp).lower() in ("nan", "none"):
        return None
    s = str(exp).replace("～", "-").replace("~", "-").replace("—", "-")
    m = re.search(r"(-?\d+(?:\.\d+)?)\s*-\s*(-?\d+(?:\.\d+)?)", s)
    if m:
        return float(m.group(1)), float(m.group(2))
    m = re.search(r"[<＜]\s*(-?\d+(?:\.\d+)?)", s)
    if m:
        return None, float(m.group(1))
    m = re.search(r"[>＞]\s*(-?\d+(?:\.\d+)?)", s)
    if m:
        return float(m.group(1)), None
    return None


class Cohort:
    """Data view of one batch (the same exam cohort)."""

    def __init__(self, batch_dir):
        self.dir = batch_dir
        self.results = json.load(
            open(os.path.join(batch_dir, "batch_results.json"), encoding="utf-8")
        )
        self.by_app = {}
        for p in self.results:
            self.by_app[str(p.get("app_no", "")).lstrip("0")] = p
        self._refs = None
        self._labels = None
        self._dist = None
        self._items = None
        self._base = None

    # ---------- Threshold reference ----------
    @property
    def refs(self):
        """model -> raw gauge_reference (the most frequent non-empty value in that model's CSV)."""
        if self._refs is None:
            refs = {}
            for f in glob.glob(os.path.join(self.dir, "*.csv")):
                model = os.path.basename(f)[:-4]
                try:
                    rows = list(csv.DictReader(open(f, encoding="utf-8-sig")))
                except Exception:
                    continue
                if not rows or "gauge_reference" not in rows[0]:
                    continue
                vals = [
                    str(r.get("gauge_reference") or "").strip() for r in rows
                ]
                vals = [v for v in vals if v and v.lower() not in ("nan", "none", "-")]
                if vals:
                    refs[model] = Counter(vals).most_common(1)[0][0]
            self._refs = refs
        return self._refs

    @property
    def labels(self):
        """model -> display name (English from model_labels.json, CSV fallback)."""
        if self._labels is None:
            labels = {}
            for f in glob.glob(os.path.join(self.dir, "*.csv")):
                model = os.path.basename(f)[:-4]
                try:
                    with open(f, encoding="utf-8-sig") as fh:
                        head = fh.readline().strip().split(",")
                        if "model_name" not in head:
                            continue
                        idx = head.index("model_name")
                        row = fh.readline().strip().split(",")
                    if len(row) > idx and row[idx].strip():
                        labels[model] = row[idx].strip()
                except Exception:
                    continue
            # Prefer English labels: model_labels.json overrides Chinese model names in the CSV
            extra = os.path.join(os.path.dirname(__file__), "model_labels.json")
            if os.path.exists(extra):
                for k, v in json.load(open(extra, encoding="utf-8")).items():
                    labels[k] = v
            self._labels = labels
        return self._labels

    # ---------- Population distribution ----------
    @property
    def dist(self):
        """model -> {'scores': [...], 'levels': [...]}, used for percentile fallback."""
        if self._dist is None:
            d = defaultdict(lambda: {"scores": [], "levels": []})
            for p in self.results:
                for a in p.get("computable", []):
                    for m in a.get("model_results", []):
                        s = _num(m.get("score"))
                        if s is None:
                            continue
                        d[m["model"]]["scores"].append(s)
                        d[m["model"]]["levels"].append(level_rank(m.get("risk_level")))
            self._dist = dict(d)
        return self._dist

    # ---------- Lab items ----------
    @property
    def items(self):
        if self._items is None:
            path = os.path.join(self.dir, "item_result.csv")
            rows = []
            if os.path.exists(path):
                rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
            self._items = rows
        return self._items

    @property
    def base(self):
        if self._base is None:
            path = os.path.join(self.dir, "base.csv")
            rows = []
            if os.path.exists(path):
                rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
            self._base = {str(r.get("app_no", "")).lstrip("0"): r for r in rows}
        return self._base

    # ---------- Patients ----------
    def patient(self, app_no):
        key = str(app_no).lstrip("0")
        if key not in self.by_app:
            raise KeyError(f"app_no={app_no} not found in this batch")
        return Patient(self, key, self.by_app[key])


# Cover metric priority (match when the name contains the keyword)
METRIC_PRIORITY = [
    ("HbA1c", "HbA1c", "%"),
    ("Uric acid", "Uric acid", "umol/L"),
    ("ALT", "ALT", "U/L"),
    ("Triglycerides", "Triglycerides", "mmol/L"),
    ("Fasting glucose", "Fasting glucose", "mmol/L"),
    ("LDL-C", "LDL-C", "mmol/L"),
    ("Homocysteine", "Homocysteine", "umol/L"),
    ("Creatinine", "Creatinine", "umol/L"),
    ("Total cholesterol", "Total cholesterol", "mmol/L"),
    ("AST", "AST", "U/L"),
    # English lab names (English snapshot)
    ("hba1c", "HbA1c", "%"),
    ("glycated hemoglobin", "HbA1c", "%"),
    ("uric acid", "Uric acid", "μmol/L"),
    ("alanine aminotransferase", "ALT", "U/L"),
    ("triglyceride", "Triglycerides", "mmol/L"),
    ("fasting glucose", "Fasting glucose", "mmol/L"),
    ("glucose", "Fasting glucose", "mmol/L"),
    ("ldl", "LDL-C", "mmol/L"),
    ("homocysteine", "Homocysteine", "μmol/L"),
    ("creatinine", "Creatinine", "μmol/L"),
    ("total cholesterol", "Total cholesterol", "mmol/L"),
    ("aspartate aminotransferase", "AST", "U/L"),
]


class Patient:
    def __init__(self, cohort: Cohort, key, raw):
        self.cohort = cohort
        self.key = key
        self.raw = raw
        self.app_no = str(raw.get("app_no", "")).zfill(8)
        self.diseases = self._build_diseases()
        self.demo = self._build_demo()
        self.metrics = self._build_metrics()

    # ---- Diseases and models ----
    def _build_diseases(self):
        out = []
        for a in self.raw.get("computable", []):
            models = []
            for m in a.get("model_results", []):
                models.append(
                    {
                        "model": m["model"],
                        "label": _clean_label(
                            m.get("model_name")
                            or self.cohort.labels.get(m["model"])
                            or m["model"]
                        ),
                        "score": m.get("score"),
                        "unit": _unit_of(m.get("score")),
                        "level": LEVEL_NAME.get(level_rank(m.get("risk_level")),
                                                m.get("risk_level") or ""),
                        "rank": level_rank(m.get("risk_level")),
                        "note": m.get("note") or m.get("assessment_note") or "",
                    }
                )
            # Continuous models: infer the tier from same-batch percentiles (use percentiles when no threshold exists)
            for m in models:
                if m["rank"] == 0 and is_continuous(m["level"]):
                    m["rank"] = self._percentile_rank(m["model"], m["score"])
                    if m["rank"]:
                        m["level"] = LEVEL_NAME[m["rank"]]
                        m["derived"] = True
            # Disease tier = highest risk among its models (avoids the contradictory "both high and low" look)
            top = max([m["rank"] for m in models], default=0)
            out.append(
                {
                    "name": a.get("english_name") or a.get("disease_name") or a.get("disease") or "Unnamed",
                    "level": LEVEL_NAME[top],
                    "rank": top,
                    "models": models,
                    "raw_level": a.get("risk_level") or "",
                }
            )
        return out

    def _percentile_rank(self, model, score):
        """Convert continuous scores to risk tiers by same-batch distribution (P80/P50 cut points)."""
        d = self.cohort.dist.get(model)
        if not d or len(d["scores"]) < 12:
            return 0
        v = _num(score)
        if v is None:
            return 0
        s = sorted(d["scores"])
        n = len(s)
        below = sum(1 for x in s if x < v)
        p = below / n
        if p >= 0.8:
            return 3
        if p >= 0.5:
            return 2
        return 1

    # ---- Demographics ----
    def _build_demo(self):
        b = self.cohort.base.get(self.key, {})
        age = _num(b.get("app_pat_age"))
        sex = str(b.get("pat_sex") or "").strip()
        sex_txt = {"1": "Male", "2": "Female", "Male": "Male", "Female": "Female"}.get(sex, "")
        d = {
            "name": b.get("pat_name") or "",
            "age": int(age) if age else None,
            "sex": sex_txt,
            "height": None,
            "weight": None,
            "waist": None,
            "sbp": None,
            "dbp": None,
            "bmi": None,
        }
        for it in self.cohort.items:
            if str(it.get("app_no", "")).lstrip("0") != self.key:
                continue
            name = (it.get("res_itm_name") or "").strip()
            val = _num(it.get("res_value"))
            if name in ("Height",) and d["height"] is None:
                d["height"] = val
            elif name in ("Weight",) and d["weight"] is None:
                d["weight"] = val
            elif name in ("Waist",) and d["waist"] is None:
                d["waist"] = val
            elif name in ("Blood pressure", "BP") and d["sbp"] is None:
                m = re.match(r"\s*(\d+)\s*/\s*(\d+)", str(it.get("res_value") or ""))
                if m:
                    d["sbp"], d["dbp"] = int(m.group(1)), int(m.group(2))
        if d["height"] and d["weight"]:
            d["bmi"] = round(d["weight"] / (d["height"] / 100) ** 2, 1)
        return d

    # ---- Cover metrics ----
    def _build_metrics(self):
        rows = [
            it
            for it in self.cohort.items
            if str(it.get("app_no", "")).lstrip("0") == self.key
        ]
        picked, used = [], set()
        for needle, label, unit in METRIC_PRIORITY:
            for it in rows:
                name = (it.get("res_itm_name") or "").strip()
                if needle not in name or name in used:
                    continue
                val = _num(it.get("res_value"))
                if val is None:
                    continue
                used.add(name)
                lo, hi = _parse_range(it.get("res_exp")) or (None, None)
                tone, note = "ok", ""
                if hi is not None and val > hi:
                    tone = "bad"
                    note = f"↑ above upper limit {hi:g}"
                    if hi and val / hi >= 1.5:
                        note += f" (approx. {val / hi:.1f}x)"
                elif lo is not None and val < lo:
                    tone = "warn"
                    note = f"↓ below lower limit {lo:g}"
                picked.append(
                    {
                        "label": label,
                        "value": f"{val:g}",
                        "unit": it.get("res_unit") or unit,
                        "note": note,
                        "tone": tone,
                    }
                )
                break
            if len(picked) >= 4:
                break
        # Pad with BMI when fewer than 4 items
        if len(picked) < 4 and self.demo.get("bmi"):
            bmi = self.demo["bmi"]
            tone = "bad" if bmi >= 28 else ("warn" if bmi >= 24 else "ok")
            picked.insert(
                0,
                {
                    "label": "BMI",
                    "value": f"{bmi:g}",
                    "unit": "kg/m²",
                    "note": "Obese" if bmi >= 28 else ("Overweight" if bmi >= 24 else "Normal"),
                    "tone": tone,
                },
            )
        return picked[:4]


# ---------------------------------------------------------------- Threshold parsing
def categorical_bands(rank):
    """Three-band ticks (low/medium/high) when a model gives only tiers without continuous thresholds."""
    bands = [
        {"lo": 0.0, "hi": 1.0, "label": "Low"},
        {"lo": 1.0, "hi": 2.0, "label": "Medium"},
        {"lo": 2.0, "hi": 3.0, "label": "High"},
    ]
    return bands


def resolve_bands(model, cohort: Cohort, score=None, rank=0):
    """Returns (bands, source, note).

    source:
      * ``"model"``  - the model's own gauge_reference
      * ``"cohort"`` - inferred from the same-batch population distribution
      * ``None``     - cannot be tiered
    """
    from . import gauge

    raw = cohort.refs.get(model)
    if raw:
        parsed = gauge.parse_bands(raw)
        if parsed and len(parsed["bands"]) >= 2:
            return parsed["bands"], "model", raw
    bands = percentile_bands(model, cohort, score)
    if bands:
        return bands, "cohort", ""
    if rank:
        return categorical_bands(rank), "level", ""
    return None, None, ""


def percentile_bands(model, cohort: Cohort, score=None):
    """Infer tier thresholds from the same-batch distribution.

    Prefer cut points from the model's own risk-level score distribution in the population (consistent with model conclusions);
    If the population has only one tier, degrade to tertiles (P33/P67).
    """
    d = cohort.dist.get(model)
    if not d or len(d["scores"]) < 12:
        return None
    scores, levels = d["scores"], d["levels"]
    uniq = sorted(set(levels))
    if len(uniq) >= 2:
        med = {}
        for lv in uniq:
            vals = [s for s, l in zip(scores, levels) if l == lv]
            if vals:
                med[lv] = statistics.median(vals)
        keys = sorted(med)
        cuts = []
        ok = True
        for a, b in zip(keys, keys[1:]):
            if med[b] <= med[a]:
                ok = False
                break
            cuts.append((med[a] + med[b]) / 2)
        if ok and cuts:
            edges = [min(scores)] + cuts + [max(scores)]
            return _bands_from_edges(edges, len(uniq))
    # Degenerate case: tertiles
    s = sorted(scores)
    n = len(s)
    q1, q2 = s[int(n * 0.33)], s[int(n * 0.67)]
    if q1 >= q2:
        return None
    return _bands_from_edges([s[0], q1, q2, s[-1]], 3)


def _bands_from_edges(edges, n):
    bands = []
    for i in range(len(edges) - 1):
        bands.append({"lo": edges[i], "hi": edges[i + 1], "label": ""})
    return bands
