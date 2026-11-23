# -*- coding: utf-8 -*-
"""
json-render component catalog definitions
================================================================================
The json-render spec is a flat tree:

    {
      "root": "page",
      "elements": {
        "page":  {"type": "Page",  "props": {...}, "children": ["sec-1", ...]},
        "sec-1": {"type": "Section", "props": {...}, "children": [...]},
        ...
      }
    }

This module serves both:
  1. Step-2 conversion prompt (describes the catalog to the LLM);
  2. Backend spec validation (types / child references);
  3. Front-end renderer (component registry in web.html).
================================================================================
"""

# Component definitions: type -> {props: {name: type description}, slots: [...]}
CATALOG = {
    "Page": {
        "desc": "report root container, unique per page",
        "props": {
            "title": "report title",
            "subtitle": "subtitle (e.g. Primary Prevention and Health Management)",
            "patient": "patient summary (name/sex/age/exam number)",
            "generatedAt": "generation time",
        },
        "slots": ["default"],
    },
    "Section": {
        "desc": "top-level chapter (1, 2, 3 ...)",
        "props": {
            "title": "chapter title",
            "icon": "icon name (see icon list; optional)",
            "level": "level, 1 = top-level chapter",
        },
        "slots": ["default"],
    },
    "SubSection": {
        "desc": "sub-section (e.g. a specific disease)",
        "props": {"title": "sub-section title", "icon": "icon name (optional)"},
        "slots": ["default"],
    },
    "Paragraph": {
        "desc": "body paragraph; supports **bold** and quotes",
        "props": {"text": "paragraph text"},
    },
    "RiskDashboard": {
        "desc": "risk dashboard: all model scores arranged by risk tier, high risk most prominent, low risk de-emphasized",
        "props": {
            "title": "panel title",
            "items": "array; each item {disease, tier, tierLabel, score, scoreLabel, icon, color}",
        },
    },
    "Gauge": {
        "desc": "single gauge (usually rendered inside RiskDashboard; can be used alone)",
        "props": {
            "label": "indicator name",
            "value": "value (0-100)",
            "max": "maximum, default 100",
            "tier": "risk tier 1-4",
            "color": "red/orange/green/blue",
            "size": "lg/md/sm",
        },
    },
    "Metric": {
        "desc": "key-metric card (value + unit + note)",
        "props": {
            "label": "indicator name",
            "value": "value",
            "unit": "unit (optional)",
            "hint": "supplementary note (optional)",
            "tier": "risk tier 1-4 (optional)",
        },
    },
    "RiskBadge": {
        "desc": "risk chip (high/medium/low)",
        "props": {"level": "high/medium/low/very high risk", "text": "display text (optional)"},
    },
    "Callout": {
        "desc": "callout box",
        "props": {
            "type": "info/warning/danger/success",
            "title": "title (optional)",
            "text": "content",
        },
    },
    "List": {
        "desc": "list",
        "props": {"items": "string array", "ordered": "whether ordered (bool)"},
    },
    "Table": {
        "desc": "table",
        "props": {"headers": "header string array", "rows": "2-D string array"},
    },
    "KeyValue": {
        "desc": "key-value list (e.g. basic information)",
        "props": {"items": "array; each item {key, value}"},
    },
    "Timeline": {
        "desc": "timeline (follow-up / action plan)",
        "props": {"items": "array; each item {time, title, text}"},
    },
    "ActionItem": {
        "desc": "action-item card",
        "props": {
            "priority": "priority: high/medium/low",
            "title": "action title",
            "detail": "concrete steps",
            "tags": "tag string array (optional)",
        },
    },
    "Evidence": {
        "desc": "evidence citation line (labels the knowledge source)",
        "props": {"text": "evidence content", "source": "source (corpus relation / general knowledge supplement)"},
    },
    "Icon": {
        "desc": "icon",
        "props": {"name": "icon name", "color": "red/orange/green/blue", "size": "lg/md/sm"},
    },
    "Divider": {"desc": "divider", "props": {}},
    "Columns": {
        "desc": "multi-column layout container",
        "props": {"ratio": "column width ratio, e.g. '1:1' or '2:1'"},
        "slots": ["default"],
    },
    "Card": {
        "desc": "generic card container",
        "props": {"title": "card title", "icon": "icon name (optional)"},
        "slots": ["default"],
    },
}


def describe_for_prompt() -> str:
    """Render the catalog into a compact description for the LLM."""
    lines = []
    for name, meta in CATALOG.items():
        props = "; ".join(f"{k}: {v}" for k, v in meta["props"].items()) or "none"
        slots = ", may contain children" if meta.get("slots") else ""
        lines.append(f"- {name}（{meta['desc']}）props: {props}{slots}")
    return "\n".join(lines)


def validate_spec(spec) -> list:
    """Validate the spec structure; returns a list of errors (empty means pass)."""
    errors = []
    if not isinstance(spec, dict):
        return ["spec is not a JSON object"]
    root = spec.get("root")
    elements = spec.get("elements")
    if not isinstance(elements, dict) or not elements:
        return ["missing elements object"]
    if not root or root not in elements:
        errors.append(f"root points to a non-existent element: {root!r}")

    for eid, el in elements.items():
        if not isinstance(el, dict):
            errors.append(f"element {eid} is not an object")
            continue
        etype = el.get("type")
        if etype not in CATALOG:
            errors.append(f"element {eid} uses an undefined component type: {etype!r}")
        children = el.get("children")
        if children is not None:
            if not isinstance(children, list):
                errors.append(f"element {eid} children is not an array")
            else:
                for c in children:
                    if c not in elements:
                        errors.append(f"element {eid} references a non-existent child: {c!r}")
    return errors
