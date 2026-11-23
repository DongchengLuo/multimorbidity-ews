"""Structured extraction of chest imaging reports V3 (standalone).

Extract the Brock/PanCan inputs directly from chest imaging reports in ``desc_item.csv``
Six pulmonary fields. This file contains the full prompt, request, validation and batch implementation with no external script dependency.
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import re
import sys
from typing import Any, Dict

import pandas as pd
import requests

import llm_config  # noqa: F401  ensure the project .env is loaded (model name / endpoint)


LLM_URL = (
    os.environ.get("MEDICAL_LLM_CHAT_URL")
    or os.environ.get("LOCAL_LLM_API_URL")
    or ""
)
LLM_MODEL = (
    os.environ.get("MEDICAL_LLM_MODEL")
    or os.environ.get("LOCAL_LLM_MODEL_NAME")
    or ""
)
LLM_API_KEY = os.environ.get("MEDICAL_LLM_API_KEY") or os.environ.get("LOCAL_LLM_API_KEY", "")
LLM_TIMEOUT_SECONDS = max(10, int(os.environ.get("MEDICAL_LLM_TIMEOUT_SECONDS", "60")))
DEFAULT_MAX_WORKERS = max(1, int(os.environ.get("MEDICAL_LLM_MAX_WORKERS", "5")))

EXPECTED_FIELDS = (
    "emphysema",
    "nodule_size",
    "nodule_type",
    "nodule_location",
    "nodule_count",
    "spiculation",
)
STATE_VALUES = ("present", "absent", "no_nodule", "not_mentioned", "indeterminate")
LUNG_KEYWORDS = ("chest", "lung", "CT", "DR", "X-ray", "plain film")


def build_prompt(report_text: str) -> str:
    """Build the pulmonary six-field extraction prompt with V3 rules built in."""
    return f"""
# Role
You are a rigorous thoracic radiology data analyst. Extract structured pulmonary nodule and emphysema features from chest imaging reports.

# Output
Output only one valid JSON object; no explanation, Markdown or reasoning.
The JSON must contain exactly these six fields:
"emphysema", "nodule_size", "nodule_type", "nodule_location", "nodule_count", "spiculation".

# General principles
1. Extract only information explicitly stated in the source; do not supplement with age, smoking history, medical knowledge or defaults.
2. When a definite nodule exists, nodule_size, nodule_type, nodule_location and spiculation must describe the same target nodule.
3. The target nodule is the intrapulmonary nodule with the largest long-axis diameter; on a tie take the most completely described, then the first described.
4. If a nodule exists but a field is not described, output "not_mentioned" for that field; if the text is contradictory, suspected or cannot be judged reliably, output "indeterminate".
5. If there is explicitly no pulmonary nodule, or the whole report has no nodule description, output "no_nodule" for all five nodule fields.

# Emphysema (V3 highest priority)
1. Output "present" only when the source directly diagnoses or explicitly describes emphysema. Do not infer from bullae, sparse lung markings or increased lucency alone.
2. Output "absent" only when the same sentence containing "emphysema" explicitly states a negation such as "no", "not seen" or "not found".
3. Generic normal descriptions such as "no obvious abnormality in both lungs", "clear lung fields", "normal lung markings", "normal lucency" or "no abnormal density" do not prove the absence of emphysema; if the report never directly evaluates emphysema, output "not_mentioned".
4. Output "indeterminate" when there are only indirect signs, suspected/possible/considered emphysema, contradictions, or insufficient examination conditions.

# Nodule size
1. Output the maximum long-axis diameter of the target nodule in millimetres, digits only; e.g. 8mm x 5mm -> 8, 0.8cm x 0.5cm -> 8.
2. If a nodule exists but no size is given, output "not_mentioned".

# Nodule type
1. Output only "solid", "ground_glass", "part_solid", "perifissural", "not_mentioned", "indeterminate" or "no_nodule".
2. "part_solid" requires the same nodule to contain both ground-glass and solid components; separate solid and ground-glass nodules must not be merged.
3. "perifissural" applies only to a target nodule explicitly described as perifissural, juxtafissural or intrapulmonary-lymph-node-like; mere proximity to a fissure is not enough.

# Nodule location
Output "upper_lobe" when the target nodule is in any upper lobe; otherwise output the explicit location from the source. If location is not described, output "not_mentioned".

# Nodule count (V3 highest priority)
1. This is the total number of intrapulmonary nodules explicitly described in the report, independent of target-nodule selection.
2. Build an itemised lesion list first, then sum, but do not output the list. E.g. "two, plus another one" = 3; "one in each lung" = 2; multiple sites with one each must be summed.
3. "Calcified nodule" and "nodular calcification" count; pure calcification foci, fibrotic foci, linear opacities, bullae, pleural nodules, lymph nodes and endobronchial high-density shadows do not.
4. Prior nodules explicitly resolved or no longer visible this time do not count.
5. Output "indeterminate" only when the total cannot be enumerated ("multiple", "scattered", "several"); if nodules are definite but there is no count information at all, output "not_mentioned".

# Spiculated margin
Evaluate the target nodule only: output "present" for definite spiculation, "absent" for definite absence, "not_mentioned" when not described, "indeterminate" when contradictory or suspected.

[Source report]
{report_text}
""".strip()


def _response_content(payload: Any) -> str:
    if not isinstance(payload, dict):
        raise ValueError("LLM response is not a JSON object")
    choices = payload.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        raise ValueError("LLM response is missing choices")
    message = choices[0].get("message") or {}
    content = message.get("content", "") if isinstance(message, dict) else ""
    if isinstance(content, list):
        content = "".join(
            str(item.get("text", "")) if isinstance(item, dict) else str(item)
            for item in content
        )
    return str(content).strip()


def _parse_json_object(content: str) -> Dict[str, Any]:
    text = str(content or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("LLM-returned JSON is not an object")
    return value


def extract_with_llm(report_text: str) -> Dict[str, Any]:
    """Call an OpenAI-compatible endpoint to extract the six pulmonary fields. Returns an empty dict on failure."""
    payload = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": build_prompt(str(report_text or ""))}],
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }
    if os.environ.get("LOCAL_LLM_SEND_CHAT_TEMPLATE_KWARGS", "true").strip().lower() not in {
        "0", "false", "no", "off"
    }:
        payload["chat_template_kwargs"] = {"enable_thinking": False}

    headers = {"Content-Type": "application/json"}
    if LLM_API_KEY and LLM_API_KEY.upper() != "EMPTY":
        headers["Authorization"] = f"Bearer {LLM_API_KEY}"

    try:
        response = requests.post(
            LLM_URL,
            json=payload,
            headers=headers,
            timeout=LLM_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        return _parse_json_object(_response_content(response.json()))
    except Exception as exc:
        print(f"  Lung LLM extraction error: {exc}")
        return {}


def _clean_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        value = value[0] if value else ""
    return str(value).strip().strip("\"'").strip()


def normalize_extraction(extracted: Any) -> Dict[str, str]:
    """Correct the field set and common formats so six fields are always output."""
    source = extracted if isinstance(extracted, dict) else {}
    result: Dict[str, str] = {}

    for field in EXPECTED_FIELDS:
        value = _clean_value(source.get(field))
        if not value or value.lower() in {"null", "none", "n/a", "na", "unknown"}:
            value = "not_mentioned"

        if field == "emphysema":
            if value in {"present", "yes", "positive", "definite"}:
                value = "present"
            elif value in {"absent", "no", "negative", "none"}:
                value = "absent"
            if value not in {"present", "absent", "not_mentioned", "indeterminate"}:
                value = "indeterminate"

        elif field == "nodule_size" and value not in {"no_nodule", "not_mentioned", "indeterminate"}:
            number = re.search(r"\d+(?:\.\d+)?", value)
            if not number:
                value = "indeterminate"
            else:
                size = float(number.group())
                if re.search(r"cm", value, flags=re.IGNORECASE):
                    size *= 10
                value = f"{size:g}"

        elif field == "nodule_type" and value not in {"no_nodule", "not_mentioned", "indeterminate"}:
            aliases = {
                "pure ground-glass": "ground_glass",
                "nonsolid": "ground_glass",
                "ground-glass": "ground_glass",
                "mixed ground-glass": "part_solid",
                "part-solid": "part_solid",
                "perifissural": "perifissural",
            }
            lowered = value.lower()
            value = next((target for alias, target in aliases.items() if alias.lower() in lowered), value)
            if value not in {"solid", "ground_glass", "part_solid", "perifissural"}:
                value = "indeterminate"

        elif field == "nodule_count" and value not in {"no_nodule", "not_mentioned", "indeterminate"}:
            number = re.search(r"\d+", value)
            value = number.group() if number else "indeterminate"

        elif field == "spiculation" and value not in {"no_nodule", "not_mentioned", "indeterminate"}:
            if value in {"present", "yes", "positive", "definite"}:
                value = "present"
            elif value in {"absent", "no", "negative", "none"}:
                value = "absent"
            if value not in {"present", "absent"}:
                value = "indeterminate"

        result[field] = value

    nodule_fields = EXPECTED_FIELDS[1:]
    if any(result[field] == "no_nodule" for field in nodule_fields):
        for field in nodule_fields:
            result[field] = "no_nodule"
    return result


def process_single_record(row: pd.Series) -> list[Dict[str, str]]:
    extracted = normalize_extraction(extract_with_llm(str(row.get("res_result", ""))))
    return [
        {
            "app_no": str(row.get("app_no", "")),
            "pat_id": str(row.get("pat_id", "")),
            "res_itm_name": field,
            "res_value": extracted[field],
        }
        for field in EXPECTED_FIELDS
    ]


def run_lung_extraction(input_dir=".", output_dir=".", max_workers=None):
    """Filter chest imaging reports and return a long table appendable to item_result.csv."""
    del output_dir  # kept for compatibility; results are appended by batch_processor.
    workers = max(1, int(max_workers or DEFAULT_MAX_WORKERS))
    input_path = os.path.join(str(input_dir), "desc_item.csv")
    print("--- Starting pulmonary imaging feature extraction V3 (standalone) ---")

    if not os.path.exists(input_path):
        print(f"  {input_path} not found; skipping pulmonary feature extraction")
        return None

    try:
        frame = pd.read_csv(input_path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        frame = pd.read_csv(input_path, encoding="utf-8")

    required = {"app_no", "pat_id", "res_result", "det_com_name"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"desc_item.csv missing fields: {', '.join(missing)}")

    frame = frame.dropna(subset=["res_result"])
    pattern = "|".join(re.escape(keyword) for keyword in LUNG_KEYWORDS)
    filtered = frame[
        frame["det_com_name"].astype(str).str.contains(pattern, na=False, case=False)
    ].copy()
    print(f"  total raw reports: {len(frame)}")
    print(f"  pulmonary-related reports: {len(filtered)}")
    if filtered.empty:
        print("  no pulmonary-related reports; skipping")
        return None

    records: list[Dict[str, str]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(process_single_record, row) for _, row in filtered.iterrows()]
        for future in concurrent.futures.as_completed(futures):
            try:
                records.extend(future.result())
            except Exception as exc:
                print(f"  pulmonary extraction thread error: {exc}")

    if not records:
        print("  No valid data extracted")
        return None
    output = pd.DataFrame(records).sort_values(["app_no", "res_itm_name"]).reset_index(drop=True)
    print(f"  pulmonary feature extraction complete: {len(output)} records")
    return output


if __name__ == "__main__":
    source_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    target_dir = sys.argv[2] if len(sys.argv) > 2 else source_dir
    run_lung_extraction(source_dir, target_dir)
