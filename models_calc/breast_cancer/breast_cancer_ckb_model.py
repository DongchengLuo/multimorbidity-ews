"""China Kadoorie Biobank (CKB) 10-year breast-cancer absolute risk."""

from __future__ import annotations

import math
import re
from typing import Any

import pandas as pd


MODEL_NAME = "China Kadoorie Biobank (CKB) 10-Year Absolute Breast Cancer Risk Model"
REFERENCE = "Han Y, et al. Breast Cancer Res. 2021;23:62."

FIELD_MAPPING = {
    "age": ["app_pat_age", "age"],
    "gender": ["pat_sex", "sex", "gender"],
    "BMI": ["BMI", "bmi"],
    "height": ["height"],
    "weight": ["weight"],
    "residential_area": ["residential_area", "urban_rural", "residence"],
    "education": ["highest_education", "education_level", "education"],
    "family_cancer_history": ["first_degree_cancer_family_count", "first_degree_cancer_history", "family_cancer_history"],
    "parity": ["live_births", "live_birth_count", "parity"],
    "age_at_menarche": ["menarche_age", "age_at_menarche"],
}

# National urban/rural breast-cancer incidence (h1) and non-breast-cancer
# mortality (h2), per 100,000 person-years, from Additional file 2.
AGE_BANDS = list(range(30, 90, 5))
RATES = {
    "urban": {
        "h1": [16.7, 34.2, 64.8, 85.7, 105.2, 107.7, 108.6, 98.7, 92.5, 83.4, 74.9, 47.8],
        "h2": [35.5, 43.2, 80.2, 101.1, 225.6, 275.9, 606.4, 1105.4, 1843.2, 3482.2, 6984.1, 16282.3],
    },
    "rural": {
        "h1": [13.2, 27.1, 54.1, 70.8, 71.5, 73.6, 73.2, 52.3, 46.0, 39.0, 30.0, 22.1],
        "h2": [52.2, 62.1, 96.8, 134.7, 268.5, 344.3, 683.5, 1285.2, 2225.7, 3821.9, 6776.3, 14950.6],
    },
}

PAR = {("urban", False): 0.74, ("urban", True): 0.76, ("rural", False): 0.63, ("rural", True): 0.65}


def _norm(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return re.sub(r"[\s_()（）\[\]{}]+", "", str(value)).lower()


def _number(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", str(value).replace(",", ""))
    return float(match.group()) if match else None


def _unit(value: Any) -> str:
    return str(value or "").strip().lower().replace(" ", "").replace("μ", "u").replace("µ", "u")


def _height_cm(value: Any, unit: Any) -> float | None:
    number, source = _number(value), _unit(unit)
    if number is None:
        return None
    if source in {"cm"}:
        return number
    if source in {"m"}:
        return number * 100.0
    if source in {"mm"}:
        return number / 10.0
    return None


def _weight_kg(value: Any, unit: Any) -> float | None:
    number, source = _number(value), _unit(unit)
    if number is None:
        return None
    if source in {"kg"}:
        return number
    if source in {"g"}:
        return number / 1000.0
    return None


def _residence(value: Any) -> str:
    text = _norm(value)
    return "rural" if any(token in text for token in ("rural", "rural", "rural", "county", "township", "village")) else "urban"


def _education_rr(value: Any) -> float:
    text = _norm(value)
    if any(x in text for x in ("university", "college", "bachelor", "associate", "postgraduate", "college", "university")):
        return 1.87
    if any(x in text for x in ("high school", "vocational_school", "highschool")):
        return 1.60
    if any(x in text for x in ("junior_high", "middleschool")):
        return 1.37
    if any(x in text for x in ("primary_school", "primary")):
        return 1.17
    return 1.00


def _bmi_rr(bmi: float, age: float) -> float:
    index = 0 if bmi < 18.5 else 1 if bmi < 24 else 2 if bmi < 28 else 3
    return ([1.00, 0.95, 0.90, 0.85] if age < 50 else [1.00, 1.25, 1.57, 1.96])[index]


def _height_rr(height_cm: float) -> float:
    if height_cm < 150.2:
        return 1.00
    if height_cm < 154.2:
        return 1.13
    if height_cm < 158.2:
        return 1.28
    return 1.45


def _family_rr(count: float) -> float:
    return 1.00 if count < 1 else 1.10 if count < 2 else 1.57


def _parity_rr(value: float) -> float:
    if value <= 0:
        return 1.78
    if value < 2:
        return 1.66
    if value < 3:
        return 1.41
    return 1.00


def _menarche_rr(age: float) -> float:
    if age <= 12:
        return 1.52
    if age <= 14:
        return 1.32
    if age <= 16:
        return 1.15
    return 1.00


def relative_risk(*, age: float, bmi: float, height_cm: float, education: Any, family_count: float, parity: float, menarche_age: float) -> float:
    return (
        _education_rr(education) * _bmi_rr(bmi, age) * _height_rr(height_cm)
        * _family_rr(family_count) * _parity_rr(parity) * _menarche_rr(menarche_age)
    )


def _band_index(age: float) -> int:
    if age < 30:
        return 0
    return min(len(AGE_BANDS) - 1, int((age - 30) // 5))


def calculate_ckb_10y_risk(
    *, age: float, residence: str, bmi: float, height_cm: float, education: Any,
    family_count: float, parity: float, menarche_age: float, horizon: float = 10.0,
) -> tuple[float, float]:
    """Return (10-year absolute risk, RR at starting age)."""
    residence = "rural" if residence == "rural" else "urban"
    current, end = float(age), float(age) + float(horizon)
    survival, absolute_risk = 1.0, 0.0
    starting_rr = relative_risk(age=current, bmi=bmi, height_cm=height_cm, education=education, family_count=family_count, parity=parity, menarche_age=menarche_age)
    while current < end - 1e-12:
        index = _band_index(current)
        band_end = AGE_BANDS[index] + 5 if current < 90 else end
        segment_end = min(end, band_end, 50.0 if current < 50.0 else end)
        if segment_end <= current:
            segment_end = end
        dt = segment_end - current
        rr = relative_risk(age=current, bmi=bmi, height_cm=height_cm, education=education, family_count=family_count, parity=parity, menarche_age=menarche_age)
        baseline = RATES[residence]["h1"][index] / 100000.0 * (1.0 - PAR[(residence, current >= 50.0)])
        cancer_hazard = baseline * rr
        competing_hazard = RATES[residence]["h2"][index] / 100000.0
        total_hazard = cancer_hazard + competing_hazard
        event = cancer_hazard / total_hazard * (1.0 - math.exp(-total_hazard * dt)) if total_hazard else 0.0
        absolute_risk += survival * event
        survival *= math.exp(-total_hazard * dt)
        current = segment_end
    return absolute_risk, starting_rr


def _read_csv(path: str) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "gb18030", "gbk", "utf-8"):
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False, dtype=str)
        except Exception:
            continue
    return pd.DataFrame()


def _match_field(name: Any) -> str | None:
    wanted = _norm(name)
    for key, aliases in FIELD_MAPPING.items():
        if any(_norm(alias) == wanted for alias in aliases):
            return key
    return None


def process_data(base_path: str, item_path: str, output_path: str) -> None:
    base, items = _read_csv(base_path), _read_csv(item_path)
    if base.empty:
        return
    for frame in (base, items):
        if not frame.empty:
            frame["app_no"] = frame.get("app_no", "").astype(str).str.strip()
            frame["pat_id"] = frame.get("pat_id", "").astype(str).str.strip()
    grouped: dict[tuple[str, str], dict[str, tuple[Any, Any]]] = {}
    if not items.empty:
        for _, row in items.iterrows():
            key = _match_field(row.get("res_itm_name"))
            if key:
                person = (str(row.get("app_no", "")).strip(), str(row.get("pat_id", "")).strip())
                grouped.setdefault(person, {}).setdefault(key, (row.get("res_value"), row.get("res_unit", "")))

    output = []
    for _, row in base.iterrows():
        person = (str(row.get("app_no", "")).strip(), str(row.get("pat_id", "")).strip())
        data = grouped.get(person, {})
        age = _number(row.get("app_pat_age")) or _number(data.get("age", (None, ""))[0])
        residence = _residence(data.get("residential_area", (row.get("pat_address", ""), ""))[0])
        height = _height_cm(*data.get("height", (None, "")))
        weight = _weight_kg(*data.get("weight", (None, "")))
        bmi = _number(data.get("BMI", (None, ""))[0])
        notes = []
        if bmi is None and height and weight:
            bmi = weight / (height / 100.0) ** 2
            notes.append("bmi_from_explicit_units")
        if bmi is None:
            bmi = 18.0
            notes.append("BMI missing; using the model reference group for compatibility")
        if height is None:
            height = 149.0
            notes.append("height missing; using the model reference group for compatibility")
        family_count = _number(data.get("family_cancer_history", (0, ""))[0]) or 0.0
        parity = _number(data.get("parity", (3, ""))[0])
        parity = 3.0 if parity is None else parity
        menarche = _number(data.get("age_at_menarche", (17, ""))[0])
        menarche = 17.0 if menarche is None else menarche
        education = data.get("education", ("no_formal_education", ""))[0]
        if age is None:
            risk, rr, status, reason = None, None, "not_computed", "missing_age"
        else:
            risk, rr = calculate_ckb_10y_risk(age=age, residence=residence, bmi=bmi, height_cm=height, education=education, family_count=family_count, parity=parity, menarche_age=menarche)
            status, reason = "computed", ""
        output.append({
            "pat_id": person[1], "app_no": person[0], "pat_name": row.get("pat_name", ""),
            "model_name": MODEL_NAME, "age": age if age is not None else "-", "urban_rural": "rural" if residence == "rural" else "urban",
            "relative_risk (RR)": round(rr, 6) if rr is not None else "-",
            "10-year_absolute_breast_cancer_risk (%)": round(risk * 100.0, 6) if risk is not None else "-",
            "risk_level": "continuous risk (no fixed tier threshold in the source)" if risk is not None else "-",
            "code": "C50" if risk is not None else "-", "gauge_reference": "",
            "unit handling": "；".join(notes), "calculation_status": status, "not_computed_reason": reason, "calculation_basis": REFERENCE,
        })
    pd.DataFrame(output).to_csv(output_path, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    process_data("base.csv", "item_result.csv", "breast_cancer_risk_C50_result.csv")
