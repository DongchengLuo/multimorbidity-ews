"""2010 Caprini Risk Assessment Model for peri-operative VTE risk."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd


MODEL_NAME = "Caprini 2010 Venous Thromboembolism Risk Assessment Model"
REFERENCE = "Caprini JA. Am J Surg. 2010;199(1 Suppl):S3-S10."
GAUGE_REFERENCE = "0-2|3-4|5-6|7-8|9+"

FIELD_MAPPING = {
    "age": ["app_pat_age", "age"], "gender": ["pat_sex", "sex", "gender"],
    "BMI": ["BMI", "bmi"], "height": ["height"], "weight": ["weight"],
    "operation_minutes": ["estimated_surgery_duration (min)", "surgery_duration", "operation_minutes", "operative_time"],
    "major_surgery": ["planned_major_surgery", "major_surgery"],
    "minor_surgery": ["planned_minor_surgery", "minor_surgery"], "prior_major_surgery": ["prior_major_surgery"],
    "varicose_veins": ["varicose_veins"], "ibd_history": ["inflammatory_bowel_disease_history", "ibd_history"],
    "swollen_legs": ["current_leg_swelling", "swollen_legs"], "acute_mi": ["acute_MI_within_1_month", "acute_mi"],
    "chf": ["CHF_within_1_month", "chf"], "sepsis": ["sepsis_within_1_month", "sepsis"],
    "serious_lung_disease": ["severe_lung_disease_or_pneumonia_within_1_month", "serious_lung_disease"],
    "copd": ["chronic_obstructive_pulmonary_disease", "abnormal_pulmonary_function", "copd"], "bed_rest": ["medical_patient_bedridden", "bed_rest", "immobilized"],
    "leg_cast_brace": ["leg_cast_or_brace", "leg_cast_brace"], "central_venous_access": ["central_venous_access", "cvc"],
    "recent_transfusion": ["transfusion_within_1_month", "recent_transfusion"], "other_risk_factor": ["other_Caprini_risk_factors", "other_risk_factor"],
    "hormone_therapy": ["oral_contraceptive_or_HRT", "hormone_therapy"],
    "pregnancy_postpartum": ["pregnancy_or_postpartum_within_1_month", "pregnancy", "pregnancy_postpartum"],
    "adverse_pregnancy_history": ["adverse_pregnancy_history"],
    "arthroscopic_surgery": ["arthroscopic_surgery"], "laparoscopic_surgery": ["laparoscopic_surgery"],
    "previous_malignancy": ["prior_malignancy", "previous_malignancy"], "active_cancer": ["active_cancer_or_chemotherapy", "malignancy", "active_cancer"],
    "vte_history": ["superficial_thrombophlebitis_or_DVT_or_PE_history", "VTE_history", "vte_history"],
    "family_vte_history": ["DVT_or_PE_family_history", "family_vte_history"],
    "factor_v_leiden": ["Factor_V_Leiden_positive", "factor_v_leiden"], "prothrombin_20210a": ["prothrombin_20210A_positive", "prothrombin_20210a"],
    "hyperhomocysteinemia": ["elevated_homocysteine", "hyperhomocysteinemia"],
    "lupus_anticoagulant": ["lupus_anticoagulant_positive", "lupus_anticoagulant"],
    "anticardiolipin": ["elevated_anticardiolipin_antibody", "anticardiolipin"], "hit": ["heparin_induced_thrombocytopenia", "HIT", "hit"],
    "other_thrombophilia": ["other_thrombophilia"],
    "lower_extremity_arthroplasty": ["elective_major_lower_limb_joint_replacement", "lower_extremity_arthroplasty"],
    "recent_hip_pelvis_leg_fracture": ["hip_pelvis_or_lower_limb_fracture_within_1_month", "recent_hip_pelvis_leg_fracture"],
    "recent_stroke": ["stroke_within_1_month", "stroke", "recent_stroke"], "recent_multiple_trauma": ["multiple_trauma_within_1_month", "major_trauma", "recent_multiple_trauma"],
    "recent_spinal_cord_injury": ["acute_spinal_cord_injury_or_paralysis_within_1_month", "spinal_injury", "recent_spinal_cord_injury"],
}

ONE_POINT = ("prior_major_surgery", "varicose_veins", "ibd_history", "swollen_legs", "acute_mi", "chf", "sepsis", "serious_lung_disease", "copd", "bed_rest", "leg_cast_brace", "central_venous_access", "recent_transfusion", "other_risk_factor")
WOMEN_ONE_POINT = ("hormone_therapy", "pregnancy_postpartum", "adverse_pregnancy_history")
THREE_POINT = ("vte_history", "family_vte_history", "factor_v_leiden", "prothrombin_20210a", "hyperhomocysteinemia", "lupus_anticoagulant", "anticardiolipin", "hit", "other_thrombophilia")
FIVE_POINT = ("lower_extremity_arthroplasty", "recent_hip_pelvis_leg_fracture", "recent_stroke", "recent_multiple_trauma", "recent_spinal_cord_injury")


def _norm(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return re.sub(r"[\s_()（）\[\]{}]+", "", str(value)).lower()


def _number(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", str(value).replace(",", ""))
    return float(match.group()) if match else None


def _truth(value: Any) -> bool:
    text = _norm(value)
    if not text:
        return False
    if text in {"0", "no", "absent", "false", "negative"} or any(x in text for x in ("denies", "not seen", "none", "no disease")):
        return False
    return text in {"1", "yes", "present", "true", "positive"} or any(x in text for x in ("confirmed", "medical_history", "ongoing", "current"))


def _is_female(value: Any) -> bool:
    text = _norm(value)
    return text in {"female", "f", "2"} or text.startswith("female")


def _unit(value: Any) -> str:
    return str(value or "").strip().lower().replace(" ", "")


def _height_m(value: Any, unit: Any) -> float | None:
    number, source = _number(value), _unit(unit)
    if number is None:
        return None
    if source in {"m"}:
        return number
    if source in {"cm"}:
        return number / 100.0
    if source in {"mm"}:
        return number / 1000.0
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


def calculate_caprini_2010(*, age: float | None, female: bool, bmi: float | None, operation_minutes: float | None, flags: dict[str, bool]) -> tuple[int, list[str]]:
    score, detail = 0, []
    if age is not None:
        points = 1 if 40 <= age < 60 else 2 if 60 <= age < 75 else 3 if age >= 75 else 0
        if points:
            score += points; detail.append(f"age:{points}")
    if bmi is not None:
        points = 1 if 30 <= bmi < 40 else 2 if 40 <= bmi < 50 else 3 if bmi >= 50 else 0
        if points:
            score += points; detail.append(f"BMI:{points}")

    surgery_points = 0
    if operation_minutes is not None:
        surgery_points = 1 if operation_minutes < 60 else 2 if operation_minutes < 120 else 3 if operation_minutes < 180 else 5
    elif flags.get("major_surgery"):
        surgery_points = 2
    elif flags.get("minor_surgery"):
        surgery_points = 1
    if (flags.get("arthroscopic_surgery") or flags.get("laparoscopic_surgery")) and (operation_minutes is None or operation_minutes >= 60):
        surgery_points = max(surgery_points, 2)
    if surgery_points:
        score += surgery_points; detail.append(f"surgery:{surgery_points}")

    for key in ONE_POINT:
        if flags.get(key):
            score += 1; detail.append(f"{key}:1")
    if female:
        for key in WOMEN_ONE_POINT:
            if flags.get(key):
                score += 1; detail.append(f"{key}:1")

    cancer_points = 3 if flags.get("active_cancer") else 2 if flags.get("previous_malignancy") else 0
    if cancer_points:
        score += cancer_points; detail.append(f"malignancy:{cancer_points}")
    for key in THREE_POINT:
        if flags.get(key):
            score += 3; detail.append(f"{key}:3")
    for key in FIVE_POINT:
        if flags.get(key):
            score += 5; detail.append(f"{key}:5")
    return score, detail


def risk_level(score: int) -> str:
    if score <= 2:
        return "low risk (0-2)"
    if score <= 4:
        return "medium risk (3-4)"
    if score <= 6:
        return "high risk (5-6)"
    if score <= 8:
        return "very high risk (7-8)"
    return "extremely high risk (>8)"


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
        female = _is_female(row.get("pat_sex") or data.get("gender", ("", ""))[0])
        bmi = _number(data.get("BMI", (None, ""))[0])
        unit_note = ""
        if bmi is None:
            height = _height_m(*data.get("height", (None, "")))
            weight = _weight_kg(*data.get("weight", (None, "")))
            if height and weight:
                bmi = weight / height ** 2
                unit_note = "bmi_from_explicit_units"
        operation_minutes = _number(data.get("operation_minutes", (None, ""))[0])
        flags = {key: _truth(value[0]) for key, value in data.items() if key not in {"age", "gender", "BMI", "height", "weight", "operation_minutes"}}
        score, detail = calculate_caprini_2010(age=age, female=female, bmi=bmi, operation_minutes=operation_minutes, flags=flags)
        output.append({
            "pat_id": person[1], "app_no": person[0], "pat_name": row.get("pat_name", ""),
            "model_name": MODEL_NAME, "predicted_score": score, "risk_level": risk_level(score),
            "code": "I82", "gauge_reference": GAUGE_REFERENCE, "scoring_details": "；".join(detail),
            "unit handling": unit_note, "calculation_status": "computed", "not_computed_reason": "", "calculation_basis": REFERENCE,
        })
    pd.DataFrame(output).to_csv(output_path, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    process_data("base.csv", "item_result.csv", "caprini_vte_assessment_result.csv")
