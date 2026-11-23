"""QDiabetes-2018 A/B/C 10-year type-2 diabetes risk equations.

This is a faithful Python port of the open-source R implementation bundled in
``evaluation_modelx/03_official_compare/sources/qdiabetes2018``.  The
original algorithm is AGPL-3.0 licensed.  ClinRisk's required warning applies:
an inaccurate implementation can assign the wrong risk; users must validate
results against the original source before clinical use.

Project compatibility: missing questionnaire variables retain the upstream
algorithm defaults.  No age, geography or disease-history hard stop is added.
Laboratory variants are selected only when ``res_unit`` supports an explicit
conversion to the unit required by the source equation.
"""

from __future__ import annotations

import math
import re
from typing import Any

import pandas as pd


MODEL_NAME = "QDiabetes-2018 10-Year Type 2 Diabetes Risk Model"
REFERENCE = "Hippisley-Cox J, Coupland C. BMJ. 2017;359:j5019."

FIELD_MAPPING = {
    "age": ["app_pat_age", "age"],
    "gender": ["pat_sex", "sex", "gender"],
    "BMI": ["BMI", "bmi"],
    "height": ["height"],
    "weight": ["weight"],
    "FBG": ["fasting_glucose", "FBG", "FPG", "fpg"],
    "hba1c": ["HbA1c", "hba1c"],
    "ethnicity": ["QDiabetes_ethnicity", "ethnicity", "ethn"],
    "smoke": ["smoking_status", "smoke"],
    "townsend_score": ["townsend_deprivation_index", "Townsend score", "tds"],
    "diabetes_family_history": ["diabetes_family_history", "first_degree_diabetes_history", "fhdm"],
    "has_cvd": ["cardiovascular_history", "prior_cardiovascular_disease", "cvd"],
    "bp_treatment": ["antihypertensive_treatment", "antihypertensive_therapy", "htn"],
    "statin_use": ["statin_use", "on_statin", "stat"],
    "atypical_antipsychotics": ["atypical_antipsychotic", "apsy"],
    "schizophrenia_bipolar": ["schizophrenia_or_bipolar", "psy"],
    "learning_disability": ["learning_disability", "learn"],
    "corticosteroids": ["glucocorticoid_use", "corticosteroid", "ster"],
    "gestational_diabetes": ["gestational_diabetes_history", "gdm"],
    "pcos": ["polycystic_ovary_syndrome", "PCOS", "pcos"],
}

ETHNICITIES = {
    "whitena": "WhiteNA", "white": "WhiteNA", "white": "WhiteNA",
    "indian": "Indian", "indian": "Indian",
    "pakistani": "Pakistani", "pakistani": "Pakistani",
    "bangladeshi": "Bangladeshi", "bangladeshi": "Bangladeshi",
    "otherasian": "OtherAsian", "other_asian": "OtherAsian",
    "blackcaribbean": "BlackCaribbean", "caribbean_black": "BlackCaribbean",
    "blackafrican": "BlackAfrican", "african_black": "BlackAfrican",
    "chinese": "Chinese", "chinese": "Chinese", "Han": "Chinese",
    "other": "Other", "other": "Other",
}

SMOKING = {
    "non": "Non", "never": "Non", "non-smoker": "Non", "never_smoker": "Non", "no": "Non", "0": "Non",
    "ex": "Ex", "former": "Ex", "former_smoker": "Ex", "former_smoker": "Ex",
    "light": "Light", "mild": "Light", "<10": "Light",
    "moderate": "Moderate", "moderate": "Moderate", "10-19": "Moderate",
    "heavy": "Heavy", "severe": "Heavy", ">=20": "Heavy", "≥20": "Heavy",
}

BINARY_KEYS = ("apsy", "ster", "cvd", "gdm", "learn", "psy", "pcos", "stat", "htn", "fhdm")

# Coefficients are copied verbatim from QDR2018A/B/C.R.
COEFFICIENTS = {
    "A": {
        "Female": {
            "eth": [1.0695857881565456, 1.3430172097414006, 1.8029022579794518, 1.127465451770802, 0.42146314902399101, 0.2850919645908353, 0.88151087975891995, 0.36605733431684873],
            "smoke": [0.065601690175059055, 0.28450988673698374, 0.3567664381700702, 0.53595171106787753],
            "age": [4.3400852699139278, -0.0048771702696158879], "bmi": [2.9320361259524925, -0.04740020587484349], "tds": 0.037340569618049151,
            "bin": [0.55267646110984381, 0.26792233680674599, 0.17797229054586691, 1.5248871531467574, 0.27835143587172717, 0.26180852109179059, 0.34061739882066661, 0.65907287732808217, 0.43947582858137119, 0.53133594565587339],
            "int": [-0.80315183983163951, 0.00046840411810210498, -0.86415960028820571, 0.00067249688089533602, -1.9757776696583935, 0.0023750534194347966, 0.65531387575629452, -0.0044719662445263054, -0.036209657201630177, 0.0001185479967753342, -0.26411714505588962, 0.00041610258289047683],
            "survival": 0.986227273941040,
        },
        "Male": {
            "eth": [1.1000230829124793, 1.290384012614721, 1.6740908848727458, 1.1400446789147816, 0.46824681690655806, 0.69905649963015448, 0.68943657127111568, 0.41722228467738209],
            "smoke": [0.16387409105485573, 0.31851449113958979, 0.32207266567783432, 0.45052437163409531],
            "age": [4.4642324388691348, -0.0040750108019255568], "bmi": [0.95129027867120675, -0.14352488277885475], "tds": 0.025918182067678725,
            "bin": [0.42101092346005436, 0.22183580932925384, 0.20269605756290021, 0.0, 0.23315321407986961, 0.22770449520517727, 0.0, 0.58490075431141342, 0.33379392183501078, 0.64799284899369536],
            "int": [-0.94637722268534152, -0.0000519927442172335, -0.93842375526499833, 0.00071026438559688141, -1.7479070653003299, 0.0013508364599531669, 0.45147599241879766, -0.0011797722394560309, -0.10795481262776381, 0.00021471509139319291, -0.60118530429301198, 0.00049141855940878034],
            "survival": 0.978732228279114,
        },
    },
    "B": {
        "Female": {
            "eth": [0.9898906127239111, 1.2511504196326508, 1.493475756819612, 0.96738874345659664, 0.48446445195931781, 0.47842149553601027, 0.75209462708055774, 0.40508807415414244],
            "smoke": [0.037415630723696323, 0.22529736725144828, 0.30997364280236628, 0.43619421394964175],
            "age": [3.765012950751728, -0.0056043343436614941], "bmi": [2.4410935031672469, -0.042152633479909642], "tds": 0.03580462976631265,
            "lab": [-2.1887891946337308, -69.960841982866029],
            "bin": [0.47483785502538534, 0.37679334437547285, 0.19672615680665251, 1.0689325033692647, 0.45422934089510347, 0.16161718890842605, 0.35653657895767171, 0.58092873827186675, 0.28366320201229073, 0.45221497662061116],
            "int": [-0.76835916427865225, 0.00051944556244134762, -0.79831281242975882, 0.00030283275671618906, -1.9033508839833257, 0.0024397111406018711, 0.48447476024049152, -0.0041572976682154057, -0.031939988307181345, 0.00011268821942042522, 2.244290304740435, 0.019934530853431255, 13.006838869978303, -0.071667718752930668, -0.30406273740345013, 0.00045236396712023254],
            "survival": 0.990905702114105,
        },
        "Male": {
            "eth": [1.0081475800686235, 1.3359138425778705, 1.4815419524892652, 1.0384996851820663, 0.52023480708875247, 0.85796734182585588, 0.64131089607656155, 0.48383402208215048],
            "smoke": [0.11194757923641625, 0.31101320954122047, 0.33288984693260421, 0.42570690269419931],
            "age": [4.1149143302364717, -0.0047593576668505362], "bmi": [0.81693615876442971, -0.12502377403433362], "tds": 0.025374175519894356,
            "lab": [-54.841788128097107, -53.11207849848136],
            "bin": [0.44179340888895774, 0.34135473483394541, 0.21589774543727566, 0.0, 0.40128850275853001, 0.21817693913997793, 0.0, 0.51476576001117347, 0.24672092874070373, 0.57494373339875127],
            "int": [-0.95022243138231266, 0.00014729720771628743, -0.83583701630900453, 0.00060129192649664091, -1.814178691926946, 0.0016393484911405418, 0.37484820920783846, -0.0010774782221531713, -0.090983657956248742, 0.00019110487304583101, 21.011730121764334, -0.039004607922383527, 23.824460044746974, -0.041127719805895947, -0.67806477052916658, 0.00062575882488594993],
            "survival": 0.985019445419312,
        },
    },
    "C": {
        "Female": {
            "eth": [0.599095159929154, 0.78320309656353893, 1.1947351247960103, 0.71417446991681433, 0.11953284683887688, 0.013668872878490427, 0.57092265376939455, 0.17091076281069292],
            "smoke": [0.065848258510000673, 0.1458413689734224, 0.15258642474801187, 0.30787416796613976],
            "age": [3.5655214891947722, -0.0056158243572733135], "bmi": [2.5043028874544841, -0.042875801892690461], "tds": 0.035866822056348294,
            "lab": [8.7368031307362184, -0.07823138666994997],
            "bin": [0.54976333110422004, 0.16872205506389704, 0.16443300362739344, 1.125009810517114, 0.28912058310739658, 0.31825122490684077, 0.33806444140981745, 0.45593968473811164, 0.4040022295023758, 0.44280154048260317],
            "int": [-0.81254341971621313, 0.0004665611306005428, -0.90846657652698082, 0.00085189801399280065, -1.8557960585560658, 0.0022627250963352537, 0.6023218765235252, -0.0043386645663133425, -0.03449503839680447, 0.00011627785616712089, 25.441203322736715, -0.052254135588592522, -6.8076080421556107, 0.014054825906114453, -0.27275713515061872, 0.00043545197952207749],
            "survival": 0.988788545131683,
        },
        "Male": {
            "eth": [0.67571207054987803, 0.83147325049663456, 1.0969133802228563, 0.76822446364560482, 0.20897529259108502, 0.38091593781970579, 0.34235836796612695, 0.22046477853433083],
            "smoke": [0.11592891206878651, 0.14624182637633271, 0.10781424112493142, 0.19848629163668474],
            "age": [4.0193435623978031, -0.0048396442306278238], "bmi": [0.81829168905349325, -0.12558808701359642], "tds": 0.025229965184900727,
            "lab": [8.0511642238857934, -0.14652346893914495],
            "bin": [0.45541525220173301, 0.13816187686823922, 0.14546988896239518, 0.0, 0.2596046658040857, 0.28523788490585894, 0.0, 0.42551951901185525, 0.33169430006459311, 0.56612325943680619],
            "int": [-1.0013331909079835, 0.00022455973985742407, -0.89164657372215927, 0.00066044360765696482, -1.7074561167819817, 0.0013873509357389619, 0.45074527472672443, -0.0012224736160287865, -0.10851859809165601, 0.0002266731010346126, 27.670593827146565, -0.05920145812475433, -7.4006134846785434, 0.015592089485149988, -0.61410093887097161, 0.00050602582894772091],
            "survival": 0.981181740760803,
        },
    },
}

ETH_ORDER = ["Indian", "Pakistani", "Bangladeshi", "OtherAsian", "BlackCaribbean", "BlackAfrican", "Chinese", "Other"]
SMOKE_ORDER = ["Ex", "Light", "Moderate", "Heavy"]


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
    return text in {"1", "yes", "present", "true", "positive"} or any(x in text for x in ("confirmed", "medical_history", "currently_taking"))


def _sex(value: Any) -> str:
    text = _norm(value)
    return "Male" if text in {"male", "m", "1"} or text.startswith("male") else "Female"


def _category(value: Any, mapping: dict[str, str], default: str) -> str:
    text = _norm(value)
    if not text:
        return default
    for key, result in mapping.items():
        if _norm(key) == text or _norm(key) in text:
            return result
    return default


def _unit(value: Any) -> str:
    text = str(value or "").strip().lower().replace(" ", "")
    text = text.replace("μ", "u").replace("µ", "u").replace("／", "/")
    return text


def _height_m(value: Any, unit: Any) -> float | None:
    number = _number(value)
    source = _unit(unit)
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
    number = _number(value)
    source = _unit(unit)
    if number is None:
        return None
    if source in {"kg"}:
        return number
    if source in {"g"}:
        return number / 1000.0
    return None


def _glucose_mmol_l(value: Any, unit: Any) -> tuple[float | None, str]:
    number = _number(value)
    source = _unit(unit)
    if number is None:
        return None, ""
    if source in {"mmol/l", "mmoll-1"}:
        return number, "mmol/L (unit consistent)"
    if source in {"mg/dl", "mgdl"}:
        return number / 18.018, "mg/dL ÷ 18.018 → mmol/L"
    return None, f"fasting glucose unit unrecognized: {unit or 'missing'}"


def _hba1c_mmol_mol(value: Any, unit: Any) -> tuple[float | None, str]:
    number = _number(value)
    source = _unit(unit)
    if number is None:
        return None, ""
    if source in {"mmol/mol", "mmolmol"}:
        return number, "mmol/mol (unit consistent)"
    if source in {"%", "percent", "percentage"}:
        return (number - 2.15) * 10.929, "%（NGSP）→ mmol/mol（IFCC）"
    return None, f"HbA1c unit unrecognized: {unit or 'missing'}"


def calculate_qdiabetes_2018(
    variant: str, sex: str, age: float, bmi: float, *, lab: float | None = None,
    ethnicity: str = "WhiteNA", smoke: str = "Non", tds: float = 0.0,
    **flags: bool,
) -> float:
    """Return the original QDiabetes-2018 10-year risk percentage."""
    variant = variant.upper()
    sex = "Male" if sex == "Male" else "Female"
    c = COEFFICIENTS[variant][sex]
    bmi = min(40.0, max(20.0, float(bmi)))
    dage, dbmi = float(age) / 10.0, bmi / 10.0
    if sex == "Female":
        age1, age2 = math.sqrt(dage) - 2.123332023620606, dage ** 3 - 91.644744873046875
        bmi1, bmi2 = dbmi - 2.571253299713135, dbmi ** 3 - 16.999439239501953
        tds_center = 0.391116052865982
    else:
        age1, age2 = math.log(dage) - 1.496392488479614, dage ** 3 - 89.048171997070313
        bmi1, bmi2 = dbmi ** 2 - 6.817805767059326, dbmi ** 3 - 17.801923751831055
        tds_center = 0.515986680984497

    eth_term = c["eth"][ETH_ORDER.index(ethnicity)] if ethnicity in ETH_ORDER else 0.0
    smoke_term = c["smoke"][SMOKE_ORDER.index(smoke)] if smoke in SMOKE_ORDER else 0.0
    age_term = c["age"][0] * age1 + c["age"][1] * age2
    bmi_term = c["bmi"][0] * bmi1 + c["bmi"][1] * bmi2
    tds_term = c["tds"] * (float(tds) - tds_center)
    values = [1.0 if flags.get(key, False) else 0.0 for key in BINARY_KEYS]
    binary_term = sum(weight * value for weight, value in zip(c["bin"], values))

    apsy, _, _, _, learn, _, _, stat, _, fhdm = values
    interaction_basis = [
        age1 * apsy, age2 * apsy, age1 * learn, age2 * learn,
        age1 * stat, age2 * stat, age1 * bmi1, age2 * bmi1,
        age1 * bmi2, age2 * bmi2,
    ]
    lab_term = 0.0
    if variant == "B":
        if lab is None:
            raise ValueError("Model B requires fasting plasma glucose")
        if sex == "Female":
            lab1, lab2 = lab ** -1 - 0.208309367299080, math.log(lab) * lab ** -1 - 0.326781362295151
        else:
            lab1, lab2 = lab ** -0.5 - 0.448028832674026, math.log(lab) * lab ** -0.5 - 0.719442605972290
        lab_term = c["lab"][0] * lab1 + c["lab"][1] * lab2
        interaction_basis.extend([age1 * lab1, age2 * lab1, age1 * lab2, age2 * lab2])
    elif variant == "C":
        if lab is None:
            raise ValueError("Model C requires HbA1c")
        scaled = lab / 10.0
        if sex == "Female":
            lab1, lab2 = math.sqrt(scaled) - 1.886751174926758, scaled - 3.559829950332642
        else:
            lab1, lab2 = math.sqrt(scaled) - 1.900265336036682, scaled - 3.611008167266846
        lab_term = c["lab"][0] * lab1 + c["lab"][1] * lab2
        interaction_basis.extend([age1 * lab1, age2 * lab1, age1 * lab2, age2 * lab2])
    interaction_basis.extend([age1 * fhdm, age2 * fhdm])
    interaction_term = sum(weight * value for weight, value in zip(c["int"], interaction_basis))
    score = eth_term + smoke_term + age_term + bmi_term + tds_term + lab_term + binary_term + interaction_term
    return 100.0 * (1.0 - c["survival"] ** math.exp(score))


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
    base = _read_csv(base_path)
    items = _read_csv(item_path)
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
            if key is None:
                continue
            person = (str(row.get("app_no", "")).strip(), str(row.get("pat_id", "")).strip())
            grouped.setdefault(person, {}).setdefault(key, (row.get("res_value"), row.get("res_unit", "")))

    results = []
    flag_map = {
        "fhdm": "diabetes_family_history", "cvd": "has_cvd", "htn": "bp_treatment",
        "stat": "statin_use", "apsy": "atypical_antipsychotics", "psy": "schizophrenia_bipolar",
        "learn": "learning_disability", "ster": "corticosteroids", "gdm": "gestational_diabetes", "pcos": "pcos",
    }
    for _, row in base.iterrows():
        person = (str(row.get("app_no", "")).strip(), str(row.get("pat_id", "")).strip())
        data = grouped.get(person, {})
        age = _number(row.get("app_pat_age")) or _number(data.get("age", (None, ""))[0])
        sex = _sex(row.get("pat_sex") or data.get("gender", ("", ""))[0])
        bmi = _number(data.get("BMI", (None, ""))[0])
        unit_notes = []
        if bmi is None:
            height = _height_m(*data.get("height", (None, "")))
            weight = _weight_kg(*data.get("weight", (None, "")))
            if height and weight:
                bmi = weight / height ** 2
                unit_notes.append("bmi_from_explicit_units")
        if bmi is None:
            bmi = 25.0
            unit_notes.append("BMI missing; using project default 25")

        fpg, fpg_note = _glucose_mmol_l(*data.get("FBG", (None, "")))
        hba1c, hba1c_note = _hba1c_mmol_mol(*data.get("hba1c", (None, "")))
        if fpg is not None:
            variant, lab = "B", fpg
            unit_notes.append(fpg_note)
        elif hba1c is not None:
            variant, lab = "C", hba1c
            unit_notes.append(hba1c_note)
        else:
            variant, lab = "A", None
            if fpg_note:
                unit_notes.append(fpg_note + ", not used in computation")
            if hba1c_note:
                unit_notes.append(hba1c_note + ", not used in computation")

        ethn = _category(data.get("ethnicity", (None, ""))[0], ETHNICITIES, "WhiteNA")
        smoke = _category(data.get("smoke", (None, ""))[0], SMOKING, "Non")
        tds = _number(data.get("townsend_score", (0, ""))[0]) or 0.0
        flags = {target: _truth(data.get(source, (None, ""))[0]) for target, source in flag_map.items()}
        if sex == "Male":
            flags["gdm"] = flags["pcos"] = False

        if age is None:
            risk, status, reason = None, "not_computed", "missing_age"
        else:
            try:
                risk = calculate_qdiabetes_2018(variant, sex, age, bmi, lab=lab, ethnicity=ethn, smoke=smoke, tds=tds, **flags)
                status, reason = "computed", ""
            except Exception as exc:
                risk, status, reason = None, "not_computed", str(exc)

        results.append({
            "pat_id": person[1], "app_no": person[0], "pat_name": row.get("pat_name", ""),
            "model_name": MODEL_NAME, "model_variant": f"QDiabetes-2018 {variant}",
            "age": age if age is not None else "-", "sex": sex, "BMI": round(bmi, 4),
            "risk_probability": round(risk, 6) if risk is not None else "-",
            "risk_level": "continuous risk (no fixed tier threshold in the original algorithm)" if risk is not None else "-",
            "code": "E14" if risk is not None else "-", "gauge_reference": "",
            "unit handling": "；".join(unit_notes), "calculation_status": status, "not_computed_reason": reason,
            "calculation_basis": REFERENCE,
        })
    pd.DataFrame(results).to_csv(output_path, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    process_data("base.csv", "item_result.csv", "qdiabetes_screening_E14_result.csv")
