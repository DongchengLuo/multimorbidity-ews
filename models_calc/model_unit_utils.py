"""Conservative recognition and conversion of model input units.

Convert only when the indicator + source unit + target unit rule is explicit; unknown or missing units raise an error,
Stop guessing units from magnitude. Model scripts still read the original item_result.csv.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
import unicodedata
from typing import Any


@dataclass(frozen=True)
class ConversionResult:
    value: float | None
    source_unit: str
    target_unit: str
    rule: str
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.value is not None and not self.error


def _numeric(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", text.replace(",", ""))
    if not match:
        return None
    try:
        number = float(match.group())
        return number if math.isfinite(number) else None
    except ValueError:
        return None


def normalize_unit(unit: Any) -> str:
    """Normalize common full-width characters, micro symbols, case and spacing differences."""
    if unit is None:
        return ""
    text = unicodedata.normalize("NFKC", str(unit)).strip().lower()
    if text in {"", "nan", "none", "-", "/"}:
        return ""
    text = (
        text.replace("μ", "u")
        .replace("µ", "u")
        .replace("／", "/")
        .replace("·", "")
        .replace("per", "/")
        .replace("mL", "ml")
        .replace("dL", "dl")
        .replace("mg", "mg")
        .replace("ug", "ug")
        .replace("s", "sec")
        .replace("L", "l")
        .replace("g", "g")
    )
    text = re.sub(r"\s+", "", text)
    text = text.replace("litre", "l").replace("liter", "l")
    text = text.replace("mmo1", "mmol").replace("mo1", "mol")

    aliases = {
        "mmol/l": "mmol/l",
        "mmoll-1": "mmol/l",
        "mmol/litre": "mmol/l",
        "umol/l": "umol/l",
        "umoll-1": "umol/l",
        "mg/dl": "mg/dl",
        "mgdl": "mg/dl",
        "g/dl": "g/dl",
        "gdl": "g/dl",
        "g/l": "g/l",
        "mg/l": "mg/l",
        "ug/l": "ug/l",
        "ug/ml": "ug/ml",
        "ng/ml": "ng/ml",
        "pmol/l": "pmol/l",
        "u/l": "u/l",
        "iu/l": "iu/l",
        "miu/ml": "iu/l",
        "u/ml": "u/ml",
        "ku/l": "u/ml",
        "ratio": "ratio",
        "fraction": "ratio",
        "ratio": "ratio",
        "dimensionless": "ratio",
        "1": "ratio",
        "ml": "ml",
        "cm3": "ml",
        "cm^3": "ml",
        "cm³": "ml",
        "mm": "mm",
        "cm": "cm",
        "m": "m",
        "kg": "kg",
        "kg/m2": "kg/m2",
        "kg/m^2": "kg/m2",
        "10^9/l": "10^9/l",
        "mg/mmol": "mg/mmol",
        "mg/mmoll": "mg/mmol",
        "mg/g": "mg/g",
        "meq/l": "meq/l",
        "s": "sec",
        "sec": "sec",
        "second": "sec",
        "seconds": "sec",
        "mmhg": "mmhg",
        "%": "%",
        "ml/min": "ml/min/1.73m2",
        "ml/min/1.73m2": "ml/min/1.73m2",
        "ml/min/1.73m^2": "ml/min/1.73m2",
        "ml/min/1.73m²": "ml/min/1.73m2",
        "mg/lfeu": "mg/l feu",
        "ug/mlfeu": "ug/ml feu",
        "ug/lfeu": "ug/l feu",
        "ng/mlfeu": "ng/ml feu",
        "mg/lddu": "mg/l ddu",
        "ug/mlddu": "ug/ml ddu",
        "ug/lddu": "ug/l ddu",
        "ng/mlddu": "ng/ml ddu",
    }
    return aliases.get(text, text)


def _unit_from_value(value: Any) -> str:
    """Only when res_unit is empty, try to read an explicit unit after the number in res_value."""
    text = str(value or "").strip()
    match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", text.replace(",", ""))
    if not match:
        return ""
    return normalize_unit(text[match.end():])


# Multiplicative factor: target_value = source_value * factor
_FACTORS = {
    ("cholesterol", "mmol/l", "mg/dl"): 38.67,
    ("cholesterol", "mg/dl", "mmol/l"): 1 / 38.67,
    ("cholesterol", "g/l", "mg/dl"): 100.0,
    ("cholesterol", "mg/l", "mg/dl"): 0.1,
    ("glucose", "mmol/l", "mg/dl"): 18.018,
    ("glucose", "mg/dl", "mmol/l"): 1 / 18.018,
    ("creatinine", "umol/l", "mg/dl"): 1 / 88.4,
    ("creatinine", "mg/dl", "umol/l"): 88.4,
    ("albumin", "g/l", "g/dl"): 0.1,
    ("albumin", "g/dl", "g/l"): 10.0,
    ("calcium", "mmol/l", "mg/dl"): 4.008,
    ("calcium", "mg/dl", "mmol/l"): 1 / 4.008,
    ("phosphate", "mmol/l", "mg/dl"): 3.097,
    ("phosphate", "mg/dl", "mmol/l"): 1 / 3.097,
    ("acr", "mg/mmol", "mg/g"): 8.84,
    ("acr", "mg/g", "mg/mmol"): 1 / 8.84,
    ("bicarbonate", "mmol/l", "meq/l"): 1.0,
    ("bicarbonate", "meq/l", "mmol/l"): 1.0,
    ("fibrinogen", "mg/dl", "g/l"): 0.01,
    ("triglyceride", "mg/dl", "mmol/l"): 0.01129,
    ("triglyceride", "mmol/l", "mg/dl"): 1 / 0.01129,
    ("triglyceride", "g/l", "mmol/l"): 1.129,
    ("psa", "ug/l", "ng/ml"): 1.0,
    ("pepsinogen", "ug/l", "ng/ml"): 1.0,
    ("psa_ratio", "%", "ratio"): 0.01,
    ("percentage", "ratio", "%"): 100.0,
    ("ratio_value", "%", "ratio"): 0.01,
    ("eosinophils", "ratio", "%"): 100.0,
    ("height", "cm", "m"): 0.01,
    ("height", "m", "cm"): 100.0,
    ("height", "mm", "cm"): 0.1,
    ("blood_pressure", "kpa", "mmhg"): 7.50062,
    ("weight", "g", "kg"): 0.001,
    ("liver_enzyme", "iu/l", "u/l"): 1.0,
    ("d_dimer_mass", "ug/ml", "mg/l"): 1.0,
    ("d_dimer_mass", "ug/l", "mg/l"): 0.001,
    ("d_dimer_mass", "ng/ml", "mg/l"): 0.001,
    ("d_dimer", "mg/l feu", "mg/l feu"): 1.0,
    ("d_dimer", "ug/ml feu", "mg/l feu"): 1.0,
    ("d_dimer", "ug/l feu", "mg/l feu"): 0.001,
    ("d_dimer", "ng/ml feu", "mg/l feu"): 0.001,
    ("d_dimer", "mg/l ddu", "mg/l feu"): 2.0,
    ("d_dimer", "ug/ml ddu", "mg/l feu"): 2.0,
    ("d_dimer", "ug/l ddu", "mg/l feu"): 0.002,
    ("d_dimer", "ng/ml ddu", "mg/l feu"): 0.002,
}

# target_value = source_value * scale + offset
_AFFINE = {
    # IFCC HbA1c (mmol/mol) -> NGSP HbA1c (%)
    ("percentage", "mmol/mol", "%"): (0.09148, 2.152),
}


def convert_measurement(
    value: Any,
    unit: Any,
    *,
    indicator: str,
    target_unit: str,
    allow_unitless: bool = False,
) -> ConversionResult:
    number = _numeric(value)
    target = normalize_unit(target_unit)
    source = normalize_unit(unit) or _unit_from_value(value)
    if number is None:
        return ConversionResult(None, source, target, "", "value_unparseable")
    if not source:
        if allow_unitless:
            return ConversionResult(number, "", target, "no unit provided; using target unit")
        return ConversionResult(None, "", target, "missing_res_unit")
    if source == target:
        return ConversionResult(number, source, target, "unit_consistent")
    affine = _AFFINE.get((indicator, source, target))
    if affine is not None:
        scale, offset = affine
        return ConversionResult(
            number * scale + offset, source, target,
            f"{source} × {scale:.12g} + {offset:.12g} -> {target}"
        )
    factor = _FACTORS.get((indicator, source, target))
    if factor is None:
        return ConversionResult(None, source, target, "", f"unsupported unit conversion: {source} -> {target}")
    return ConversionResult(number * factor, source, target, f"{source} × {factor:.12g} -> {target}")


def align_d_dimer_pair(value: Any, unit: Any, uln: Any, uln_unit: Any):
    """Align D-dimer with ULN; with the same non-empty unit, take the ratio directly."""
    value_num, uln_num = _numeric(value), _numeric(uln)
    value_unit, limit_unit = normalize_unit(unit), normalize_unit(uln_unit)
    if value_num is None or uln_num is None:
        return None, None, "D-dimer or ULN value unparseable"
    if uln_num <= 0:
        return None, None, "D-dimer ULN must be > 0"
    if value_unit and value_unit == limit_unit:
        return value_num, uln_num, "direct comparison in same unit"
    value_result = convert_measurement(
        value, unit, indicator="d_dimer", target_unit="mg/L FEU"
    )
    limit_result = convert_measurement(
        uln, uln_unit, indicator="d_dimer", target_unit="mg/L FEU"
    )
    if not value_result.ok or not limit_result.ok:
        errors = "; ".join(x.error for x in (value_result, limit_result) if x.error)
        return None, None, errors
    return value_result.value, limit_result.value, (
        f"D-dimer[{value_result.rule}]; ULN[{limit_result.rule}]"
    )
