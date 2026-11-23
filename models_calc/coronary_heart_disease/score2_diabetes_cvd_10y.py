"""SCORE2-Diabetes with the four published European calibrations.

The project uses the published high-risk-region calibration as its configured
default when a region is missing or is not one of the four published regions.
"""

import math
import re
import pandas as pd
import numpy as np

MODEL_NAME = "SCORE2-Diabetes (European Calibration)"
DEFAULT_RISK_REGION = "high"
FIELD_MAPPING = {
    "age": ["app_pat_age", "age"], "gender": ["pat_sex", "sex"],
    "score2_region": ["score2_region", "score2_risk_region"],
    "sbp": ["systolic_bp", "blood_pressure"], "TC": ["total_cholesterol", "CHOL"],
    "HDL-C": ["HDL_cholesterol", "HDL-C"],
    "Hba1c": ["HbA1c", "GHb"],
    "egfr": ["eGFR (CKD-EPI)", "eGFR"],
    "scr": ["creatinine (Cr)", "creatinine"], "smoke": ["smoking_history", "current_smoker"],
    "diabetes_diagnosis_age": ["diabetes_diagnosis_age"],
}

CALIBRATION = {
    "low": {"male": (-0.5699, 0.7476), "female": (-0.7380, 0.7019)},
    "moderate": {"male": (-0.1565, 0.8009), "female": (-0.3143, 0.7701)},
    "high": {"male": (0.3207, 0.9360), "female": (0.5710, 0.9369)},
    "very_high": {"male": (0.5836, 0.8294), "female": (0.9412, 0.8329)},
}


def normalize_str(value):
    if pd.isna(value): return ""
    return re.sub(r"[()（）\[\]【】{}\s_-]", "", str(value)).lower()


def get_category(name):
    normalized = normalize_str(name)
    for key, aliases in FIELD_MAPPING.items():
        if any(normalize_str(alias) == normalized for alias in aliases): return key
    return None


def clean_numeric(value):
    if pd.isna(value) or str(value).strip() == "": return np.nan
    match = re.search(r"[-+]?\d*\.?\d+", str(value))
    return float(match.group()) if match else np.nan


def parse_gender(value):
    text = str(value).strip().lower()
    if text in {"male", "1", "m"}: return "male"
    if text in {"female", "0", "2", "f"}: return "female"
    return None


def parse_smoking(value):
    if pd.isna(value) or str(value).strip() == "": return None
    text = normalize_str(value)
    if text in {"0", "no", "absent", "never_smoker", "non-smoker", "never"}: return 0
    if text in {"1", "yes", "present", "smoking", "current_smoker", "current", "daily"}: return 1
    return None


def parse_region(value):
    text = normalize_str(value)
    if text.startswith("europe"):
        text = text[2:]
    aliases = {
        "low": {"low", "low_risk_region"}, "moderate": {"moderate", "medium_risk_region"},
        "high": {"high", "high_risk_region"},
        "very_high": {"veryhigh", "veryhighrisk", "very_high_risk_region"},
    }
    for region, names in aliases.items():
        if text in {normalize_str(name) for name in names}: return region
    return None


def calculate_egfr(scr_umol, age, sex):
    if pd.isna(scr_umol) or pd.isna(age) or sex not in {"male", "female"}: return np.nan
    scr_mgdl = scr_umol / 88.4
    kappa, alpha, multiplier = (0.7, -0.329, 1.018) if sex == "female" else (0.9, -0.411, 1.0)
    return 141 * min(scr_mgdl / kappa, 1) ** alpha * max(scr_mgdl / kappa, 1) ** -1.209 * 0.993 ** age * multiplier


def calculate_score2_diabetes(sex, age, smoking, sbp, tchol, hdl, age_diagnosis, hba1c_percent, egfr, risk_region):
    region = parse_region(risk_region) if risk_region not in CALIBRATION else risk_region
    region = region or DEFAULT_RISK_REGION
    values = [age, smoking, sbp, tchol, hdl, age_diagnosis, hba1c_percent, egfr]
    if sex not in {"male", "female"} or any(v is None or pd.isna(v) for v in values): return None
    if age_diagnosis > age or egfr <= 0: return None
    hba1c_mmol = (hba1c_percent - 2.15) * 10.929
    c_age, c_sbp, c_tchol, c_hdl = (age - 60) / 5, (sbp - 120) / 20, tchol - 6, (hdl - 1.3) / 0.5
    c_agediab, c_hba1c = (age_diagnosis - 50) / 5, (hba1c_mmol - 31) / 9.34
    c_lnegfr = (math.log(egfr) - 4.5) / 0.15
    if sex == "male":
        lp = (0.5368*c_age + 0.4774*smoking + 0.1322*c_sbp + 0.6457 + 0.1102*c_tchol - 0.1087*c_hdl
              - 0.0672*c_age*smoking - 0.0268*c_age*c_sbp - 0.0983*c_age - 0.0181*c_age*c_tchol
              + 0.0095*c_age*c_hdl - 0.0998*c_agediab + 0.0955*c_hba1c - 0.0591*c_lnegfr
              + 0.0058*c_lnegfr**2 - 0.0134*c_hba1c*c_age + 0.0115*c_lnegfr*c_age)
        baseline = 0.9605
    else:
        lp = (0.6624*c_age + 0.6139*smoking + 0.1421*c_sbp + 0.8096 + 0.1127*c_tchol - 0.1568*c_hdl
              - 0.1122*c_age*smoking - 0.0167*c_age*c_sbp - 0.1272*c_age - 0.0200*c_age*c_tchol
              + 0.0186*c_age*c_hdl - 0.1180*c_agediab + 0.1173*c_hba1c - 0.0640*c_lnegfr
              + 0.0062*c_lnegfr**2 - 0.0196*c_hba1c*c_age + 0.0169*c_lnegfr*c_age)
        baseline = 0.9776
    uncalibrated = 1 - baseline ** math.exp(lp)
    uncalibrated = min(max(uncalibrated, 0.0001), 0.9999)
    scale1, scale2 = CALIBRATION[region][sex]
    calibrated = 1 - math.exp(-math.exp(scale1 + scale2 * math.log(-math.log(1 - uncalibrated))))
    return calibrated * 100


def classify_risk(score, age):
    low_cutoff, high_cutoff = (2.5, 7.5) if age < 50 else (5.0, 10.0)
    if score < low_cutoff: return "low risk", "I25.4.1", f"0-{low_cutoff}|{low_cutoff}-{high_cutoff}|{high_cutoff}-100"
    if score < high_cutoff: return "medium risk", "I25.4.2", f"0-{low_cutoff}|{low_cutoff}-{high_cutoff}|{high_cutoff}-100"
    return "high risk", "I25.4.3", f"0-{low_cutoff}|{low_cutoff}-{high_cutoff}|{high_cutoff}-100"


def read_csv_safe(path):
    for encoding in ("utf-8-sig", "gb18030", "gbk", "utf-8"):
        try: return pd.read_csv(path, encoding=encoding, low_memory=False)
        except Exception: continue
    return pd.DataFrame()


def process_data(base_path, item_path, output_path):
    base_df, item_df = read_csv_safe(base_path), read_csv_safe(item_path)
    if base_df.empty: return
    for df in (base_df, item_df):
        if not df.empty:
            df["app_no"] = df["app_no"].astype(str).str.strip(); df["pat_id"] = df["pat_id"].astype(str).str.strip()
    if not item_df.empty: item_df["cat"] = item_df["res_itm_name"].apply(get_category)
    rows = []
    numeric = ("sbp", "TC", "HDL-C", "Hba1c", "egfr", "scr", "diabetes_diagnosis_age")
    for _, base in base_df.iterrows():
        group = item_df[(item_df["app_no"] == base["app_no"]) & (item_df["pat_id"] == base["pat_id"])] if not item_df.empty else pd.DataFrame()
        pool = {}
        for _, item in (group.dropna(subset=["cat"]) if "cat" in group.columns else group).iterrows(): pool.setdefault(item["cat"], item.get("res_value"))
        age, sex = clean_numeric(base.get("app_pat_age")), parse_gender(base.get("pat_sex"))
        vals = {key: clean_numeric(pool.get(key)) for key in numeric}
        smoking = parse_smoking(pool.get("smoke"))
        parsed_region = parse_region(pool.get("score2_region"))
        region = parsed_region or DEFAULT_RISK_REGION
        if pd.isna(vals["egfr"]) and not pd.isna(vals["scr"]): vals["egfr"] = calculate_egfr(vals["scr"], age, sex)
        score = calculate_score2_diabetes(sex, age, smoking, vals["sbp"], vals["TC"], vals["HDL-C"],
                                          vals["diabetes_diagnosis_age"], vals["Hba1c"], vals["egfr"], region)
        if score is None: level = code = gauge = "-"
        else: level, code, gauge = classify_risk(score, age)
        reason = ""
        region_detail = region if parsed_region else f"{region} (project default)"
        rows.append({"app_no": base["app_no"], "pat_id": base["pat_id"], "pat_name": base.get("pat_name", "unknown"),
                     "model_name": MODEL_NAME, "score2_region": region_detail, "risk_probability": f"{score:.2f}%" if score is not None else "-",
                     "prediction_horizon": "next 10 years" if score is not None else "-", "risk_level": level, "code": code,
                     "gauge_reference": gauge, "calculation_status": "computed" if score is not None else "not_computable", "not_computed_reason": reason})
    pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    process_data("base.csv", "item_result.csv", "score2_CVD10year_result.csv")
