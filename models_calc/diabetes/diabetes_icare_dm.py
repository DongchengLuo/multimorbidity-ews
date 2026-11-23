"""iCARE-DM score, reproduced from Table 1 of the matched paper."""

import math
import re
import pandas as pd
import numpy as np

MODEL_NAME = "iCARE-DM Diabetes Risk Prediction Model"
GAUGE_REFERENCE = "0-22.4|22.5-54"

FIELD_MAPPING = {
    "age": ["app_pat_age", "age"], "gender": ["pat_sex", "sex"],
    "height": ["height"], "weight": ["weight"],
    "wc": ["waist", "wc"], "BMI": ["(body mass index) BMI", "BMI"],
    "smoke": ["smoking_history", "smoke", "smoker"],
    "exercise": ["exercise", "regular_exercise"],
    "central_obesity": ["central_obesity", "abdominal_obesity"],
    "has_hypertension": ["hypertension_history", "has_hypertension"],
    "has_fatty_liver": ["fatty_liver_history", "has_fatty_liver"],
    "diabetes_family_history": ["diabetes_family_history"],
    "FBG": ["GLU", "FBG", "fasting_glucose"],
    "TG": ["triglyceride", "TG"],
    "low_hdl_c": ["low_HDL_C", "low_HDL_cholesterol", "low_hdl_c"],
    "high_uric_acid": ["hyperuricemia", "high_uric_acid"],
}

BINARY_FIELDS = (
    "smoke", "exercise", "central_obesity", "has_hypertension",
    "has_fatty_liver", "diabetes_family_history", "low_hdl_c", "high_uric_acid",
)


def normalize_str(value):
    if pd.isna(value): return ""
    return re.sub(r"[()（）\[\]【】{}\s]", "", str(value)).lower()


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


def parse_boolean(value):
    if pd.isna(value) or str(value).strip() == "": return None
    text = normalize_str(value)
    if text in {"0", "no", "absent", "none", "negative", "false"} or any(x in text for x in ("not seen", "non-smoker", "no_exercise")):
        return 0
    if text in {"1", "yes", "present", "positive", "true"} or any(x in text for x in ("confirmed", "present")):
        return 1
    return None


def calculate_icare_dm_score(data):
    def numeric(key):
        value = data.get(key, np.nan)
        return float(value) if value is not None and not pd.isna(value) else np.nan

    def flag(key):
        return 1 if data.get(key) in (1, True, "1") else 0

    age, height, wc, bmi, fbg, tg = (numeric(k) for k in ("age", "height", "wc", "BMI", "FBG", "TG"))
    fbg_mg = fbg * 18.0 if pd.notna(fbg) and fbg < 20 else fbg
    tg_mg = tg * 88.57 if pd.notna(tg) and tg < 20 else tg
    whtr = wc / (height * 100 if height < 3 else height) if pd.notna(wc) and pd.notna(height) and height > 0 else 0
    tyg = math.log(tg_mg * fbg_mg / 2.0) if pd.notna(tg_mg) and pd.notna(fbg_mg) and tg_mg > 0 and fbg_mg > 0 else 0
    parts = {
        "male": 1 if data["gender"] == "male" else 0,
        "age": 5 if age >= 80 else 3 if age >= 70 else 1 if age >= 60 else 0,
        "smoking": 2 * flag("smoke"),
        "central_obesity": 3 * flag("central_obesity"),
        "no_exercise": 1 if flag("exercise") == 0 else 0,
        "bmi": 5 if bmi >= 28 else 3 if bmi >= 24 else -2 if bmi < 18.5 else 0,
        "whtr": 5 if whtr > 0.5 else 0,
        "hypertension": 2 * flag("has_hypertension"),
        "fatty_liver": 4 * flag("has_fatty_liver"),
        "family_history": 2 * flag("diabetes_family_history"),
        "fbg": 12 if fbg_mg > 110 else 6 if fbg_mg >= 100 else 0,
        "low_hdl_c": flag("low_hdl_c"),
        "tg": 3 if tg >= 1.7 else 0,
        "high_uric_acid": flag("high_uric_acid"),
        "tyg": 7 if tyg >= 9 else 4 if tyg >= 8.6 else 3 if tyg >= 8.25 else 0,
    }
    return sum(parts.values()), {"parts": parts, "TyG": tyg, "WHtR": whtr}


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
    for _, base in base_df.iterrows():
        group = item_df[(item_df["app_no"] == base["app_no"]) & (item_df["pat_id"] == base["pat_id"])] if not item_df.empty else pd.DataFrame()
        pool = {}
        for _, item in (group.dropna(subset=["cat"]) if "cat" in group.columns else group).iterrows(): pool.setdefault(item["cat"], item.get("res_value"))
        data = {key: clean_numeric(pool.get(key)) for key in ("height", "weight", "wc", "BMI", "FBG", "TG")}
        data["age"] = clean_numeric(base.get("app_pat_age"))
        data["gender"] = parse_gender(base.get("pat_sex"))
        if pd.isna(data["BMI"]) and not pd.isna(data["height"]) and not pd.isna(data["weight"]) and data["height"] > 0:
            h = data["height"] / 100 if data["height"] > 3 else data["height"]; data["BMI"] = data["weight"] / h ** 2
        for key in BINARY_FIELDS: data[key] = parse_boolean(pool.get(key))
        score, details = calculate_icare_dm_score(data)
        if score is None: level = code = "-"
        elif score >= 22.5: level, code = "high risk", "E14.3.3"
        else: level, code = "low risk", "E14.3.1"
        rows.append({"app_no": base["app_no"], "pat_id": base["pat_id"], "pat_name": base.get("pat_name", "unknown"),
                     "model_name": MODEL_NAME, "predicted_score": score if score is not None else "-", "risk_level": level, "code": code,
                     "gauge_reference": GAUGE_REFERENCE, "calculation_status": "computed" if score is not None else "not_computable",
                     "calculation_details": str(details)})
    pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    process_data("base.csv", "item_result.csv", "iCARE_DM_result_E14.csv")
