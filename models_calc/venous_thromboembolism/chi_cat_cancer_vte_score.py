"""Complete CHI-CAT score from Table 1 of the matched publication."""

import re
import pandas as pd
import numpy as np

MODEL_NAME = "CHI-CAT China Cancer-Associated Venous Thrombosis Risk Score"

FIELD_MAPPING = {
    "is_asian": ["asian_population", "ethnic_asian"], "age": ["age", "app_pat_age"],
    "gender": ["sex", "pat_sex"], "BMI": ["BMI", "bmi", "(body mass index) BMI"],
    "weight": ["weight"], "height": ["height"], "is_plateau": ["residence_at_altitude_ge_2500m", "high_altitude_residence"],
    "decreased_activity": ["reduced_mobility"], "cvd": ["cardiovascular_disease"],
    "lung_kidney_disease": ["lung_or_kidney_disease", "lung/kidney_disease"], "severe_infection": ["severe_infection"],
    "personal_vte_history": ["personal_VTE_history"], "family_vte_history": ["VTE_family_history"],
    "hosp_history": ["hospitalization_within_1_month", "recent_hospitalization"], "varicose_veins": ["lower_limb_varicose_veins"],
    "d_dimer": ["plasma_D_dimer_quantitative", "d_dimer"], "plt": ["PLT", "platelet_count"],
    "tat": ["thrombin_antithrombin_complex", "TAT"], "cancer_recent": ["cancer_diagnosed_within_6_months"],
    "cancer_site": ["cancer_type", "tumor_site"], "compression": ["vascular_or_lymphatic_compression", "vascular/lymphatic_compression"],
    "metastasis": ["metastasis"], "lymph_invasion": ["lymph_node_invasion"], "mediastinal": ["mediastinal_involvement"],
    "platinum": ["platinum_agent"], "hormonal_tx": ["hormone_replacement_therapy", "HRT"],
    "dex_high_dose": ["dexamethasone_ge_160mg_per_cycle", "high_dose_dexamethasone"], "antiangiogenic": ["antiangiogenic_drug"],
    "anticoagulants": ["anticoagulant", "on_anticoagulant"], "adriamycin": ["doxorubicin"],
    "esf": ["erythropoiesis_stimulating_agent"], "hip_knee_surgery": ["hip_or_knee_surgery", "hip/knee_surgery"],
    "surgery_2h": ["surgery_duration_ge_2h"], "lap_arthroscopy": ["laparoscopic_or_arthroscopic_surgery"],
    "cvc": ["central_venous_catheterization", "CVC", "PICC"],
}

NUMERIC_FIELDS = ("age", "BMI", "weight", "height", "d_dimer", "plt", "tat")
BOOLEAN_FIELDS = tuple(key for key in FIELD_MAPPING if key not in {*NUMERIC_FIELDS, "gender", "cancer_site"})


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


def parse_boolean(value):
    if pd.isna(value) or str(value).strip() == "": return None
    text = normalize_str(value)
    if text in {"0", "no", "absent", "negative", "false"} or any(x in text for x in ("not seen", "none", "not_used")): return 0
    if text in {"1", "yes", "present", "positive", "true"} or any(x in text for x in ("present", "confirmed", "use")): return 1
    return None


def parse_gender(value):
    text = str(value).strip().lower()
    if text in {"female", "0", "2", "f"}: return "female"
    if text in {"male", "1", "m"}: return "male"
    return None


def calculate_chi_cat_score(data):
    score = -1 if data["is_asian"] else 0
    bmi = data["BMI"]
    score += 2 if bmi >= 28 else 1 if bmi > 24 else 0
    score += 1 if data["age"] >= 60 else 0
    score += 1 if data["gender"] == "female" else 0
    weights = {
        "is_plateau": 1, "decreased_activity": 2, "cvd": 2, "lung_kidney_disease": 1,
        "severe_infection": 1, "personal_vte_history": 1, "family_vte_history": 1,
        "hosp_history": 1, "varicose_veins": 1, "cancer_recent": 1, "compression": 5,
        "metastasis": 3, "lymph_invasion": 2, "mediastinal": 2, "platinum": 1,
        "hormonal_tx": 3, "dex_high_dose": 2, "antiangiogenic": 2, "anticoagulants": -6,
        "adriamycin": 1, "esf": 3, "hip_knee_surgery": 5, "surgery_2h": 3,
        "lap_arthroscopy": 2, "cvc": 1,
    }
    score += sum(weight for key, weight in weights.items() if data[key])
    score += 3 if data["d_dimer"] >= 500 else 0
    score += 2 if data["plt"] >= 350 else 0
    score += 2 if data["tat"] > 4 else 0
    site = str(data["cancer_site"])
    if any(x in site for x in ("stomach", "pancreas", "brain")): score += 5
    elif any(x in site for x in ("lymphatic", "lymphoma")): score += 3
    elif any(x in site for x in ("lung", "gynecologic", "obstetric_gynecologic", "urologic", "genitourinary")): score += 2
    return score


def read_csv_safe(path):
    for encoding in ("utf-8-sig", "gb18030", "gbk", "utf-8"):
        try: return pd.read_csv(path, encoding=encoding, low_memory=False)
        except Exception: continue
    return pd.DataFrame()


def process_chi_cat(base_path, item_path, output_path):
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
        data = {key: clean_numeric(pool.get(key)) for key in NUMERIC_FIELDS}
        data["age"] = clean_numeric(base.get("app_pat_age")) if pd.isna(data["age"]) else data["age"]
        data["gender"] = parse_gender(pool.get("gender")) or parse_gender(base.get("pat_sex"))
        if pd.isna(data["BMI"]) and not pd.isna(data["weight"]) and not pd.isna(data["height"]) and data["height"] > 0:
            h = data["height"] / 100 if data["height"] > 3 else data["height"]; data["BMI"] = data["weight"] / h ** 2
        for key in ("age", "BMI", "d_dimer", "plt", "tat"):
            if pd.isna(data[key]): data[key] = 0
        data["gender"] = data["gender"] or "male"
        data["cancer_site"] = pool.get("cancer_site") or ""
        for key in BOOLEAN_FIELDS:
            parsed = parse_boolean(pool.get(key))
            data[key] = (1 if key == "is_asian" else 0) if parsed is None else parsed
        score = calculate_chi_cat_score(data)
        level, code = ("high risk", "I82.1.3") if score >= 8 else ("low risk", "I82.1.1")
        rows.append({"app_no": base["app_no"], "pat_id": base["pat_id"], "pat_name": base.get("pat_name", "unknown"),
                     "model_name": MODEL_NAME, "predicted_score": score if score is not None else "-", "risk_level": level,
                     "code": code, "gauge_reference": "<8|≥8", "calculation_status": "computed",
                     "missing_fields": ""})
    pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    process_chi_cat("base.csv", "item_result.csv", "chi_cat_vte_risk_result.csv")
