"""LiFeCRC overall reduced model for both sexes (Table 2)."""

import math
import re
import pandas as pd
import numpy as np

MODEL_NAME = "Colorectal Cancer - LiFeCRC Overall Simplified Model"
OVERALL_SURVIVAL_10Y = 0.9943
OVERALL_MEAN_RISK_SCORE = 6.8089

FIELD_MAPPING = {
    "age": ["app_pat_age", "age"], "gender": ["pat_sex", "sex", "gender"],
    "height": ["height"], "wc": ["waist", "wc"], "smoke": ["smoking_history", "smoke"],
    "history": ["past_history", "past_medical_history"],
    "alcohol": ["high_daily_alcohol", "high_daily_alcohol_intake", "alcohol"],
    "activity": ["physical_activity", "exercise_status", "activity"], "veg_g": ["vegetable_intake", "daily_vegetables", "vegetables_g"],
    "dairy_g": ["dairy_intake", "daily_dairy", "dairy_g"],
    "meat_g": ["processed_meat_intake", "processed_meat", "processed_meat_g"],
    "sugar_g": ["sugar_sweetened_beverage_intake", "soft_drink_intake", "soft_drinks_g"],
}


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
    if text in {"0", "no", "absent", "false", "inactive", "non-smoker", "low"}: return 0
    if text in {"1", "yes", "present", "true", "active", "smoking", "high"}: return 1
    return None


class LiFeCRCModel:
    def predict(self, raw_data):
        def numeric(key):
            value = raw_data.get(key, 0)
            return 0.0 if value is None or pd.isna(value) else float(value)

        def flag(key):
            return 1 if raw_data.get(key) in (1, True, "1") else 0

        age, waist, height = (numeric(key) for key in ("age", "wc", "height"))
        score = (0.0781*age + 0.0117*waist + 0.0115*height + 0.1292*flag("alcohol")
                 + 0.2125*flag("smoke") - 0.0964*flag("activity")
                 - 0.0773*(numeric("veg_g")/100) - 0.0166*(numeric("dairy_g")/100)
                 + 0.0808*(numeric("meat_g")/50) + 0.0268*(numeric("sugar_g")/50))
        probability = (1 - OVERALL_SURVIVAL_10Y ** math.exp(score - OVERALL_MEAN_RISK_SCORE)) * 100
        if probability < 0.62: level, code = "low risk", "C18.2.1"
        elif probability <= 1.60: level, code = "medium risk", "C18.2.2"
        else: level, code = "high risk", "C18.2.3"
        return round(probability, 4), level, round(score, 4), code, ""


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
    model, rows = LiFeCRCModel(), []
    for _, base in base_df.iterrows():
        group = item_df[(item_df["app_no"] == base["app_no"]) & (item_df["pat_id"] == base["pat_id"])] if not item_df.empty else pd.DataFrame()
        pool = {}
        for _, item in (group.dropna(subset=["cat"]) if "cat" in group.columns else group).iterrows(): pool.setdefault(item["cat"], item.get("res_value"))
        data = {key: clean_numeric(pool.get(key)) for key in ("height", "wc", "veg_g", "dairy_g", "meat_g", "sugar_g")}
        data["age"] = clean_numeric(base.get("app_pat_age")); data["history"] = pool.get("history")
        for key in ("smoke", "alcohol", "activity"): data[key] = parse_boolean(pool.get(key))
        probability, level, score, code, reason = model.predict(data)
        rows.append({"app_no": base["app_no"], "pat_id": base["pat_id"], "pat_name": base.get("pat_name", "unknown"),
                     "model_name": MODEL_NAME, "prediction_horizon": "next 10 years" if probability is not None else "-",
                     "LiFeCRC_total_score": score, "risk_probability": f"{probability}%" if probability is not None else "-",
                     "risk_level": level, "code": code, "gauge_reference": "0-0.62|0.62-1.60|1.60-100",
                     "calculation_status": "computed" if probability is not None else "not_computable", "not_computed_reason": reason})
    pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    process_data("base.csv", "item_result.csv", "lifecrc_risk_results.csv")
