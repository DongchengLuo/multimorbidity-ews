"""NPC screening risk using P85-Ab and the published dual-antibody model."""

import math
import re
import pandas as pd
import numpy as np

FIELD_MAPPING = {
    "age": ["app_pat_age", "age"],
    "gender": ["pat_sex", "sex"],
    "p85_ab": ["P85", "BNLF2b", "P85-Ab", "anti_BNLF2b_total_antibody", "p85_antibody"],
    "ebna1_iga": ["EBV_antibody_IgA", "EBNA_IgA", "EBV-EBNA1-IgA"],
    "vca_iga": ["EBV_VCA_IgA", "anti_EBV_VCA_IgA", "EBV-VCA-IgA"],
}


def normalize_str(value):
    if pd.isna(value):
        return ""
    return re.sub(r"[()（）\[\]【】{}\s]", "", str(value)).lower()


def get_category(name):
    normalized = normalize_str(name)
    for key, aliases in FIELD_MAPPING.items():
        if any(normalize_str(alias) == normalized for alias in aliases):
            return key
    return None


def clean_numeric(value):
    if pd.isna(value) or str(value).strip() == "":
        return np.nan
    match = re.search(r"[-+]?\d*\.?\d+", str(value))
    return float(match.group()) if match else np.nan


def parse_qualitative(value):
    if pd.isna(value) or str(value).strip() == "":
        return None
    text = normalize_str(value)
    if any(token in text for token in ("negative", "negative", "neg", "not_detected", "no")) or text == "0":
        return 0
    if any(token in text for token in ("positive", "positive", "pos", "detected", "yes")) or text in {"1", "+"}:
        return 1
    return None


def calculate_npc_risk(ebna1_val, vca_val, p85_status):
    """P85 positivity is independently screen-positive in the original path."""
    if p85_status not in (0, 1):
        return None
    if p85_status == 1:
        probability = None
        if not pd.isna(ebna1_val) and not pd.isna(vca_val):
            z = -3.934 + 4.797 * ebna1_val + 2.203 * vca_val
            probability = 1 / (1 + math.exp(-z))
        return {"prob": probability, "risk_level": "high risk", "risk_code": "C11.2.3", "screen_positive": True}
    if pd.isna(ebna1_val) or pd.isna(vca_val):
        return None
    z = -3.934 + 4.797 * ebna1_val + 2.203 * vca_val
    probability = 1 / (1 + math.exp(-z))
    if probability >= 0.98:
        level, code = "high risk", "C11.2.3"
    elif probability >= 0.65:
        level, code = "medium risk", "C11.2.2"
    else:
        level, code = "low risk", "C11.2.1"
    return {"prob": probability, "risk_level": level, "risk_code": code, "screen_positive": probability >= 0.98}


def read_csv_safe(path):
    for encoding in ("utf-8-sig", "gb18030", "gbk", "utf-8"):
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False)
        except Exception:
            continue
    return pd.DataFrame()


def process_data(base_path, item_path, output_path):
    base_df, item_df = read_csv_safe(base_path), read_csv_safe(item_path)
    if base_df.empty:
        return
    for df in (base_df, item_df):
        if not df.empty:
            df["app_no"] = df["app_no"].astype(str).str.strip()
            df["pat_id"] = df["pat_id"].astype(str).str.strip()
    if not item_df.empty:
        item_df["cat"] = item_df["res_itm_name"].apply(get_category)

    rows = []
    for _, base in base_df.iterrows():
        group = item_df[(item_df["app_no"] == base["app_no"]) & (item_df["pat_id"] == base["pat_id"])] if not item_df.empty else pd.DataFrame()
        pool = {}
        for _, item in (group.dropna(subset=["cat"]) if "cat" in group.columns else group).iterrows():
            pool.setdefault(item["cat"], item.get("res_value"))
        ebna1, vca = clean_numeric(pool.get("ebna1_iga")), clean_numeric(pool.get("vca_iga"))
        p85 = parse_qualitative(pool.get("p85_ab"))
        result = calculate_npc_risk(ebna1, vca, p85)
        rows.append({
            "app_no": base["app_no"], "pat_id": base["pat_id"], "pat_name": base.get("pat_name", "unknown"),
            "model_name": "NPC P85-Ab Combined Screening Model", "calculation_status": "computed" if result else "core_indicator_missing",
            "risk_probability": f"{result['prob']:.2%}" if result and result["prob"] is not None else "-",
            "P85_screening_result": "positive" if result and result["screen_positive"] else ("negative" if result else "-"),
            "risk_level": result["risk_level"] if result else "-", "code": result["risk_code"] if result else "-",
            "EBNA1-IgA": ebna1 if not pd.isna(ebna1) else "missing", "VCA-IgA": vca if not pd.isna(vca) else "missing",
            "P85-Ab_result": "positive" if p85 == 1 else ("negative" if p85 == 0 else "missing"),
        })
    pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    process_data("base.csv", "item_result.csv", "npc_screening_result.csv")
