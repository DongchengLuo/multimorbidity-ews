"""Nasopharyngeal-cancer PRS from the 12 published joint ORs."""

import math
import re
import pandas as pd
import numpy as np

MODEL_NAME = "NPC_PRS_NatureComm2022"

FIELD_MAPPING = {
    **{snp: [snp] for snp in (
        "rs2106123", "rs31489", "rs3131875", "rs1611163",
        "rs9357092", "rs9261506", "rs2251830", "rs2596506",
        "rs2844484", "rs9268644", "rs6475604", "rs9507124",
    )},
    "age": ["app_pat_age", "age"],
    "gender": ["pat_sex", "sex"],
}

SNP_OR_DATA = {
    "rs2106123": 0.84, "rs31489": 0.77, "rs3131875": 1.23,
    "rs1611163": 0.61, "rs9357092": 1.26, "rs9261506": 0.84,
    "rs2251830": 1.30, "rs2596506": 0.70, "rs2844484": 0.87,
    "rs9268644": 0.63, "rs6475604": 0.74, "rs9507124": 1.16,
}
# Keep the sign: OR < 1 is protective and therefore has a negative weight.
SNP_WEIGHTS = {snp: math.log(odds_ratio) for snp, odds_ratio in SNP_OR_DATA.items()}


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


def clean_dosage(value):
    if pd.isna(value) or str(value).strip() == "":
        return np.nan
    match = re.search(r"[-+]?\d*\.?\d+", str(value))
    if not match:
        return np.nan
    dosage = float(match.group())
    return dosage if 0 <= dosage <= 2 else np.nan


def calculate_npc_prs(snp_values):
    """Return signed log-odds PRS and calculation details.

    Each dosage must be the dosage of the effect allele used for the published
    OR. A population reference distribution is not available in the matched
    material, so this function deliberately does not invent risk cut-points.
    """
    missing = [snp for snp in SNP_WEIGHTS if pd.isna(snp_values.get(snp, np.nan))]
    if missing:
        return np.nan, {"missing_snps": missing}
    terms = {snp: float(snp_values[snp]) * weight for snp, weight in SNP_WEIGHTS.items()}
    return sum(terms.values()), terms


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
        group = item_df[
            (item_df.get("app_no") == base["app_no"])
            & (item_df.get("pat_id") == base["pat_id"])
        ] if not item_df.empty else pd.DataFrame()
        values = {}
        for _, item in (group.dropna(subset=["cat"]) if "cat" in group.columns else group).iterrows():
            if item["cat"] in SNP_WEIGHTS and item["cat"] not in values:
                values[item["cat"]] = clean_dosage(item.get("res_value"))
        score, details = calculate_npc_prs(values)
        success = not pd.isna(score)
        rows.append({
            "app_no": base["app_no"], "pat_id": base["pat_id"],
            "pat_name": base.get("pat_name", "unknown"), "model_name": MODEL_NAME,
            "calculation_status": "continuous PRS only; no risk tiering" if success else "data_missing",
            "NPC_PRS_log_ratio": round(score, 6) if success else "-",
            "NPC_PRS_relative_ratio": round(math.exp(score), 6) if success else "-",
            "risk_level": "-", "code": "-", "calculation_details": str(details),
        })
    pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    process_data("base.csv", "item_result.csv", "npc_prs_screening_result.csv")
