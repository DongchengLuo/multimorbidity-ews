"""Breast-cancer PRS313 (PGS Catalog PGS000004, GRCh37)."""

from __future__ import annotations

import gzip
import re
from pathlib import Path
from typing import Any

import pandas as pd


MODEL_NAME = "Breast Cancer PRS313 (PGS000004)"
REFERENCE = "Mavaddat N, et al. Am J Hum Genet. 2019;104:21-34. PGS000004."

FIELD_MAPPING = {
    "genotype_prefix": ["PRS-313 genotype", "PGS000004 genotype", "genotype_prefix"],
}


def _norm(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return re.sub(r"\s+", "", str(value)).upper().replace("CHR", "")


def _default_scoring_file() -> Path:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "evaluation_modelx" / "03_official_compare" / "sources" / "prs313" / "PGS000004.txt.gz"
        if candidate.exists():
            return candidate
    return Path("PGS000004.txt.gz")


def load_scoring_file(path: str | Path | None = None) -> pd.DataFrame:
    target = Path(path) if path else _default_scoring_file()
    with gzip.open(target, "rt", encoding="utf-8") as handle:
        table = pd.read_csv(handle, sep="\t", comment="#", dtype=str)
    table["variant_key"] = table["chr_name"].astype(str) + ":" + table["chr_position"].astype(str)
    table["effect_weight"] = table["effect_weight"].astype(float)
    return table


def effect_allele_dosage(value: Any, effect_allele: str, other_allele: str) -> float | None:
    """Parse an effect-allele dosage (0..2) or a diploid genotype string."""
    text = _norm(value)
    if not text:
        return None
    try:
        number = float(text)
        return number if 0.0 <= number <= 2.0 else None
    except ValueError:
        pass
    effect, other = _norm(effect_allele), _norm(other_allele)
    alleles = [part for part in re.split(r"[/|,;]+", text) if part]
    if len(alleles) == 1 and len(effect) == len(other) == 1 and len(text) == 2:
        alleles = list(text)
    if len(alleles) != 2 or any(allele not in {effect, other} for allele in alleles):
        return None
    return float(sum(allele == effect for allele in alleles))


def calculate_prs313(genotypes: dict[str, Any], scoring: pd.DataFrame) -> tuple[float, int, list[str]]:
    """Return additive sum(beta * effect-allele dosage), observed count and errors."""
    normalized = {_norm(key).replace("-", ":"): value for key, value in genotypes.items()}
    score, observed, errors = 0.0, 0, []
    for _, row in scoring.iterrows():
        key = _norm(row["variant_key"])
        raw = normalized.get(key)
        if raw is None:
            continue
        dosage = effect_allele_dosage(raw, row["effect_allele"], row["other_allele"])
        if dosage is None:
            errors.append(key)
            continue
        score += dosage * float(row["effect_weight"])
        observed += 1
    return score, observed, errors


def _read_csv(path: str) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "gb18030", "gbk", "utf-8"):
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False, dtype=str)
        except Exception:
            continue
    return pd.DataFrame()


def _variant_key(item_name: Any) -> str | None:
    text = _norm(item_name)
    match = re.search(r"(?:^|[^0-9])([0-9XYM]+)[:_-]([0-9]{4,})(?:$|[^0-9])", text)
    return f"{match.group(1)}:{match.group(2)}" if match else None


def process_data(base_path: str, item_path: str, snp_ref_path: str | None = None, output_path: str | None = None) -> None:
    # Backward compatible with both process_data(base,item,output) and the old
    # four-argument signature process_data(base,item,reference,output).
    if output_path is None:
        output_path, snp_ref_path = snp_ref_path, None
    output_path = output_path or "breast_cancer_prs_313_eval.csv"
    base, items = _read_csv(base_path), _read_csv(item_path)
    if base.empty:
        return
    for frame in (base, items):
        if not frame.empty:
            frame["app_no"] = frame.get("app_no", "").astype(str).str.strip()
            frame["pat_id"] = frame.get("pat_id", "").astype(str).str.strip()
    scoring = load_scoring_file(snp_ref_path)
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    if not items.empty:
        for _, row in items.iterrows():
            variant = _variant_key(row.get("res_itm_name"))
            if variant:
                person = (str(row.get("app_no", "")).strip(), str(row.get("pat_id", "")).strip())
                grouped.setdefault(person, {})[variant] = row.get("res_value")

    output = []
    expected = len(scoring)
    for _, row in base.iterrows():
        person = (str(row.get("app_no", "")).strip(), str(row.get("pat_id", "")).strip())
        score, observed, errors = calculate_prs313(grouped.get(person, {}), scoring)
        output.append({
            "pat_id": person[1], "app_no": person[0], "pat_name": row.get("pat_name", ""),
            "model_name": MODEL_NAME, "genome_build": "GRCh37", "PGS_id": "PGS000004",
            "variants_read": observed, "total_model_variants": expected, "unparsed_variants": len(errors),
            "predicted_score": round(score, 10),
            "risk_level": "continuous genetic risk (PGS000004 defines no fixed clinical tier threshold)",
            "code": "C50", "gauge_reference": "",
            "missing_handling": f"{expected - observed} missing variants counted as 0 effect alleles for compatibility",
            "calculation_status": "computed", "not_computed_reason": "", "calculation_basis": REFERENCE,
        })
    pd.DataFrame(output).to_csv(output_path, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    process_data("base.csv", "item_result.csv", None, "breast_cancer_prs_313_eval.csv")
