"""Utilities for models that must not emit a risk until the source is complete."""

from __future__ import annotations

import pandas as pd


def read_csv_safe(path: str) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "gb18030", "gbk", "utf-8"):
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False)
        except (OSError, UnicodeError, pd.errors.ParserError):
            continue
    return pd.DataFrame()


def write_unavailable_result(
    base_path: str,
    output_path: str,
    model_name: str,
    reason: str,
) -> None:
    """Write one auditable no-calculation row per application.

    A row deliberately has no usable ``code`` or risk layer, so the downstream
    report aggregator cannot mistake an incomplete implementation for a model
    result.
    """
    base_df = read_csv_safe(base_path)
    if base_df.empty:
        print(f"{model_name}: base data is empty; no results generated.")
        return

    rows = []
    for _, row in base_df.iterrows():
        rows.append(
            {
                "app_no": str(row.get("app_no", "")).strip(),
                "pat_id": str(row.get("pat_id", "")).strip(),
                "pat_name": row.get("pat_name", "unknown"),
                "model_name": model_name,
                "calculation_status": "model not computable",
                "not_computed_reason": reason,
                "risk_level": "-",
                "code": "-",
            }
        )
    pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"{model_name}: wrote {len(rows)} not-computable records to {output_path}")
