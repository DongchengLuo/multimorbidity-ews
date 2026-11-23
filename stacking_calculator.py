# stacking_calculator.py
#
# Read the trained stacking Cox model JSONs from the stacking_formulas directory,
# Use each risk model's computed score as input to derive each disease's 5-year integrated risk.
#
# Prediction definition in the model JSON:
#   risk(5y) = 1 - exp[-H0(5y) * exp(spline_design(scores) dot beta)]
#
# spline_design is implemented from spline_spec in the JSON:
#   1. Standardize each input score: z = (x - mean) / sd
#   2. Build a B-spline basis with knots as nodes and degree as order (len(knots)-degree-1 columns)
#   3. Drop the first column (collinear with the intercept), keep nb-1 columns
#   4. Subtract centers (training-set column means) to center
#   5. Weighted sum in cox_beta order
#
# Missing-score handling: reuse training-time median imputation (spline_spec.medians),
# Also list imputed score names and values separately in the results for traceability.
import glob
import json
import math
import os
import re
from typing import Dict, List, Optional

import numpy as np
from scipy.interpolate import BSpline


_MISSING_TOKENS = ("missing", "cannot_compute", "not_computable", "not_computed", "not_applicable")


def _is_missing(value) -> bool:
    text = str(value).strip()
    if text in ("", "nan", "NaN", "None", "null", "-", "--"):
        return True
    return any(token in text for token in _MISSING_TOKENS)


def _parse_number(value) -> Optional[float]:
    """Extract the number from a score string; for percentages take the digits only and do not rescale here."""
    if value is None:
        return None
    text = str(value).strip()
    if _is_missing(text):
        return None
    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group())
    except ValueError:
        return None


class StackingCalculator:
    """Compute the stacking integrated risk per disease."""

    def __init__(self, project_root: str):
        self.project_root = project_root
        self.models_dir = os.path.join(project_root, "stacking_formulas")
        self.mapping_path = os.path.join(project_root, "stacking_score_mapping.json")
        self._models: Optional[Dict[str, List[dict]]] = None
        self._mapping: Optional[dict] = None

    # ------------------------------------------------------------------
    # Configuration loading
    # ------------------------------------------------------------------
    def _load_mapping(self) -> dict:
        if self._mapping is None:
            try:
                with open(self.mapping_path, "r", encoding="utf-8") as f:
                    self._mapping = json.load(f)
            except Exception as exc:
                print(f"WARNING: failed to load stacking_score_mapping.json: {exc}")
                self._mapping = {"disease_map": {}, "score_map": {}}
        return self._mapping

    def _load_models(self) -> Dict[str, List[dict]]:
        if self._models is None:
            models: Dict[str, List[dict]] = {}
            for path in sorted(glob.glob(os.path.join(self.models_dir, "*.json"))):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        model = json.load(f)
                    disease = str(model.get("disease", "")).strip()
                    if not disease:
                        continue
                    model["_source_file"] = os.path.basename(path)
                    models.setdefault(disease, []).append(model)
                except Exception as exc:
                    print(f"WARNING: failed to read stacking model {os.path.basename(path)}: {exc}")
            self._models = models
        return self._models

    def stacking_diseases_for(self, project_disease_name: str) -> List[str]:
        """Project disease name -> disease name in stacking models (one disease may map to several models)."""
        mapping = self._load_mapping().get("disease_map", {})
        target = mapping.get(project_disease_name)
        if target is None:
            return []
        if isinstance(target, list):
            return [str(item) for item in target]
        return [str(target)]

    # ------------------------------------------------------------------
    # Score extraction
    # ------------------------------------------------------------------
    def _extract_score(self, entry: dict, row: dict) -> Optional[float]:
        column = entry.get("column")
        if not column or column not in row:
            return None

        raw_value = row.get(column)
        if entry.get("extractor") == "count_hits":
            # Models such as STOP-Bang output only the hit-item text; the total is reconstructed from the number of hits
            text = str(raw_value).strip()
            if _is_missing(text):
                return None
            parts = [part for part in re.split(r"[,，、;；]", text) if part.strip()]
            return float(len(parts)) if parts else None

        value = _parse_number(raw_value)
        if value is None:
            return None

        # Scale alignment: some stacking inputs were 0-1 probabilities and others percentages during training.
        # fraction means the input was a 0-1 probability in training; divide by 100 if the project outputs a percentage.
        if entry.get("scale") == "fraction":
            text = str(raw_value)
            column_is_percent = ("(%)" in column) or ("percentage" in column)
            if "%" in text or column_is_percent:
                value = value / 100.0
        return value

    def _collect_scores(self, model: dict, model_results: List[dict]) -> Dict[str, float]:
        """Extract the scores stacking needs from the project's model results."""
        score_map = self._load_mapping().get("score_map", {})
        # model name -> raw row (a model may have several results; take the first)
        rows_by_model: Dict[str, dict] = {}
        for result in model_results:
            model_name = str(result.get("model", "")).strip()
            raw_row = result.get("raw_row")
            if model_name and isinstance(raw_row, dict) and model_name not in rows_by_model:
                rows_by_model[model_name] = raw_row

        scores: Dict[str, float] = {}
        for input_name in model.get("input_score_columns_in_order", []):
            entry = score_map.get(input_name)
            if not entry:
                continue
            row = rows_by_model.get(entry.get("file"))
            if row is None:
                continue

            condition = entry.get("condition")
            if condition:
                condition_value = str(row.get(condition.get("column", ""), ""))
                if "contains" in condition and condition["contains"] not in condition_value:
                    continue
                if "not_contains" in condition and condition["not_contains"] in condition_value:
                    continue

            value = self._extract_score(entry, row)
            if value is not None and math.isfinite(value):
                scores[input_name] = value
        return scores

    # ------------------------------------------------------------------
    # Formula computation
    # ------------------------------------------------------------------
    @staticmethod
    def _spline_basis(z: float, knots: List[float], degree: int) -> np.ndarray:
        """Build the B-spline basis and drop the first column; clamp out-of-range values to the boundary to avoid extrapolation error."""
        lower, upper = float(knots[0]), float(knots[-1])
        if z < lower:
            z = lower
        elif z > upper:
            z = upper

        n_basis = len(knots) - degree - 1
        basis = np.zeros(n_basis, dtype=float)
        for index in range(n_basis):
            coeffs = np.zeros(n_basis, dtype=float)
            coeffs[index] = 1.0
            basis[index] = BSpline(knots, coeffs, degree, extrapolate=False)(z)
        return basis[1:]

    def _predict(self, model: dict, scores: Dict[str, float]):
        spline = model.get("spline_spec", {})
        beta = model.get("cox_beta", [])
        baseline = float(model.get("baseline_cumulative_hazard_at_5y", 0.0))
        degree = int(spline.get("degree", 3))

        linear_predictor = 0.0
        beta_offset = 0
        used_scores: Dict[str, float] = {}
        imputed_scores: Dict[str, float] = {}

        for index, input_name in enumerate(model.get("input_score_columns_in_order", [])):
            value = scores.get(input_name)
            if value is None:
                value = float(spline["medians"][index])
                imputed_scores[input_name] = value
            else:
                used_scores[input_name] = value

            mean = float(spline["mean"][index])
            sd = float(spline["sd"][index])
            z = (value - mean) / sd if sd else 0.0
            basis = self._spline_basis(z, spline["knots"][index], degree)
            centers = spline["centers"][index]
            for column_index, basis_value in enumerate(basis):
                linear_predictor += (basis_value - float(centers[column_index])) * float(
                    beta[beta_offset + column_index]
                )
            beta_offset += len(basis)

        try:
            risk = 1.0 - math.exp(-baseline * math.exp(linear_predictor))
        except OverflowError:
            # exp(lp) may overflow for extreme inputs; risk is already near 1
            risk = 1.0
        risk = min(max(risk, 0.0), 1.0)
        return risk, used_scores, imputed_scores

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
    def compute_for_disease(self, project_disease_name: str, model_results: List[dict]) -> List[dict]:
        """Return the stacking results for that disease; empty list when no model matches."""
        stacking_diseases = self.stacking_diseases_for(project_disease_name)
        if not stacking_diseases:
            return []

        models = self._load_models()
        results: List[dict] = []
        for stacking_disease in stacking_diseases:
            for model in models.get(stacking_disease, []):
                try:
                    scores = self._collect_scores(model, model_results)
                    risk, used_scores, imputed_scores = self._predict(model, scores)
                    results.append(
                        {
                            "stacking_model": model.get("_source_file", ""),
                            "stacking_disease": stacking_disease,
                            "target_years": model.get("target_years"),
                            "risk_5y": round(risk, 8),
                            "risk_5y_percent": round(risk * 100.0, 4),
                            "input_scores": {key: round(value, 8) for key, value in used_scores.items()},
                            "imputed_scores": {
                                key: round(value, 8) for key, value in imputed_scores.items()
                            },
                        }
                    )
                except Exception as exc:
                    results.append(
                        {
                            "stacking_model": model.get("_source_file", ""),
                            "stacking_disease": stacking_disease,
                            "error": str(exc),
                        }
                    )
        return results
