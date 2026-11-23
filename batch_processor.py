# batch_processor.py (refactored)
import os
import shutil
import subprocess
import sys
import json
import pandas as pd
import glob
import time
import re
from typing import List, Dict
from dotenv import load_dotenv
from db_fetcher import DatabaseFetcher
from path_utils import get_project_root, ensure_dir_exists
from stacking_calculator import StackingCalculator

# Chinese disease name -> English standard name (for English report/HTML display)
_DISEASE_EN = None


def _disease_en_map():
    global _DISEASE_EN
    if _DISEASE_EN is None:
        try:
            path = os.path.join(get_project_root(), "disease_name_mapping.json")
            with open(path, encoding="utf-8") as f:
                _DISEASE_EN = json.load(f).get("mapping", {})
        except Exception:
            _DISEASE_EN = {}
    return _DISEASE_EN


def _english_name(zh_name):
    if not zh_name:
        return ""
    return _disease_en_map().get(str(zh_name).strip(), "")


load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"), override=False)


def load_database_config() -> Dict[str, str]:
    """Read database credentials from the deployment environment; never ship real passwords."""
    return {
        "server": os.environ.get("MEDICAL_DB_SERVER", "").strip(),
        "db_name": os.environ.get("MEDICAL_DB_NAME", "").strip(),
        "user": os.environ.get("MEDICAL_DB_USER", "").strip(),
        "password": os.environ.get("MEDICAL_DB_PASSWORD", ""),
    }


# Risk-level priority (higher value = higher risk)
RISK_LEVEL_PRIORITY = {
    "-": 0,
    "cannot compute": 0,
    "cannot assess": 0,
    "missing": 0,
    "low": 1,
    "low risk": 1,
    "medium risk": 2,
    "medium": 2,
    "high risk": 3,
    "high": 3,
    "very high risk": 4,
    "very high": 4,
}


def risk_level_priority(risk_level: str) -> int:
    """
    Get the risk-level priority; supports annotated text such as "high risk (recheck advised)".

    Module-level function shared by BatchProcessor and pipeline_runner (retrieval input filtering),
    Ensure both places use the same "medium risk and above" criterion.
    """
    text = str(risk_level or "").strip()
    if text in RISK_LEVEL_PRIORITY:
        return RISK_LEVEL_PRIORITY[text]
    if re.search(r"(?:very high|extremely high)(?: risk)?", text, re.I):
        return 4
    if text.startswith(("high risk",)) or re.search(r"[:(]high(?: risk)?", text, re.I):
        return 3
    if text.startswith(("medium risk",)) or re.search(r"[:(]medium(?: risk)?", text, re.I):
        return 2
    if text.startswith(("low risk",)) or re.search(r"[:(]low(?: risk)?", text, re.I):
        return 1
    return 0


# ---------------------------------------------------------------- Sex gating
# Sex-specific cancers: computable only for the matching sex; never applicable to the other.
# Rationale: breast/ovarian/cervical cancers are female-specific (existing models are built on female cohorts,
# male breast cancer is extremely rare and these models do not apply); prostate cancer is male-only.
SEX_EXCLUSIVE_DISEASES = {
    "Breast Cancer": "2",
    "Ovarian Cancer": "2",
    "Cervical Cancer": "2",
    "Prostate Cancer": "1",
}
SEX_LABEL = {"1": "Male", "2": "Female"}


def load_sex_map(batch_dir: str) -> Dict[str, str]:
    """Read app_no -> pat_sex from the batch base.csv ('1' male / '2' female)."""
    path = os.path.join(batch_dir, "base.csv")
    if not os.path.exists(path):
        return {}
    try:
        df = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    except Exception:
        return {}
    if "app_no" not in df.columns or "pat_sex" not in df.columns:
        return {}
    out = {}
    for app, sex in zip(df["app_no"], df["pat_sex"]):
        app = str(app).strip()
        sex = str(sex).strip()
        if app:
            out[app] = sex
    return out


def apply_sex_gating(records, sex_map):
    """Move sex-inapplicable cancers from the computable part to the not-computable part.

    Modify records in place and return statistics.
    After migration this disease no longer participates in risk tiering, stacking or subgraph retrieval.
    """
    stats = {"moved": 0, "patients": 0, "details": []}
    for rec in records:
        app = str(rec.get("app_no", "")).strip()
        sex = str(sex_map.get(app, "")).strip()
        if sex not in SEX_LABEL:
            continue

        computable = rec.get("computable") or []
        keep, moved = [], []
        for d in computable:
            name = str(d.get("disease_name", "")).strip()
            need = SEX_EXCLUSIVE_DISEASES.get(name)
            if need and sex != need:
                moved.append((d, need))
            else:
                keep.append(d)
        if not moved:
            continue

        rec["computable"] = keep
        uncomputable = rec.setdefault("not_computable", [])
        for d, need in moved:
            entry = {
                "disease_name": d.get("disease_name", ""),
                "model_results": [
                    {
                        "model": m.get("model", ""),
                        "risk_level": "",
                        "score": "",
                        "calculation_status": "not_applicable",
                        "source_table": m.get("source_table", {}),
                        "source_page": m.get("source_page", ""),
                        "guideline_version": m.get("guideline_version", ""),
                    }
                    for m in (d.get("model_results") or [])
                ],
                "sex_gate": {
                    "required_sex": SEX_LABEL[need],
                    "actual_sex": SEX_LABEL[sex],
                    "reason": f"{SEX_LABEL[sex]} not applicable to {SEX_LABEL[need]}-specific cancers",
                },
            }
            # The report side reads the top-level "reason" field; write it in sync
            entry["reason"] = entry["sex_gate"]["reason"]
            uncomputable.append(entry)
            stats["moved"] += 1
            stats["details"].append({
                "app_no": app,
                "disease": entry["disease_name"],
                "sex": SEX_LABEL[sex],
            })
        stats["patients"] += 1
    return stats


class BatchProcessor:
    def __init__(self, db_config: Dict = None):
        self.project_root = get_project_root()
        self.models_dir = os.path.join(self.project_root, "models_calc")
        self.db_config = db_config or load_database_config()
        # Load the disease configuration map
        self.disease_map = self._load_disease_map()
        self.stacking_calculator = StackingCalculator(self.project_root)
        self.last_batch_id = None
        self.last_batch_dir = None

    def _load_disease_map(self) -> Dict[str, str]:
        """Load the ICD-prefix -> disease-name map from disease_config.json"""
        config_path = os.path.join(self.project_root, "disease_config.json")
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            return {code: info["name"] for code, info in config.items()}
        except Exception as e:
            print(f"WARNING: failed to load disease_config.json: {e}")
            return {}

    def _code_to_disease_name(self, risk_code: str) -> str:
        """Map a risk code (e.g. I25.5.1) to a disease name (e.g. ASCVD)"""
        if not risk_code or risk_code in ("-", "", "N/A"):
            return "-"
        # Extract the prefix: text before the first '.' (e.g. I25.5.1 -> I25)
        prefix = risk_code.split('.')[0] if '.' in risk_code else risk_code
        return self.disease_map.get(prefix, risk_code)

    def _empty_results(self, app_nos: List[str]) -> List[Dict]:
        """Return empty results for app_no without data"""
        return [
            {"app_no": no, "pat_id": "", "computable": [], "not_computable": []}
            for no in app_nos
        ]

    # ------------------------------------------------------------------
    # source_table: restore the input fields actually used from the batch cache (model result CSVs)
    # ------------------------------------------------------------------
    # Columns in the result CSV that are output/metadata are excluded from input fields
    SOURCE_TABLE_EXCLUDED_COLUMNS = {"app_no", "pat_id", "pat_name", "name", "serial_number", "ID", "model_name", "Model_Name", "assessment_model_name", "model_used", "model_variant", "prediction_horizon", "predicted_target_node", "prediction_result", "risk_criteria", "reference_basis", "reference", "clinical_recommendation", "recommendation", "suggestion", "assessment_note", "note", "note (abnormal reason)", "calculation_status", "not_computed_reason", "calculation_basis", "calculation_details", "unit handling", "unit_conversion", "unit_error", "score_details", "scoring_details", "missing_fields", "data_annotation", "conclusion_status", "determination_details", "high_risk_hits", "current_diagnosis_category", "5y_liver_event_risk", "targeted_interpretation", "total_model_variants", "variants_read", "unparsed_variants", "genome_build", "PGS_id", "missing_handling", "risk_determination", "risk_level", "code", "gauge_reference", "overall_CVD_risk_tier", "overall_CVD_tier_scale", "overall_CVD_risk", "risk_probability", "predicted_score", "risk_score", "score", "screening_score", "total_score", "assessment_total", "risk_probability(P)", "risk_value", "cancer_probability", "risk_probability(%)", "cancer_probability(%)", "5y_cancer_probability", "5y_absolute_risk", "5y_kidney_failure_risk(%)", "1y_cancer_probability(%)", "3-year_cancer_probability (%)", "copd_probability (%)", "asthma_probability (%)", "CIN3+ immediate risk (%)", "CIN3+ 5-year risk (%)", "prostate_cancer_risk", "high_grade_prostate_cancer_risk (%)", "npc_probability", "acFibroMASH_probability", "aMAP_score", "Liver Cancer aMAP score", "Lung Cancer Risk Score (LCRS)", "Gastric Cancer NGCSS Screening Score", "ad8_score", "total_malignancy_probability", "total_relative_risk(RR)", "VTE risk probability (6 months after cancer diagnosis and hospitalization)", "CKD-SCORED_score", "OSTA_index", "SCORED_total", "osta_index", "10y_heart_failure_risk", "10y_heart_failure_probability", "overall_CVD_probability (P_CVD)", "cvd_risk", "chd_risk", "PRS313_raw_weighted_score", "Caprini 2010 total", "CHI-CAT total", "HFA_PEFF total", "IMPROVEDD total", "Legend total", "METS-VF score", "NPC-CRS total", "risk_grade", "risk_tier", "risk_category", "risk_code", "ICD_code", "disease_code", "source_table", "source_page", "guideline_version", "status", "assessment_status", "10-year_CKD_probability", "30-year absolute risk", "CKB_total_score", "EBV_genetic_score", "HCC-LSM_total_score", "lcrs_total_score", "LiFeCRC_total_score", "logit_intermediate", "NPC_PRS_log_ratio", "NPC_PRS_relative_ratio", "pars_total_score", "RC1_predicted_score", "baseline_clinical_risk (RC2)", "prostate_cancer_risk (RC1)", "high_grade_cancer_risk (RC1-HG)", "relative_risk (RR)", "nomogram_total_score", "per_type_probability_details", "intermediate_variables (z)", "bmi_for_calculation", "smoker", "urban", "has_polyp", "on_aspirin", "family_history_count", "glucose_category_change_score", "indicator_details", "extended_indicators", "computation_inputs", "target_population_determination", "population_category", "population_type", "eGFR (computed)", "eGFR_final (mL/min/1.73m2)", "eGFR_source", "framingham_points", "heart_age", "acMASH_index", "estimated_5y_cumulative_cancer_rate"}
    SOURCE_TABLE_EXCLUDED_SUFFIXES = ("unit handling", "unit_conversion", "unit_error", "raw_unit")
    SOURCE_TABLE_EXCLUDED_PREFIXES = ("model",)
    SOURCE_TABLE_EXCLUDED_TOKENS = (
        "probability", "risk", "score", "total_score", "points", "rating", "computation", "details", "recommendation",
        "note", "basis", "tier", "code", "encoding", "time", "source", "page", "version",
        "variant", "handling", "conversion", "error", "extended", "population", "determination_basis",
    )
    # These are derived input fields actually used by the model; show them even though they look intermediate
    SOURCE_TABLE_INCLUDE_OVERRIDES = {
        "bmi_for_calculation", "smoker", "urban", "has_polyp", "on_aspirin",
        "family_history_count",
    }

    @staticmethod
    def _is_missing_value(value) -> bool:
        """Check whether a field value in the batch cache is empty/missing."""
        text = str(value).strip()
        if text in ("", "nan", "NaN", "None", "null", "-", "--"):
            return True
        return any(token in text for token in ("missing", "cannot_compute", "not_computable", "not_computed", "not_applicable"))

    def _is_source_table_excluded(self, column: str) -> bool:
        if column in self.SOURCE_TABLE_INCLUDE_OVERRIDES:
            return False
        if column in self.SOURCE_TABLE_EXCLUDED_COLUMNS:
            return True
        if column.endswith(self.SOURCE_TABLE_EXCLUDED_SUFFIXES):
            return True
        if column.startswith(self.SOURCE_TABLE_EXCLUDED_PREFIXES):
            return True
        return any(token in column for token in self.SOURCE_TABLE_EXCLUDED_TOKENS)

    def _build_source_table(self, row) -> Dict:
        """
        Restore computation inputs from the current row of a model result CSV:
        - Fields with values -> {"field name": "value"}
        - Fields without values -> ["field name", ...]
        """
        fields = {}
        missing_fields = []
        for column in row.index:
            if self._is_source_table_excluded(str(column)):
                continue
            value = row.get(column)
            if self._is_missing_value(value):
                missing_fields.append(str(column))
            else:
                fields[str(column)] = str(value)

        # A few models' result CSVs omit raw input fields; fall back to their detail/missing-field columns,
        # Avoid a completely empty source_table.
        if not fields and not missing_fields:
            for column in ("calculation_details", "scoring_details", "missing_handling", "assessment_note"):
                if column in row and not self._is_missing_value(row[column]):
                    fields[column] = str(row[column])
            if "missing_fields" in row and not self._is_missing_value(row["missing_fields"]):
                missing_fields.extend(
                    item.strip()
                    for item in re.split(r"[,，、;；]", str(row["missing_fields"]))
                    if item.strip()
                )

        source_table = {"fields": fields, "missing_fields": missing_fields}
        if not fields and not missing_fields:
            source_table["note"] = "this model result CSV does not record input fields"
        # Support table numbers / literature info that models such as ASCCP put in the "source table" column
        original_reference = str(row.get("source_table", "")).strip()
        if original_reference and original_reference not in ("nan", "-"):
            source_table["reference"] = original_reference
        return source_table

    def process_batch(self, app_nos: List[str], progress_callback=None) -> List[Dict]:
        """
        Process a batch of app_no and return assessment results
        """
        # 1. Create batch working directory
        batch_id = f"batch_{len(app_nos)}_{time.time_ns()}"
        batch_dir = os.path.join(self.project_root, "batches", batch_id)
        ensure_dir_exists(batch_dir)
        self.last_batch_id = batch_id
        self.last_batch_dir = batch_dir

        print(f"Starting batch: {batch_id}")
        print(f"app_nos: {app_nos}")

        def notify(stage: str, percent: int, message: str):
            if progress_callback:
                progress_callback(stage, percent, message)

        try:
            # 2. Fetch data
            notify("database_fetch", 7, "Reading patient, lab and exam report data")
            print("Step 1: fetching data from the database...")
            fetcher = DatabaseFetcher(**self.db_config)
            fetcher.fetch_patient_data(app_nos, batch_dir)

            # 2.5 Check whether key data exists; skip app_no without data
            pes_apply_path = os.path.join(batch_dir, "PES_APPLY.csv")
            pes_result_path = os.path.join(batch_dir, "PES_RESULT.csv")

            if not os.path.exists(pes_apply_path) or os.path.getsize(pes_apply_path) == 0:
                print("WARNING: PES_APPLY has no data; skipping this batch")
                return self._empty_results(app_nos)

            if not os.path.exists(pes_result_path) or os.path.getsize(pes_result_path) == 0:
                print("WARNING: PES_RESULT has no lab data; skipping this batch")
                return self._empty_results(app_nos)

            # 3. Data preprocessing
            notify("database_normalize", 18, "Merging the six business tables into model-standard input")
            print("Step 2: data preprocessing...")
            self.run_preprocessing(batch_dir)

            # 4. Model computation
            notify("database_models", 30, "Running risk models based on available indicators")
            print("Step 3: running model computation...")
            self.run_models(batch_dir)

            # 5. Collect and format results
            print("Step 4: collecting results...")
            model_results = self.collect_all_results(batch_dir, app_nos)
            print(f"  collected {len(model_results)} assessment results")
            formatted_results = self.format_results(model_results, app_nos)

            # Sex gating: drop sex-inapplicable cancers (e.g. male breast/ovarian cancer)
            sex_map = load_sex_map(batch_dir)
            gate_stats = apply_sex_gating(formatted_results, sex_map)
            if gate_stats["moved"]:
                print(f"  Sex gating: {gate_stats['patients']} patients, removed "
                      f"{gate_stats['moved']} sex-inapplicable cancer items")
                for d in gate_stats["details"]:
                    print(f"    - {d['app_no']} ({d['sex']}) removed {d['disease']}")
            else:
                print("  Sex gating: nothing to remove")

            valid_disease_count = sum(len(item.get("computable", [])) for item in formatted_results)
            notify("risk_complete", 42, f"Model computation complete: {valid_disease_count} valid disease results")

            # 6. Save results
            output_file = os.path.join(batch_dir, "batch_results.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(formatted_results, f, ensure_ascii=False, indent=2)

            print(f"[OK] Batch complete! Results saved to: {output_file}")

            # 7. Keep the batch directory (with intermediate files) for later inspection
            # To clean up manually: python -c "from batch_processor import BatchProcessor; BatchProcessor().cleanup_old_batches(days=7)"

            return formatted_results

        except Exception as e:
            print(f"[FAIL] Batch failed: {e}")
            raise

    def run_preprocessing(self, work_dir: str):
        """Run data preprocessing"""
        sys.path.insert(0, self.project_root)

        # 1. Patient basic information
        from base_select import process_patient_data
        process_patient_data(work_dir, work_dir)

        # 2. Merge lab results
        from result_itm import merge_item_data
        merge_item_data(work_dir, work_dir)

        # 3. Merge descriptive reports (preparing for LLM extraction)
        from desc_item import merge_desc_item
        desc_result = merge_desc_item(work_dir, work_dir)

        # 4. Merge descriptive reports + summary conclusions
        if desc_result:
            from desc_item_sum import merge_desc_with_summary
            merge_desc_with_summary(work_dir, work_dir)

        # 5. LLM structured extraction (only when descriptive reports exist)
        #    Append extraction results to item_result.csv; calculation scripts read from item_result.csv
        if desc_result:
            print("Step 2.5: LLM structured data extraction...")
            self.run_llm_extraction(work_dir)

    def run_llm_extraction(self, work_dir: str):
        """
        Run all LLM extraction scripts and append results directly to item_result.csv
        """
        llm_dataframes = []

        # Thyroid C-TIRADS feature extraction
        try:
            from llm_thyroid import run_thyroid_extraction
            result_df = run_thyroid_extraction(work_dir, work_dir)
            if result_df is not None:
                llm_dataframes.append(result_df)
        except Exception as e:
            print(f"  Thyroid LLM extraction failed: {e}")

        # Pulmonary imaging feature extraction
        try:
            from llm_lung import run_lung_extraction
            result_df = run_lung_extraction(work_dir, work_dir)
            if result_df is not None:
                llm_dataframes.append(result_df)
        except Exception as e:
            print(f"  Lung LLM extraction failed: {e}")

        # Breast density extraction
        try:
            from llm_breast_density import run_breast_density_extraction
            result_df = run_breast_density_extraction(work_dir, work_dir)
            if result_df is not None:
                llm_dataframes.append(result_df)
        except Exception as e:
            print(f"  Breast-density LLM extraction failed: {e}")

        # Liver stiffness (LSM/CAP) extraction
        try:
            from llm_liver_stiffness import run_liver_stiffness_extraction
            result_df = run_liver_stiffness_extraction(work_dir, work_dir)
            if result_df is not None:
                llm_dataframes.append(result_df)
        except Exception as e:
            print(f"  Liver-stiffness extraction failed: {e}")

        # Prostate volume extraction (pure regex, no LLM)
        try:
            from llm_prostate_volume import run_prostate_volume_extraction
            result_df = run_prostate_volume_extraction(work_dir, work_dir)
            if result_df is not None:
                llm_dataframes.append(result_df)
        except Exception as e:
            print(f"  Prostate-volume extraction failed: {e}")

        # Append all LLM results directly to item_result.csv
        if llm_dataframes:
            self.append_to_item_result(work_dir, llm_dataframes)

    def append_to_item_result(self, work_dir: str, dataframes: list):
        """Append the list of LLM-extracted DataFrames directly to item_result.csv"""
        item_result_path = os.path.join(work_dir, "item_result.csv")

        if not os.path.exists(item_result_path):
            print("  WARNING: item_result.csv not found; cannot append LLM results")
            return

        existing_df = pd.read_csv(item_result_path, encoding='utf-8-sig')
        appended_count = 0

        for llm_df in dataframes:
            if 'res_itm_name' in llm_df.columns and 'res_value' in llm_df.columns:
                # Align columns: fill in missing columns
                for col in existing_df.columns:
                    if col not in llm_df.columns:
                        llm_df[col] = ''
                llm_subset = llm_df[existing_df.columns]
                existing_df = pd.concat([existing_df, llm_subset], ignore_index=True)
                appended_count += len(llm_subset)

        if appended_count > 0:
            existing_df.to_csv(item_result_path, index=False, encoding='utf-8-sig')
            print(f"  [OK] appended {appended_count} LLM extraction results to item_result.csv")

    def run_models(self, work_dir: str):
        """
        Run model computation
        - Model scripts are read and executed from the original models_calc/ directory
        - Use work_dir (the batch directory) as cwd; model scripts read and write data there
        """
        self.run_model_scripts(work_dir)

    def run_model_scripts(self, work_dir: str):
        """Run all model scripts under the models_calc directory."""
        search_path = os.path.join(self.models_dir, "**", "*.py")
        model_scripts = glob.glob(search_path, recursive=True)
        model_scripts = [m for m in model_scripts if not m.endswith("__init__.py")]

        print(f"  found {len(model_scripts)} model scripts")

        for script in sorted(model_scripts):
            script_name = os.path.basename(script)
            print(f"  running model: {script_name}")

            try:
                subprocess_env = os.environ.copy()
                subprocess_env["PYTHONIOENCODING"] = "utf-8"
                # Models run from the batch directory still import shared utility modules from the project root.
                existing_pythonpath = subprocess_env.get("PYTHONPATH", "")
                subprocess_env["PYTHONPATH"] = os.pathsep.join(
                    part for part in (self.project_root, self.models_dir, existing_pythonpath) if part
                )
                # Run model scripts with the batch directory as cwd
                # Model scripts read "base.csv" and "item_result.csv" from the cwd
                # Generated result CSVs are also written to the cwd
                result = subprocess.run(
                    [sys.executable, os.path.abspath(script)],
                    cwd=work_dir,
                    check=True,
                    timeout=300,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    env=subprocess_env,
                )
            except subprocess.TimeoutExpired:
                print(f"    timed out; skipped")
            except subprocess.CalledProcessError as e:
                print(f"    execution failed: {e.stderr[:100]}")

    def collect_all_results(self, work_dir: str, app_nos: List[str]) -> List[Dict]:
        """
        Collect all model result CSVs from the batch directory
        All result CSVs live in work_dir (because cwd=work_dir)
        """
        results = []
        seen_results = set()

        # Known non-result files (raw database dumps + preprocessed files)
        skip_files = {
            "base.csv", "item_result.csv",
            "PES_APPLY.csv", "PES_PATIENT.csv", "PES_RESULT.csv",
            "PES_apply_item.csv", "PES_SUMMARY_COMBINE.csv", "PES_RESULT_DESC.csv",
            "desc_item.csv"
        }

        csv_files = glob.glob(os.path.join(work_dir, "*.csv"))

        for csv_file in csv_files:
            file_name = os.path.basename(csv_file)
            if file_name in skip_files:
                continue

            try:
                df = pd.read_csv(csv_file, encoding='utf-8-sig')
                for _, row in df.iterrows():
                    raw_app_no = str(row.get("app_no", "")).strip()
                    normalized_app_no = raw_app_no.lstrip('0')
                    normalized_requested = [no.lstrip('0') for no in app_nos]

                    if normalized_app_no in normalized_requested:
                        # Look up score/probability (covers both point-score and probability scales)
                        score = ""
                        score_columns = [
                            # Current unified output fields (preferred for new batches)
                            "predicted_score", "risk_probability",
                            # Absolute risk/probability fields (preferred over relative risk and total score)
                            "10-year_absolute_breast_cancer_risk (%)", "30-year absolute risk",
                            "10-year_CKD_probability", "5y_cancer_probability",
                            "5y_absolute_risk", "5y_kidney_failure_risk(%)",
                            "1y_cancer_probability(%)", "3-year_cancer_probability (%)",
                            "CIN3+ immediate risk (%)", "CIN3+ 1-year risk (%)", "CIN3+ 5-year risk (%)",
                            "copd_probability (%)", "asthma_probability (%)",
                            "overall_CVD_probability (P_CVD)",
                            "10y_heart_failure_risk", "10y_heart_failure_probability",
                            "VTE risk probability (6 months after cancer diagnosis and hospitalization)",
                            "prostate_cancer_risk (RC1)",
                            "prostate_cancer_risk", "high_grade_prostate_cancer_risk (%)",
                            "npc_probability", "acFibroMASH_probability",
                            # Legacy score field (for batches generated before field unification)
                            "predicted_score", "risk_score", "score", "screening_score",
                            "PRS313_raw_weighted_score", "Caprini 2010 total", "CHI-CAT total",
                            "assessment_total", "HFA_PEFF total", "IMPROVEDD total", "Legend total",
                            "METS-VF score", "NPC-CRS total",
                            "HCC-LSM_total_score", "total_score",
                            # Legacy probability field
                            "risk_probability", "risk_probability(%)", "risk_probability(%)",
                            "cancer_probability", "risk_probability(P)", "risk_value",
                            # Specific models
                            "cvd_risk", "chd_risk",
                            "aMAP_score", "Liver Cancer aMAP score",
                            "Lung Cancer Risk Score (LCRS)", "Gastric Cancer NGCSS Screening Score",
                            "ad8_score",
                            "total_malignancy_probability", "total_relative_risk(RR)",
                            "NPC_PRS_relative_ratio", "NPC_PRS_log_ratio",
                            "risk_score", "total_score",
                            # Point-score scale
                            "CHI-CAT total", "CKD-SCORED_score", "OSTA_index",
                            "SCORED_total", "osta_index",
                        ]
                        for col in score_columns:
                            if col in row and not self._is_missing_value(row[col]):
                                score = str(row[col])
                                break

                        # The STOP-Bang result CSV outputs only high-risk hits; the total is reconstructed from the number of hits
                        if self._is_missing_value(score) and "high_risk_hits" in row:
                            hits = str(row.get("high_risk_hits", "")).strip()
                            if not self._is_missing_value(hits):
                                hit_items = [item for item in re.split(r"[,，、;；]", hits) if item.strip()]
                                if hit_items:
                                    score = str(len(hit_items))

                        # Look up risk tier
                        risk_level = ""
                        for col in ["risk_level", "risk_level", "risk_grade",
                                    "risk_grade", "risk_tier", "risk_category", "risk_determination"]:
                            if col in row and not self._is_missing_value(row[col]):
                                risk_level = str(row[col])
                                break

                        # Look up code
                        risk_code = ""
                        for col in ["code", "risk_code", "code", "ICD_code", "disease_code"]:
                            if col in row and not self._is_missing_value(row[col]):
                                risk_code = str(row[col])
                                break

                        calculation_status = ""
                        for col in ["calculation_status", "assessment_status", "status"]:
                            if col in row and not self._is_missing_value(row[col]):
                                calculation_status = str(row[col])
                                break

                        # Fix calculation_status:
                        # 1) Score present but status empty -> computed successfully
                        # 2) No score and empty status -> not computable
                        # 3) Status says "computed" but no score exists -> not computable
                        score_missing = self._is_missing_value(score)
                        if not calculation_status:
                            calculation_status = "not_computable" if score_missing else "computed"
                        elif calculation_status == "computed" and score_missing:
                            calculation_status = "not_computable"

                        # Deduplicate
                        model_name = file_name.replace('.csv', '')
                        result_key = f"{normalized_app_no}:{model_name}"

                        if result_key not in seen_results:
                            seen_results.add(result_key)
                            results.append({
                                "app_no": raw_app_no,
                                "pat_id": str(row.get("pat_id", "")).strip(),
                                "model": model_name,
                                "score": score,
                                "risk_level": risk_level,
                                "risk_code": risk_code,
                                "calculation_status": calculation_status,
                                "source_table": self._build_source_table(row),
                                "source_page": str(row.get("source_page", "")) if str(row.get("source_page", "")) != "nan" else "",
                                "guideline_version": str(row.get("guideline_version", "")) if str(row.get("guideline_version", "")) != "nan" else "",
                                # Raw rows are used only for stacking score extraction and stripped before output
                                "raw_row": row.to_dict(),
                            })
            except Exception as e:
                print(f"    failed to read result file {file_name}: {e}")

        return results

    # Backward compatible: keep the class attribute (actually defined at module level as risk_level_priority)
    RISK_LEVEL_PRIORITY = RISK_LEVEL_PRIORITY

    def _get_risk_priority(self, risk_level: str) -> int:
        """Get the risk-level priority (delegates to the module-level function so callers share one criterion)."""
        return risk_level_priority(risk_level)

    def _is_valid_model_result(self, result: Dict) -> bool:
        """Only model results with a real risk tier and no explicit failure may be shown."""
        status = str(result.get("calculation_status", "") or "").strip()
        invalid_status_tokens = ("failed", "cannot", "missing", "not_computed", "skipped", "not_applicable")
        if status and any(token in status for token in invalid_status_tokens):
            return False
        return self._get_risk_priority(str(result.get("risk_level", ""))) > 0

    def _is_computable_model_result(self, result: Dict) -> bool:
        """
        Check whether a model result actually produced a value.
        Score takes precedence: a valid score with no explicit failure status is enough;
        A tier may be "continuous risk" or "-"; this does not prevent it from being computable.
        """
        status = str(result.get("calculation_status", "") or "").strip()
        invalid_status_tokens = ("failed", "cannot", "missing", "not_computed", "skipped", "not_applicable")
        if status and any(token in status for token in invalid_status_tokens):
            return False
        return not self._is_missing_value(result.get("score", ""))

    def _resolve_disease_risk(self, assessments: List[Dict]) -> str:
        """
        Decide the final risk tier from multiple model results for the same disease.
        Rules:
        1. Majority vote (the tier with the most votes wins)
        2. On equal votes, take the higher tier
        """
        from collections import Counter

        # Drop invalid results ("-", empty, not computable, etc.)
        valid_levels = [
            a["risk_level"] for a in assessments
            if a["risk_level"] and self._get_risk_priority(a["risk_level"]) > 0
        ]

        if not valid_levels:
            return "-"

        # Count votes per risk tier
        counter = Counter(valid_levels)

        # Sort by (votes desc, risk priority desc) and take the first
        sorted_levels = sorted(
            counter.keys(),
            key=lambda level: (counter[level], self._get_risk_priority(level)),
            reverse=True
        )

        return sorted_levels[0]

    @staticmethod
    def _public_model_result(result: Dict) -> Dict:
        """Model results for output: drop raw rows used only for stacking score extraction."""
        return {
            "model": result["model"],
            "risk_level": result["risk_level"],
            "score": result.get("score", ""),
            "calculation_status": result.get("calculation_status", ""),
            "source_table": result.get("source_table", {}),
            "source_page": result.get("source_page", ""),
            "guideline_version": result.get("guideline_version", ""),
        }

    def format_results(self, raw_results: List[Dict], requested_app_nos: List[str]) -> List[Dict]:
        """
        Format results: aggregate by disease and split into computable and not-computable parts.
        - Computable part: diseases where at least one model actually produced a score; with the stacking integrated score attached.
          A tier may be "continuous risk" or "-"; this does not prevent it from being computable.
        - Not-computable part: diseases where no model produced a score.
        Within a disease, models that produced no result go into failed_model_results for traceability.
        """
        grouped = {}
        normalized_requested = {no.lstrip('0'): no for no in requested_app_nos}

        for result in raw_results:
            normalized_app_no = result["app_no"].lstrip('0')
            disease_name = self._code_to_disease_name(result["risk_code"])

            if normalized_app_no not in grouped:
                requested_format = normalized_requested.get(normalized_app_no, result["app_no"])
                grouped[normalized_app_no] = {
                    "app_no": requested_format,
                    "pat_id": result["pat_id"],
                    "disease_assessments": {},  # {disease_name: [model results]}
                    "raw_results": [],          # all model results for this patient, for stacking lookup by file name
                }

            grouped[normalized_app_no]["raw_results"].append(result)

            # Skip results without a disease name
            if disease_name in ("-", "", "N/A"):
                continue

            if disease_name not in grouped[normalized_app_no]["disease_assessments"]:
                grouped[normalized_app_no]["disease_assessments"][disease_name] = []

            # Keep full results (including raw_row) for stacking score extraction; strip them on output
            grouped[normalized_app_no]["disease_assessments"][disease_name].append(result)

        # Vote per disease to produce the final result
        final_results = []
        for app_no in requested_app_nos:
            normalized = app_no.lstrip('0')
            if normalized in grouped:
                patient = grouped[normalized]
                computable = []
                non_computable = []

                for disease_name, model_results in patient["disease_assessments"].items():
                    valid_model_results = [
                        result for result in model_results
                        if self._is_computable_model_result(result)
                    ]
                    invalid_model_results = [
                        result for result in model_results
                        if not self._is_computable_model_result(result)
                    ]

                    if valid_model_results:
                        final_risk = self._resolve_disease_risk(valid_model_results)
                        # stacking takes scores from all of the patient's model results by file name:
                        # Some models (e.g. LAP) have a score but code "-", so they are not grouped under a disease,
                        # Still used as stacking input.
                        stacking_results = self.stacking_calculator.compute_for_disease(
                            disease_name, patient["raw_results"]
                        )
                        assessment = {
                            "disease_name": disease_name,
                            "english_name": _english_name(disease_name),
                            "risk_level": final_risk,
                            "model_results": [
                                self._public_model_result(result)
                                for result in valid_model_results
                            ],
                            "stacking_results": stacking_results,
                        }
                        if not stacking_results:
                            assessment["stacking_status"] = "no matching stacking model"
                        if invalid_model_results:
                            assessment["failed_model_results"] = [
                                self._public_model_result(result)
                                for result in invalid_model_results
                            ]
                        computable.append(assessment)
                    else:
                        non_computable.append({
                            "disease_name": disease_name,
                            "english_name": _english_name(disease_name),
                            "model_results": [
                                self._public_model_result(result)
                                for result in model_results
                            ],
                        })

                computable.sort(
                    key=lambda item: self._get_risk_priority(item["risk_level"]),
                    reverse=True,
                )
                non_computable.sort(key=lambda item: item["disease_name"])

                final_results.append({
                    "app_no": patient["app_no"],
                    "pat_id": patient["pat_id"],
                    "computable": computable,
                    "not_computable": non_computable,
                })
            else:
                final_results.append({
                    "app_no": app_no,
                    "pat_id": "",
                    "computable": [],
                    "not_computable": [],
                })

        return final_results

    def cleanup_batch(self, batch_dir: str):
        """
        Clean intermediate files in a single batch directory, keeping only the final result JSON
        """
        keep_files = {"batch_results.json"}

        for file_path in glob.glob(os.path.join(batch_dir, "*")):
            if os.path.basename(file_path) not in keep_files:
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception:
                    pass

    def cleanup_old_batches(self, days: int = 7):
        """
        Clean batch directories older than the given number of days
        Usage: python -c "from batch_processor import BatchProcessor; BatchProcessor().cleanup_old_batches(days=7)"
        """
        batches_dir = os.path.join(self.project_root, "batches")
        if not os.path.exists(batches_dir):
            print("no batch directory")
            return

        cutoff = time.time() - (days * 86400)
        removed = 0

        for batch_name in os.listdir(batches_dir):
            batch_path = os.path.join(batches_dir, batch_name)
            if os.path.isdir(batch_path):
                mtime = os.path.getmtime(batch_path)
                if mtime < cutoff:
                    shutil.rmtree(batch_path)
                    removed += 1
                    print(f"  deleted: {batch_name}")

        print(f"cleanup complete: removed {removed} batch directories older than {days} days")
