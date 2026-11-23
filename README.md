# Multimorbidity Early Warning System — English Demo

A knowledge graph-enhanced multimodal early warning system for personalized
disease prevention in multimorbidity. It ships with a **fully synthetic,
de-identified English snapshot** and runs entirely offline against local CSV
files — no database connection is required.

Pipeline: clinical indicators → 31-dimension comorbidity risk vector → risk-driven
subgraph retrieval from a literature-derived knowledge graph → LLM-generated
structured prevention report.

> This is a demonstration build. All patient records, names, addresses, lab
> values and report texts are generated or resampled from distributions; they do
> **not** correspond to any real person.

## Highlights

- **English UI** — every screen, label, table header, tooltip and message is in English.
- **English demo data** — field names and text values (item names, departments,
  descriptions, summaries, units, reference ranges) are translated to English.
- **Local snapshot mode** — `SNAPSHOT_MODE=1` reads `data/snapshot/*.csv`; the
  original SQL Server path is disabled and all credentials are blank.
- **Paper-aligned naming** — module names follow the system's positioning as a
  multimorbidity early warning system (risk vector, subgraph retrieval, prevention report).

## Modules

| Module | Description |
| --- | --- |
| **Patient Health Profile Explorer** | Browse the preloaded 5,000-subject snapshot with basic info and linked lab/exam details. |
| **Clinical Indicator Search** | Combine basic-info filters with lab/exam field conditions (proximity-matched field picker, result/unit preview) to build the risk vector input. |
| **Patient Intake Monitor** | Watch a directory of `app_no` list files and sync matching basic info into the local store. |
| **Prevention Report Center** | View generated prevention reports (visual json-render and text). |
| **Early Warning Pipeline** | Select subjects, run risk scoring / subgraph retrieval / report generation and track progress. |

## Quick start

```bash
# 1) install dependencies (Python 3.10+)
pip install -r requirements.txt

# 2) create your config
cp .env.example .env      # then edit LLM settings if you need report generation

# 3) self-check data / deps / config
python tools/check_env.py

# 4) start the server (default port 5001; pass a port to override)
./run_server.sh 5000
# open http://127.0.0.1:5000
```

`run_server.sh` uses `$PYTHON` if set, otherwise `python3`/`python` from PATH.
The server reads `.env`; with `SNAPSHOT_MODE=1` it loads `data/snapshot/` on the
first request (about 6–15 s) and caches it in memory.

## Migrating to another server

The repository is self-contained: snapshot data (`data/snapshot/`, ~200 MB) and
the knowledge graph (`data/kg/`, ~114 MB) are both inside the project, so a
`git clone` (or a plain directory copy) is enough to move it.

Steps on the new host:

1. **Copy the project** — `git clone <repo>` or `rsync -a` the whole directory.
2. **Install Python 3.10+ and dependencies** — `pip install -r requirements.txt`.
   For PDF rendering, `weasyprint` also needs system libraries:
   ```bash
   # Debian/Ubuntu
   sudo apt-get install -y libpango-1.0-0 libpangoft2-1.0-0 libcairo2 libgdk-pixbuf-2.0-0
   # RHEL/CentOS
   sudo yum install -y pango cairo gdk-pixbuf2
   ```
   If you do not need PDF, set `REPORT_RENDER_PDF=0` in `.env`.
3. **Configure the LLM** — edit `.env`:
   ```ini
   LOCAL_LLM_API_URL=http://<your-host>:<port>/v1/chat/completions
   LOCAL_LLM_MODEL_NAME=<your-model-name>
   LOCAL_LLM_API_KEY=EMPTY        # or your real key
   ```
   Any OpenAI-compatible endpoint works (vLLM, SGLang, Ollama `/v1`, OpenAI, …).
   Without this, browsing/search work but report generation and edge scoring
   will stop with a clear "LLM not configured" error.
4. **Run the self-check** — `python tools/check_env.py` (exit code 0 = ready).
5. **Start** — `./run_server.sh 5000`.

Optional overrides (only if you keep the big files elsewhere):

```ini
KG_DIR=/data/kg
KG_PATH=/data/kg/knowledge_graph_normalized_with_weights.json
KG_S2_PATH=/data/kg/comorbidity_combinations_S2.csv
ICON_ROOT=/path/to/ui_icons
```

Notes:

- No hard-coded absolute paths remain; all data paths resolve relative to the
  project root and can be overridden by the variables above.
- `SNAPSHOT_MODE=1` (default) needs no database. Set `SNAPSHOT_MODE=0` and fill
  `MEDICAL_DB_*` only if you want to connect to a real SQL Server.
- The bundled Flask dev server is fine for demos; for production put it behind
  gunicorn/waitress, e.g. `gunicorn -w 2 -b 0.0.0.0:5000 app:app`.

## Windows server

Yes — the project runs on Windows Server 2016+ / Windows 10+ with Python 3.10+.
The code is already platform-neutral: no `fcntl`, no `os.fork`, no shell
scripts in the runtime path, all paths built with `os.path.join`, and task IDs
use `YYYYmmdd_HHMMSS` (no `:` in filenames).

```bat
REM 1) install Python 3.10+ (python.org installer, tick "Add to PATH")
REM 2) install dependencies
pip install -r requirements.txt

REM 3) create config
copy .env.example .env
REM    edit .env: LOCAL_LLM_API_URL / LOCAL_LLM_MODEL_NAME

REM 4) self-check
python tools\check_env.py

REM 5) start (default port 5001; pass a port to override)
run_server.bat 5000
```

Windows-specific notes:

| Topic | Guidance |
| --- | --- |
| **Launcher** | Use `run_server.bat` (bash `run_server.sh` will not run in cmd). It sets `PYTHONUTF8=1` so Chinese log lines do not crash a cp936 console. |
| **PDF rendering** | `weasyprint` needs the GTK3 runtime on Windows. If you do not need PDF, set `REPORT_RENDER_PDF=0` in `.env` — HTML reports work without it. |
| **pymssql** | Only needed for `SNAPSHOT_MODE=0`. It is imported lazily, so snapshot mode starts without it. Prebuilt wheels exist for CPython 3.9–3.13. |
| **Long paths** | Some bundled icon filenames are long. Put the project at a short path (e.g. `C:\app`) or enable `LongPathsEnabled` in the registry to stay under the 260-character limit. |
| **Production** | Use `waitress` instead of the dev server: `pip install waitress` then `waitress-serve --port=5000 app:app`. |

## Demo data

`data/snapshot/` contains six CSV files plus `manifest.json`:

| File | Rows | Content |
| --- | --- | --- |
| `PES_APPLY.csv` | 5,000 | Screening applications |
| `PES_PATIENT.csv` | 5,000 | Patient basic information |
| `PES_RESULT.csv` | 638,607 | Lab results |
| `PES_apply_item.csv` | 666,086 | Applied lab/exam items |
| `PES_RESULT_DESC.csv` | 27,480 | Descriptive exam reports |
| `PES_SUMMARY_COMBINE.csv` | 5,000 | Combined summaries |

De-identification policy:

- `app_no` → synthetic contiguous range `90000001`–`90005000`
- `pat_id` → `P900000001`–`P900005000`
- `pat_name` / `app_pat_name` → `Subject NNNN`
- `pat_address` → `No. N Virtual Street, Demo District, Demo City, Demo Province`
- `pat_nation` → `Han`; ID card / mobile / email / photo / contact fields are blank
- Lab values are resampled from learned distributions; description text comes
  from a built-in template library

### Regenerating the snapshot

```bash
# rebuild the synthetic snapshot from scratch (optional)
python tools/build_snapshot.py --count 5000 --seed 20261110
```

`tools/build_snapshot.py` writes the six CSVs plus `manifest.json` into
`data/snapshot/`. English text values are produced by the built-in
template/dictionary layer inside that script, so no separate translation step
is required.

## Configuration

See `.env.example` for the full annotated template. In snapshot mode the
`MEDICAL_DB_*` settings are blank. The LLM settings (`LOCAL_LLM_*`,
`REPORT_LLM_*`) are only used by the report pipeline; the web UI and snapshot
browsing work without them. `REPORT_LLM_*` timeouts/concurrency should be tuned
to your model's throughput. `REPORT_LANGUAGE` controls report output language
(`en` default, `zh` for Chinese).

## Notes / limitations

- Report generation requires a configured LLM endpoint (see above). Without it
  the web UI and snapshot browsing still work, but no new reports are produced.
- The knowledge graph (`data/kg/`) keeps the **original Chinese summaries**
  extracted from the source clinical guidelines. They serve as retrieval
  evidence and are not shown verbatim in the English UI. Translating them would
  change retrieval behaviour, so they are intentionally left as-is.
- All Python source, comments, docstrings and user-facing strings are English.

## Data availability

This repository contains **no real patient data**. The bundled snapshot in
`data/snapshot/` is fully synthetic: identifiers are replaced by a contiguous
range (`90000001`-`90005000`), names by `Subject NNNN`, addresses by
placeholder strings, and all direct identifiers (ID card, mobile, email, photo,
contact) are blank. Lab values are resampled from learned distributions and
descriptive reports come from a template library.

The knowledge graph in `data/kg/` is derived from publicly available clinical
guidelines; it stores short paraphrased summaries rather than verbatim text.
Please observe the terms of the original guideline sources when reusing it.

## License

Released under the MIT License - see `LICENSE`.

## Citation

If you use this code or the synthetic snapshot, please cite the accompanying
paper (citation details to be added upon publication) and this repository.
