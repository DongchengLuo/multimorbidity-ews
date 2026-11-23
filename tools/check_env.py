#!/usr/bin/env python3
"""Migration self-check: verify data, dependencies and configuration on a new server.

Usage:
    python tools/check_env.py

Exit code 0 means ready to start; 1 means blocking issues exist (listed one by one).
"""
from __future__ import annotations

import importlib
import os
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OK, WARN, FAIL = "  [OK]  ", "  [WARN]", "  [FAIL]"
problems: list[str] = []
warnings: list[str] = []


IS_WINDOWS = platform.system() == "Windows"


def check_python() -> None:
    v = sys.version_info
    if v >= (3, 10):
        print(f"{OK} Python {v.major}.{v.minor}.{v.micro} on {platform.system()}")
    else:
        print(f"{FAIL} Python {v.major}.{v.minor} is too old; 3.10+ required")
        problems.append("Python version too old")
    if IS_WINDOWS:
        print(f"{OK} Windows mode: start with run_server.bat")


def check_deps() -> None:
    mods = {
        "flask": "flask",
        "flask_cors": "flask-cors",
        "pandas": "pandas",
        "numpy": "numpy",
        "scipy": "scipy",
        "networkx": "networkx",
        "matplotlib": "matplotlib",
        "dotenv": "python-dotenv",
        "requests": "requests",
        "openai": "openai",
        "tqdm": "tqdm",
        "PIL": "Pillow",
    }
    missing = []
    for mod, pkg in mods.items():
        try:
            importlib.import_module(mod)
        except Exception:
            missing.append(pkg)
    if missing:
        print(f"{FAIL} Missing dependencies: {', '.join(missing)}")
        print("         pip install -r requirements.txt")
        problems.append("Missing Python dependencies")
    else:
        print(f"{OK} Core dependencies present")

    # Optional dependencies
    try:
        importlib.import_module("weasyprint")
        print(f"{OK} weasyprint available (PDF rendering)")
    except Exception:
        print(f"{WARN} weasyprint unavailable; PDF rendering will fail (set REPORT_RENDER_PDF=0)")
        if IS_WINDOWS:
            print("         On Windows, weasyprint also needs the GTK3 runtime;")
            print("         to avoid the hassle, set REPORT_RENDER_PDF=0 in .env and output HTML only.")
        warnings.append("weasyprint missing")
    try:
        importlib.import_module("pymssql")
        print(f"{OK} pymssql available (direct database mode)")
    except Exception:
        print(f"{WARN} pymssql unavailable; only affects SNAPSHOT_MODE=0 (snapshot mode does not need it)")


def check_files() -> None:
    snapshot = ROOT / "data" / "snapshot"
    need = [
        "PES_APPLY.csv",
        "PES_PATIENT.csv",
        "PES_RESULT.csv",
        "PES_apply_item.csv",
        "PES_RESULT_DESC.csv",
        "PES_SUMMARY_COMBINE.csv",
        "manifest.json",
    ]
    miss = [n for n in need if not (snapshot / n).exists()]
    if miss:
        print(f"{FAIL} data/snapshot missing: {', '.join(miss)}")
        problems.append("Snapshot data incomplete")
    else:
        print(f"{OK} data/snapshot complete ({len(need)} files)")

    kg = ROOT / "data" / "kg"
    kg_files = [
        "knowledge_graph_normalized_with_weights.json",
        "comorbidity_combinations_S2.csv",
    ]
    miss = [n for n in kg_files if not (kg / n).exists()]
    if miss:
        print(f"{FAIL} data/kg missing: {', '.join(miss)}")
        print("         Report generation / subgraph retrieval need these two files (~114 MB)")
        problems.append("Knowledge-graph data missing")
    else:
        size = sum((kg / n).stat().st_size for n in kg_files) / 1024 / 1024
        print(f"{OK} data/kg complete ({size:.0f} MB)")

    if IS_WINDOWS:
        longest = max((len(str(p)) for p in ROOT.rglob("*")), default=0)
        if longest > 240:
            print(f"{WARN} Longest path is {longest} characters, close to the Windows 260 limit")
            print("         Put the project on a short path (e.g. C:\\app) or enable LongPathsEnabled")
            warnings.append("Path too long")

    icons = ROOT / "ui_icons"
    if icons.is_dir() and any(icons.iterdir()):
        print(f"{OK} ui_icons directory present ({len(list(icons.iterdir()))} entries)")
    else:
        print(f"{WARN} ui_icons directory missing or empty; report icons will degrade")
        warnings.append("Icon library missing")


def check_env() -> None:
    env = ROOT / ".env"
    if not env.exists():
        print(f"{WARN} .env not found; using defaults (SNAPSHOT_MODE=1)")
        print("         cp .env.example .env")
        warnings.append(".env missing")
        return
    try:
        from dotenv import dotenv_values
    except Exception:
        print(f"{WARN} Cannot parse .env (python-dotenv not installed)")
        return
    cfg = dotenv_values(env)
    mode = (cfg.get("SNAPSHOT_MODE") or "1").strip()
    print(f"{OK} .env loaded, SNAPSHOT_MODE={mode}")

    url = (cfg.get("LOCAL_LLM_API_URL") or "").strip()
    model = (cfg.get("LOCAL_LLM_MODEL_NAME") or "").strip()
    if url and model:
        print(f"{OK} LLM configured: {model} @ {url}")
    else:
        print(f"{WARN} LLM not configured (LOCAL_LLM_API_URL / LOCAL_LLM_MODEL_NAME empty)")
        print("         Browsing and search work; fill these in before generating reports or scoring edges")
        warnings.append("LLM not configured")


def main() -> int:
    print("=" * 62)
    print(" Migration self-check - Multimorbidity Early Warning System")
    print(f" Project root: {ROOT}")
    print("=" * 62)
    check_python()
    check_deps()
    check_files()
    check_env()
    print("-" * 62)
    if problems:
        print(f"Blocking issues ({len(problems)}): " + "; ".join(problems))
        print("Resolve the [FAIL] items above before starting the service.")
        return 1
    if warnings:
        print(f"Warnings ({len(warnings)}): " + "; ".join(warnings))
    launcher = "run_server.bat 5000" if IS_WINDOWS else "./run_server.sh 5000"
    print(f"Self-check passed. Run: {launcher}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
