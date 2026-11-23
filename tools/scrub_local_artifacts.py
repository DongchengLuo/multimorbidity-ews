#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scrub_local_artifacts.py - remove residual real identifiers from the repository

Processing target
--------
1. watched_app_nos/*.txt        - real app_no manifest, replaced with snapshot app_no
2. watched_app_nos/synced_base_info.csv - real names/addresses, rebuilt from the snapshot
3. batches/*/base.csv           - real names/addresses/ethnicity, de-identified in place

Notes
----
app_no / pat_id in the CSVs under batches/ are the batch directory name and the cross-table join keys,
Re-identification would break directory structure and historical artifacts, so only human-readable
Name, address and ethnicity columns; app_no / pat_id keep their original values.
Re-identifying app_no as well requires separate assessment (it affects all paths under batches/ and debug_output/).

Usage
----
    python tools/scrub_local_artifacts.py
    python tools/scrub_local_artifacts.py --dry-run
"""

import argparse
import glob
import hashlib
import os
import shutil

import pandas as pd


def _stable_int(text, mod):
    """Stable hash (consistent across processes) so pseudonyms do not change between runs."""
    h = hashlib.md5(str(text).encode("utf-8")).hexdigest()
    return int(h[:8], 16) % mod

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WATCH_DIR = os.path.join(ROOT, "watched_app_nos")
SNAPSHOT_DIR = os.path.join(ROOT, "data", "snapshot")
BACKUP_SUFFIX = ".orig"


def _backup(path):
    if os.path.exists(path) and not os.path.exists(path + BACKUP_SUFFIX):
        shutil.copy2(path, path + BACKUP_SUFFIX)


def scrub_watch(dry_run=False):
    snap = pd.read_csv(os.path.join(SNAPSHOT_DIR, "PES_PATIENT.csv"),
                       dtype=str, low_memory=False).fillna("")
    app = pd.read_csv(os.path.join(SNAPSHOT_DIR, "PES_APPLY.csv"),
                      dtype=str, low_memory=False).fillna("")
    base = app[["app_no", "pat_id", "app_pat_name", "app_pat_age"]].merge(
        snap[["pat_id", "pat_name", "pat_sex", "pat_nation", "pat_address"]],
        on="pat_id", how="inner",
    ).drop_duplicates(subset=["app_no"])

    # Take the first 79 entries to match the original list size
    picked = base.head(79).reset_index(drop=True)
    nos = picked["app_no"].tolist()

    txt_path = os.path.join(WATCH_DIR, "untitled_document_2.txt")
    proc_path = os.path.join(WATCH_DIR, ".processed_app_nos.txt")
    csv_path = os.path.join(WATCH_DIR, "synced_base_info.csv")

    print(f"[watch] manifest {len(nos)} entries -> {os.path.basename(txt_path)}")
    if not dry_run:
        _backup(txt_path)
        _backup(proc_path)
        _backup(csv_path)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(nos) + "\n")
        with open(proc_path, "w", encoding="utf-8") as f:
            f.write("\n".join(nos) + "\n")

        out = picked.copy()
        out["source_txt"] = os.path.basename(txt_path)
        out["sync_time"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        out.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"[watch] basic-info table {len(out)} rows -> synced_base_info.csv")


def scrub_batches(dry_run=False):
    files = sorted(glob.glob(os.path.join(ROOT, "batches", "*", "base.csv")))
    print(f"[batches] {len(files)} base.csv to process")
    total = 0
    for path in files:
        try:
            df = pd.read_csv(path, dtype=str, low_memory=False).fillna("")
        except Exception as e:
            print(f"  WARNING: skipping {path}: {e}")
            continue
        if df.empty:
            continue

        # Batches already in the snapshot range (app_no >= 90000001) need no re-de-identification
        if "app_no" in df.columns:
            nos = pd.to_numeric(df["app_no"], errors="coerce")
            if nos.notna().all() and (nos >= 90000001).all():
                continue

        changed = False
        for col in ("pat_name", "app_pat_name"):
            if col in df.columns:
                new = [
                    f"Subject{_stable_int(v, 9000) + 1000:04d}" if str(v).strip() else ""
                    for v in df[col]
                ]
                if new != df[col].tolist():
                    df[col] = new
                    changed = True
        if "pat_address" in df.columns:
            df["pat_address"] = [
                f"No. {_stable_int(v, 900) + 1}, Virtual Street, Demo District, Demo City, Demo Province"
                if str(v).strip() else ""
                for v in df["pat_address"]
            ]
            changed = True
        if "pat_nation" in df.columns:
            df["pat_nation"] = "Han"
            changed = True

        if changed:
            total += len(df)
            if not dry_run:
                _backup(path)
                df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[batches] de-identified {total} rows")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    scrub_watch(args.dry_run)
    scrub_batches(args.dry_run)
    print("Done" + (" (dry-run, nothing written)" if args.dry_run else ""))
