"""
02_clean.py
-----------
Build a clean firm-year panel from the raw WRDS pull.
Automatically picks the most recent pull folder.

Output
------
data/processed/panel_clean.parquet
data/processed/clean_log.txt

Usage
-----
    python code/02_clean.py
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
RAW  = ROOT / "data" / "raw"
OUT  = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

# ── Find most recent pull folder ──────────────────────────────────────────────
folders = sorted([f for f in RAW.iterdir() if f.is_dir()])
latest  = folders[-1]
print(f"Reading from: {latest.name}")

# ── Read all parquet chunks ───────────────────────────────────────────────────
files = sorted(latest.glob("fyear_*.parquet"))
print(f"Reading {len(files)} files...")

chunks = []
for f in files:
    chunk = pd.read_parquet(f)
    chunks.append(chunk)
    print(f"  {f.name:<25}  {len(chunk):>6,} rows")

df = pd.concat(chunks, ignore_index=True)
print(f"\nCombined: {df.shape[0]:,} rows x {df.shape[1]} columns")

# ── Drop duplicates ───────────────────────────────────────────────────────────
n_before = len(df)
df = df.drop_duplicates()
df = df.drop_duplicates(subset=["gvkey", "fyear"], keep="first")
print(f"\nAfter dedup: {len(df):,} rows (dropped {n_before - len(df):,})")

# ── Drop rows missing panel identifiers ───────────────────────────────────────
n_before = len(df)
df = df.dropna(subset=["gvkey", "fyear"])
print(f"After dropping missing gvkey/fyear: {len(df):,} rows (dropped {n_before - len(df):,})")

# ── Convert object columns to numeric (protect string cols) ──────────────────
STRING_COLS = {
    "gvkey", "conm", "cusip", "isin", "sedol", "tic",
    "naics", "sic", "loc", "curcd", "fic", "exchg",
    "costat", "stalt", "datafmt", "indfmt", "popsrc", "consol"
}

converted = []
for col in df.columns:
    if df[col].dtype == object and col not in STRING_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        converted.append(col)
print(f"Converted {len(converted)} object columns to numeric")

# ── Ensure fyear is integer ───────────────────────────────────────────────────
df["fyear"] = df["fyear"].astype(int)

# ── Sort into firm-year panel ─────────────────────────────────────────────────
df = df.sort_values(["gvkey", "fyear"]).reset_index(drop=True)

# ── SME filter: EU definition (<250 employees) ────────────────────────────────
n_before = len(df)
if "emp" in df.columns:
    df = df[df["emp"] < 0.25].copy()
    print(f"\nSME filter (emp < 0.25): {len(df):,} rows (dropped {n_before - len(df):,})")

# ── Compute derived variables ─────────────────────────────────────────────────
# Y: Return on Assets
df["roa"] = df["nicon"] / df["at"]

# X: R&D Intensity (missing XRD -> firm did not report, treat as 0)
df["xrd"] = df["xrd"].fillna(0)
df["rd_intensity"] = df["xrd"] / df["at"]

# Controls
df["log_at"]   = np.log(df["at"].replace(0, np.nan))
df["leverage"] = (df["dltt"] + df["dlc"]) / df["seq"]
df["sales_growth"] = (df.sort_values("fyear")
                        .groupby("gvkey")["sale"]
                        .pct_change(fill_method=None))

print(f"\nDerived variables added: roa, rd_intensity, log_at, leverage, sales_growth")

# ── Missing value summary for key variables ───────────────────────────────────
key_vars = ["at", "nicon", "xrd", "emp", "dltt", "dlc", "seq", "sale",
            "roa", "rd_intensity", "log_at", "leverage"]
available = [v for v in key_vars if v in df.columns]

print("\nCompleteness of key variables:")
for col in available:
    pct = df[col].notna().sum() / len(df) * 100
    bar = "█" * int(pct / 5)
    print(f"  {col:<15}  {pct:>5.1f}%  {bar}")

# ── Panel statistics ──────────────────────────────────────────────────────────
print("\n" + "="*45)
print("Panel Statistics")
print("="*45)
print(f"  Total firm-years:   {len(df):>8,}")
print(f"  Unique firms:       {df['gvkey'].nunique():>8,}")
print(f"  Years covered:      {df['fyear'].min()}–{df['fyear'].max()}")
if "loc" in df.columns:
    print(f"  Countries:          {df['loc'].nunique():>8,}")
print(f"  Total columns:      {df.shape[1]:>8,}")

print("\n  Observations per year:")
for year, count in df.groupby("fyear").size().items():
    print(f"    {year}: {count:>6,}")

# ── Save ──────────────────────────────────────────────────────────────────────
out_file = OUT / "panel_clean.parquet"
df.to_parquet(out_file, index=False)
size_mb = out_file.stat().st_size / 1_048_576
print(f"\nSaved: {out_file}")
print(f"Size:  {size_mb:.2f} MB")
print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")

# ── Verify read-back ──────────────────────────────────────────────────────────
check = pd.read_parquet(out_file)
assert check.shape == df.shape, "Read-back shape mismatch!"
print(f"Verified: read-back matches ({check.shape[0]:,} rows x {check.shape[1]} columns)")

# ── Write log ─────────────────────────────────────────────────────────────────
log = f"""Clean log
=========
Date:        {datetime.now().isoformat()}
Source:      {latest.name}
Raw rows:    {len(pd.concat(chunks, ignore_index=True)):,}
Clean rows:  {len(df):,}
Columns:     {len(df.columns)}
SME filter:  emp < 0.25 (EU definition, <250 employees)
"""
(OUT / "clean_log.txt").write_text(log)
print("Done.")
