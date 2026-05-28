import pandas as pd
import os
import glob
from pathlib import Path
from datetime import datetime

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
RAW  = ROOT / "data" / "raw"
OUT  = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

# ── Find most recent pull folder ─────────────────────────────────────────────
folders = sorted([f for f in RAW.iterdir() if f.is_dir()])
latest  = folders[-1]
print(f"Reading from: {latest.name}")

# ── Read all parquet files ───────────────────────────────────────────────────
files = sorted(latest.glob("fyear_*.parquet"))
df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
print(f"Raw rows: {len(df):,}")

# ── Clean ────────────────────────────────────────────────────────────────────
df = df.drop_duplicates()
df = df.dropna(subset=["gvkey", "fyear"])

# Convert object columns to numeric where possible
for col in df.select_dtypes(include="object").columns:
    try:
        df[col] = pd.to_numeric(df[col], errors="ignore")
    except:
        pass

df = df.sort_values(["gvkey", "fyear"]).reset_index(drop=True)
print(f"Clean rows: {len(df):,}")
print(f"Columns:    {len(df.columns)}")

# ── Save ─────────────────────────────────────────────────────────────────────
out_file = OUT / "panel_clean.parquet"
df.to_parquet(out_file, index=False)
print(f"Saved to:   {out_file}")

# ── Log ──────────────────────────────────────────────────────────────────────
log = f"""Clean log
=========
Date:        {datetime.now().isoformat()}
Source:      {latest.name}
Raw rows:    {len(df):,}
Clean rows:  {len(df):,}
Columns:     {len(df.columns)}
"""
(OUT / "clean_log.txt").write_text(log)
print("Done.")
