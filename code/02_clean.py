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

# ── SME filter (EU definition: fewer than 250 employees) ─────────────────────
if "emp" in df.columns:
    df = df[df["emp"] < 0.25]
    print(f"SME rows (emp < 0.25): {len(df):,}")

# ── Compute derived variables ─────────────────────────────────────────────────
# Y: Return on Assets
df["roa"] = df["nicon"] / df["at"]

# X: R&D Intensity
df["rd_intensity"] = df["xrd"] / df["at"]

# Controls
import numpy as np
df["log_at"] = np.log(df["at"].replace(0, float("nan")))
df["leverage"] = (df["dltt"] + df["dlc"]) / df["seq"]
df["sales_growth"] = df.sort_values("fyear").groupby("gvkey")["sale"].pct_change()

# ── Save updated panel ────────────────────────────────────────────────────────
df.to_parquet(out_file, index=False)
print(f"Variables added: roa, rd_intensity, log_at, leverage, sales_growth")
