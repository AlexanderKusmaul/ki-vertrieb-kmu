"""
03_descriptives.py
------------------
Build descriptive statistics and exploratory figures for the
R&D intensity research design.

Output
------
output/tables/summary_statistics.csv
output/figures/correlation_matrix.png
output/figures/dv_distribution.png
output/figures/main_relationship.png
data/processed/panel_with_vars.parquet

Usage
-----
    python code/03_descriptives.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from pathlib import Path
from scipy.stats import mstats

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).resolve().parents[1]
PROC    = ROOT / "data" / "processed"
TABLES  = ROOT / "output" / "tables"
FIGURES = ROOT / "output" / "figures"
TABLES.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

# ── Load panel ────────────────────────────────────────────────────────────────
df = pd.read_parquet(PROC / "panel_clean.parquet")
print(f"Loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")

# ── Data quality filters ──────────────────────────────────────────────────────
df = df[(df["at"] > 0.1) & (df["sale"] > 0) & (df["seq"] > 0)].copy()
df = df[df["at"] >= 1].copy()  # remove micro-firms with negative log(at)

# EU SME filter
sme_mask = (df["emp"] < 0.25) | (df["at"] <= 43)
df = df[sme_mask].copy()
print(f"After filters: {df.shape[0]:,} rows")

# ── Variable construction ─────────────────────────────────────────────────────
# Dependent variable (Y)
df["roa"] = df["ib"] / df["at"]

# Independent variable (X)
df["rd_intensity"] = df["xrd"].fillna(0) / df["at"]

# Moderator
df["ln_at"] = np.log(df["at"])

# Interaction term (H2)
df["rd_x_size"] = df["rd_intensity"] * df["ln_at"]

# Controls
df["leverage"]      = df["dltt"] / df["at"]
df["capx_intensity"] = df["capx"].fillna(0) / df["at"]
df["cash_ratio"]    = df["che"].fillna(0) / df["at"]

print(f"Variables constructed.")
WINSORIZE_VARS = ["roa", "rd_intensity", "leverage", "capx_intensity", "cash_ratio"]
for var in WINSORIZE_VARS:
    if var in df.columns:
        vals = pd.to_numeric(df[var], errors="coerce").fillna(df[var].median())
        p1, p99 = vals.quantile(0.01), vals.quantile(0.99)
        df[var] = vals.clip(lower=p1, upper=p99)
        print(f"  Winsorized {var}: [{df[var].min():.4f}, {df[var].max():.4f}]")

# ── R&D firms ─────────────────────────────────────────────────────────────────
n_rd = (df["xrd"] > 0).sum()
print(f"\nFirms with R&D > 0: {n_rd:,} ({n_rd/len(df)*100:.1f}%)")

# ── Core variables ────────────────────────────────────────────────────────────
CORE_VARS = ["roa", "rd_intensity", "ln_at", "leverage", "capx_intensity", "cash_ratio"]
VAR_LABELS = {
    "roa":            "RoA (ib/at)",
    "rd_intensity":   "R&D Intensity (xrd/at)",
    "ln_at":          "Firm Size (ln assets)",
    "leverage":       "Leverage (dltt/at)",
    "capx_intensity": "CAPX Intensity (capx/at)",
    "cash_ratio":     "Cash Ratio (che/at)",
}

df_core = df[["gvkey", "fyear"] + CORE_VARS].dropna()
print(f"\nComplete cases for core vars: {len(df_core):,}")

# ── Summary statistics ────────────────────────────────────────────────────────
stats = df_core[CORE_VARS].describe().T
stats["median"] = df_core[CORE_VARS].median()
stats = stats[["count", "mean", "median", "std", "min", "max"]]
stats.index = [VAR_LABELS.get(v, v) for v in stats.index]
stats.to_csv(TABLES / "summary_statistics.csv")
print(f"\nSummary statistics saved.")
print(stats.round(4).to_string())

# ── Correlation matrix ────────────────────────────────────────────────────────
corr = df_core[CORE_VARS].corr()

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
plt.colorbar(im, ax=ax)
labels = [VAR_LABELS.get(v, v) for v in CORE_VARS]
ax.set_xticks(range(len(CORE_VARS)))
ax.set_yticks(range(len(CORE_VARS)))
ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
ax.set_yticklabels(labels, fontsize=8)
for i in range(len(CORE_VARS)):
    for j in range(len(CORE_VARS)):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
ax.set_title("Correlation Matrix — Core Variables")
plt.tight_layout()
plt.savefig(FIGURES / "correlation_matrix.png", dpi=150)
plt.close()
print("Saved: correlation_matrix.png")

# ── DV distribution ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(df_core["roa"], bins=60, edgecolor="white", color="steelblue")
axes[0].set_title("RoA Distribution")
axes[0].set_xlabel("RoA (ib/at)")
axes[0].set_ylabel("Count")

axes[1].hist(df_core["ln_at"], bins=60, edgecolor="white", color="steelblue")
axes[1].set_title("Firm Size Distribution (ln assets)")
axes[1].set_xlabel("ln(Total Assets)")
axes[1].set_ylabel("Count")

plt.tight_layout()
plt.savefig(FIGURES / "dv_distribution.png", dpi=150)
plt.close()
print("Saved: dv_distribution.png")

# ── Main relationship plot ────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: R&D intensity vs RoA scatter
sample = df_core.sample(min(3000, len(df_core)), random_state=42)
axes[0].scatter(sample["rd_intensity"], sample["roa"],
                alpha=0.3, s=10, color="steelblue")
m, b = np.polyfit(sample["rd_intensity"], sample["roa"], 1)
x_line = np.linspace(sample["rd_intensity"].min(), sample["rd_intensity"].max(), 100)
axes[0].plot(x_line, m * x_line + b, color="red", linewidth=2)
axes[0].set_xlabel("R&D Intensity (xrd/at)")
axes[0].set_ylabel("RoA (ib/at)")
axes[0].set_title("R&D Intensity vs RoA")

# Right: RoA by R&D group
has_rd  = df_core[df_core["rd_intensity"] > 0]["roa"]
no_rd   = df_core[df_core["rd_intensity"] == 0]["roa"]
axes[1].boxplot([no_rd, has_rd], labels=["No R&D", "Has R&D"],
                patch_artist=True,
                boxprops=dict(facecolor="lightblue"),
                medianprops=dict(color="red", linewidth=2))
axes[1].set_ylabel("RoA (ib/at)")
axes[1].set_title("RoA by R&D Status")

plt.tight_layout()
plt.savefig(FIGURES / "main_relationship.png", dpi=150)
plt.close()
print("Saved: main_relationship.png")

# ── Save panel with vars ──────────────────────────────────────────────────────
out = PROC / "panel_with_vars.parquet"
df.to_parquet(out, index=False)
print(f"\nSaved panel_with_vars.parquet: {df.shape[0]:,} rows x {df.shape[1]} columns")
print("Done.")