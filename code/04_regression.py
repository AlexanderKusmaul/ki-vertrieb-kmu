"""
04_regression.py
----------------
OLS, Two-Way Fixed Effects, and interaction model for
R&D intensity and firm performance (RoA) among SMEs.

Output
------
output/tables/regression_results.csv

Usage
-----
    python code/04_regression.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

try:
    import statsmodels.formula.api as smf
    from linearmodels.panel import PanelOLS
    HAS_LINEARMODELS = True
except ImportError:
    HAS_LINEARMODELS = False
    print("Warning: linearmodels not installed. FE models will be skipped.")

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT   = Path(__file__).resolve().parents[1]
PROC   = ROOT / "data" / "processed"
TABLES = ROOT / "output" / "tables"
TABLES.mkdir(parents=True, exist_ok=True)

# ── Load panel ────────────────────────────────────────────────────────────────
df = pd.read_parquet(PROC / "panel_with_vars.parquet")
print(f"Loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")

# ── Variables ─────────────────────────────────────────────────────────────────
DV       = "roa"
X_MAIN   = "rd_intensity"
INTERACT = "rd_x_size"
CONTROLS = ["ln_at", "leverage", "capx_intensity", "cash_ratio"]

ALL_VARS = [DV, X_MAIN, INTERACT] + CONTROLS
df_reg = df[["gvkey", "fyear"] + ALL_VARS].dropna()
print(f"Regression sample: {len(df_reg):,} rows, {df_reg['gvkey'].nunique():,} firms")

# ── Helper: extract standard errors ───────────────────────────────────────────
def get_se(result):
    try:
        return result.std_errors
    except AttributeError:
        return result.bse

# ── Model 1: Pooled OLS ───────────────────────────────────────────────────────
formula1 = f"{DV} ~ {X_MAIN} + {' + '.join(CONTROLS)}"
m1 = smf.ols(formula1, data=df_reg).fit(cov_type="HC3")
print(f"\nModel 1 (OLS): N={int(m1.nobs):,}, R2={m1.rsquared:.4f}")

# ── Model 2: Two-Way Fixed Effects ────────────────────────────────────────────
results = {}
if HAS_LINEARMODELS:
    df_fe = df_reg.copy()
    df_fe = df_fe.set_index(["gvkey", "fyear"])

    formula2 = f"{DV} ~ {X_MAIN} + {' + '.join(CONTROLS)} + EntityEffects + TimeEffects"
    try:
        m2 = PanelOLS.from_formula(formula2, data=df_fe).fit(
            cov_type="clustered", cluster_entity=True
        )
        print(f"Model 2 (TWFE): N={int(m2.nobs):,}, R2={m2.rsquared:.4f}")
        fe_ok = True
    except Exception as e:
        print(f"Model 2 failed: {e}")
        fe_ok = False

    # ── Model 3: TWFE + Interaction ───────────────────────────────────────────
    formula3 = f"{DV} ~ {X_MAIN} + {INTERACT} + {' + '.join(CONTROLS)} + EntityEffects + TimeEffects"
    try:
        m3 = PanelOLS.from_formula(formula3, data=df_fe).fit(
            cov_type="clustered", cluster_entity=True
        )
        print(f"Model 3 (TWFE+Int): N={int(m3.nobs):,}, R2={m3.rsquared:.4f}")
        int_ok = True
    except Exception as e:
        print(f"Model 3 failed: {e}")
        int_ok = False
else:
    fe_ok = False
    int_ok = False

# ── Results table ─────────────────────────────────────────────────────────────
rows = []

def add_rows(label, result, var_list):
    params = result.params
    se     = get_se(result)
    try:
        pvals = result.pvalues
    except AttributeError:
        pvals = result.pvalues
    for v in var_list:
        if v in params.index:
            rows.append({
                "variable": v,
                "model":    label,
                "coef":     round(params[v], 4),
                "se":       round(se[v], 4),
                "pval":     round(pvals[v], 4),
                "sig":      "***" if pvals[v] < 0.01 else "**" if pvals[v] < 0.05 else "*" if pvals[v] < 0.1 else ""
            })

add_rows("(1) OLS",      m1, [X_MAIN] + CONTROLS)
if fe_ok:
    add_rows("(2) TWFE",     m2, [X_MAIN] + CONTROLS)
if int_ok:
    add_rows("(3) TWFE+Int", m3, [X_MAIN, INTERACT] + CONTROLS)

results_df = pd.DataFrame(rows)
out = TABLES / "regression_results.csv"
results_df.to_csv(out, index=False)
print(f"\nSaved: regression_results.csv")
print(results_df.to_string(index=False))

# ── H1 / H2 diagnostics ───────────────────────────────────────────────────────
print("\n" + "="*50)
print("H1/H2 Diagnostics")
print("="*50)

if fe_ok:
    b_rd   = m2.params[X_MAIN]
    p_rd   = m2.pvalues[X_MAIN]
    sig_rd = p_rd < 0.05
    print(f"\nH1: R&D intensity → RoA (TWFE)")
    print(f"  beta = {b_rd:.4f}, p = {p_rd:.4f}")
    if sig_rd and b_rd < 0:
        print("  → H1 SUPPORTED: negative and significant (R&D expensing effect)")
    elif sig_rd and b_rd > 0:
        print("  → H1 SUPPORTED: positive and significant")
    else:
        print("  → H1 NOT SUPPORTED: not significant")

if int_ok:
    b_int = m3.params[INTERACT]
    p_int = m3.pvalues[INTERACT]
    print(f"\nH2: Firm size moderates R&D-RoA (interaction)")
    print(f"  beta = {b_int:.4f}, p = {p_int:.4f}")
    if p_int < 0.05:
        print("  → H2 SUPPORTED: interaction significant")
    else:
        print("  → H2 NOT SUPPORTED: interaction not significant")

print(f"\nOLS vs FE comparison (R&D intensity coefficient):")
b_ols = m1.params[X_MAIN]
print(f"  OLS beta  = {b_ols:.4f}")
if fe_ok:
    print(f"  TWFE beta = {b_rd:.4f}")
    diff = abs(b_ols - b_rd)
    print(f"  Difference = {diff:.4f} {'→ possible omitted variable bias in OLS' if diff > 0.05 else '→ estimates stable'}")

print("\nDone.")