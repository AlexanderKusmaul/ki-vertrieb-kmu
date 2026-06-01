# Innovationsintensität und finanzielle Performance von KMUs
### ExInt II | WU Vienna | SS 2026 | Alexander Kusmaul

## Research Question
Wie beeinflusst die Innovationsintensität (F&E-Ausgaben) die finanzielle
Performance von KMUs im internationalen Vergleich?

## Hypotheses
- **H1:** KMUs mit höherer F&E-Intensität weisen eine höhere Eigenkapitalrendite (RoA) auf.

## Variables

### Dependent variable (Y)
| Construct | Data Item(s) | Formula |
|-----------|-------------|---------|
| RoA | nicon, at | nicon / at |

### Independent variable (X)
| Construct | Data Item(s) | Formula |
|-----------|-------------|---------|
| R&D Intensity | xrd, at | xrd / at |

### Controls
| Construct | Data Item(s) | Formula |
|-----------|-------------|---------|
| Firm size | at | log(at) |
| Leverage | dltt, dlc, seq | (dltt + dlc) / seq |
| Firm age | fyear, inco | fyear - inco |
| Sales growth | sale | (sale_t - sale_t-1) / sale_t-1 |

## SME Definition
Firms with emp < 0.25 (fewer than 250 employees, EU SME definition).

## Data
| Item | Detail |
|------|--------|
| Source | WRDS / Compustat Global |
| Table | comp_global_daily.g_funda |
| Downloaded | 2026-05-28 |
| License | WRDS subscriber agreement |
| Fiscal years | 2015–2024 |
| Raw rows | 338,462 |
| Clean rows | 338,462 |

## How to Reproduce
git clone https://github.com/AlexanderKusmaul/ki-vertrieb-kmu
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python code/01_pull_data.py
python code/02_clean.py
