# KI-Einsatz im internationalen Vertrieb: Leadgenerierung & Kundenkommunikation
### ExInt II | WU Vienna | SS 2026 | Alexander Kusmaul

## Research Question
Wie verändert der Einsatz von Künstlicher Intelligenz die Leadgenerierung und
Kundenkommunikation im internationalen Vertrieb exportorientierter KMU?

## Hypotheses
- **H1:** KMU mit KI-gestützter Leadgenerierung erzielen eine höhere
  internationale Vertriebsperformance als KMU ohne KI-Einsatz.

## Theoretical Foundation

**Technology Acceptance Model (Davis, 1989):**
Perceived usefulness und ease of use bestimmen, ob KMU KI-Tools im Vertrieb
adoptieren. Je höher der wahrgenommene Nutzen, desto stärker der Einsatz im
internationalen Kundenkontakt.

**Resource-Based View (Barney, 1991):**
KI-Kompetenz im Vertrieb stellt eine schwer imitierbare Ressource dar, die
nachhaltigen Wettbewerbsvorteil im internationalen Markt erzeugt.

## Operationalization
| Construct | Variable | Source |
|---|---|---|
| KI-Adoption | Dummy: KI-Tool im Vertrieb (ja/nein) | Primärerhebung / Survey |
| Vertriebsperformance | Umsatzwachstum international (%) | Survey / Compustat |
| Unternehmensgröße | log(Mitarbeiteranzahl) | Survey |
| Exportintensität | Exportumsatz / Gesamtumsatz | Survey |
| Branche | NACE-Code | Survey |

## Data
- Methode: Quantitative Querschnittserhebung (OLS Regression)
- Sample: Exportorientierte österreichische KMU (≤250 MA, Exportanteil ≥20%)
- Erhebung: Online-Survey (geplant Sep–Nov 2026)

## How to Reproduce
git clone https://github.com/AlexanderKusmaul/ki-vertrieb-kmu
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
task all
