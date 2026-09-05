# UDE Points

A custom UFC fighter-ranking system that scores every fight on **how a fighter won, how dominant they were, and who they beat** — not just win/loss. Built to answer two questions: *who is the greatest UFC fighter ever*, and *who is the best right now*.

## What this is

Most win/loss records treat every win identically. UDE Points doesn't: a dominant finish over a top contender earns meaningfully more than a split decision over an unranked opponent, and every component of that judgment is calibrated from the fight data itself rather than hand-picked.

Design principles the scoring engine holds to:
- **Evaluative, not predictive.** This measures accumulated career merit, not a forecasting model or a zero-sum Elo ladder.
- **Non-redundant signals.** Distinct components (finishing rate, power, durability, dominance) are kept separate rather than blended into one number that hides what's actually driving it.
- **No future leakage.** Every calibration is strictly temporal — nothing is fit using data from on or after its own cutoff.
- **Small samples are shrunk, not trusted at face value.** A 3-fight career doesn't get to post a top-10 rate purely because the denominator is tiny.

## How it works

```
raw scraped data  →  dataset_processing_pipeline.py   (merge into one row per fight)
                  →  ude_points_feature_engineering_pipeline.py  (PDI / dominance features)
                  →  ude_points_algorithm.py           (UDE point scoring)
                  →  ude_points_utils.py               (rankings & career analysis)
```

Per fight, in order: a base win/loss value → title and multi-division bonuses → a continuous **dominance adjustment** driven by PDI (Performance Dominance Index — striking, control time, takedowns, submission attempts, and knockdowns blended into one margin) → a method-of-victory residual → streak and title-defense bonuses → an opponent-quality/rating-gap bonus → age adjustments → a rematch adjustment. The scoring engine (v2.6) is locked; everything downstream of it is built without touching how a fight itself gets scored.

## Rankings & analysis available

- **Career GOAT ranking** — `rank_fighters_by_shrunk_ude_rate`: per-fight value generated, Bayesian-shrunk toward the population mean so a short, hot streak doesn't outrank a long, dominant career.
- **Division and gender-scoped rankings** — `rank_fighters_by_shrunk_ude_rate_by_weight_class` / `_by_gender`: the same ranking, scoped to one division or to men's/women's divisions, with the shrinkage prior recomputed for that population.
- **Finishing rate, power, and durability-adjusted power** — three deliberately separate metrics: who finishes fights, who hits hardest, and who hits hardest *against durable opposition specifically*.
- **Rematch history** — every fighter-pair rematch, whether it was immediate, and the outcome.
- **Opponent-similarity matching** — for a fighter's upcoming opponent, finds their most physically- or stylistically-similar past opponents.
- **Fighter status** — active/inactive as of any given date.

Full mechanics, current numbers, and known limitations are in [`canonical_project_state.md`](canonical_project_state.md).

## Repo structure

| File | Purpose |
|---|---|
| `dataset_processing_pipeline.py` | ETL — merges raw scraped fight/event/fighter-bio data into one row per fight, with validation |
| `ude_points_feature_engineering_pipeline.py` | Computes PDI and chronological fighter-state features |
| `ude_points_algorithm.py` | The scoring engine itself (locked at v2.6) |
| `ude_points_utils.py` | Rankings, career trajectories, and all analysis functions above |
| `fighter_scrape_new.py` | Incremental UFCStats.com fighter-bio scraper |
| `ude_scrape_new.py` | Incremental UFCStats.com fight-detail scraper |
| `run_refresh.py` | Single entry point chaining ETL → features → scoring → validation |
| `current_df.csv` | **The current production dataset** — one row per fight, fully scored |
| `fighters_df.csv` | Fighter bio data (height/weight/reach/stance/DOB) keyed by UFCStats URL |
| `requirements.txt` | Python dependencies |
| `.github/workflows/refresh_dataset.yml` | The automated weekly refresh (below) |
| `canonical_project_state.md` | Current architecture, scoring mechanics, and design decisions — start here |
| `data_integrity_and_invariants.md` | Data-integrity corrections and load-bearing invariants not obvious from the code |

## Automated weekly refresh

`current_df.csv` and `fighters_df.csv` are kept current by a scheduled GitHub Actions workflow, not by hand:

1. Fetches the latest source data from [greco1899's UFCStats scraper](https://github.com/Greco1899/scrape_ufc_stats).
2. Incrementally scrapes new fight results and fighter bios (only what's actually changed — new fighters, fighters who just competed, or profiles still missing a field).
3. Runs the full ETL → feature engineering → scoring pipeline and validates the result against the prior production file (fails closed on any lost fight or unexpected column change).
4. Opens a pull request with a validation summary. **Merging the PR is the only manual step** — it updates `current_df.csv` and `fighters_df.csv` in place for the next cycle.

Runs every Monday, or on demand via the Actions tab's "Run workflow" button.

## Getting started

```bash
pip install -r requirements.txt
```

```python
import pandas as pd
from ude_points_utils import rank_fighters_by_shrunk_ude_rate, rank_fighters_by_shrunk_ude_rate_by_weight_class

df = pd.read_csv('current_df.csv', low_memory=False)

# Overall GOAT ranking
goats = rank_fighters_by_shrunk_ude_rate(df, prior_strength=10.0, min_fights=10)

# Scoped to one division
lw_goats = rank_fighters_by_shrunk_ude_rate_by_weight_class(df, weight_class='LW', min_fights=10)
```

To run a refresh manually instead of waiting for the schedule:

```bash
python ude_scrape_new.py --fight-details-csv <path> --current-dataset-csv current_df.csv --output latest_fights_df.csv
python fighter_scrape_new.py --tott-csv <path> --existing-fighters-csv fighters_df.csv --latest-fights-csv latest_fights_df.csv --output fighters_df.csv
python run_refresh.py --current-dataset current_df.csv --fighters fighters_df.csv --latest-fights latest_fights_df.csv --latest-events <path> --previous-production-file current_df.csv --output current_df.csv
```

## Data source

Fight and fighter data originates from [UFCStats.com](http://ufcstats.com), via [greco1899's scrape_ufc_stats](https://github.com/Greco1899/scrape_ufc_stats) for the base feed and this project's own scrapers for incremental updates and bio enrichment.
