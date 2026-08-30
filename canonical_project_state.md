# UDE Points — Canonical Project State

**Baseline date: 21 August 2026 — v2.6 locked as production.** Utils/pipeline layer extended 26 August 2026 (§4, §7) — no scoring changes, v2.6's lock still holds.

This document is intended to be the **authoritative compact context** for continuing the UDE Points project. It reflects the current state — architecture, decisions in force, and known tradeoffs — not a history of how it was reached. For the "how we got here," see `project_history.md`.

Be certain to keep your responses succinct and efficient.

---

## 1. Project Objective & Design Scope

**UDE Points** is a custom UFC fighter-ranking system designed to quantify a fighter's historical career strength and resume quality by evaluating not merely whether they won, but **how they won, how dominant they were (PDI), who they beat, and the circumstances surrounding the fight**.

### Primary Goals
* **Who is the UFC's greatest ever fighter (GOAT status)?**
* **Who is the UFC's current best fighter (P4P resume status)?**

### Design Scope & Non-Goals
* **Evaluative, Not Predictive:** The metric evaluates accumulated fight-by-fight historical merit. It is **not** a predictive fight-forecasting engine, nor a standard zero-sum Elo ladder.
* **Non-Redundant Signals:** Different components capture distinct information rather than repeatedly rewarding the same underlying phenomenon.
* **Volume-Bias & Inflation Control:** Downstream ranking layers (`rank_fighters_by_shrunk_ude_rate`) handle volume normalization and shrinkage, keeping the per-fight scoring loop strictly focused on fight performance.

---

## 2. Scoring Mechanics (current)

Per fighter-side, per fight, in order: `raw_base_points` (±3 W/L) → `championship_bonus` (2.0x title bouts) → `multi_division_championship_bonus` (1.25x new-division title) → `dominance_adjustment` (continuous PDI-margin interpolation, 0.75x–1.30x) → `+= method_pdi_residual_points` (±0.75 cap, UD-referenced) → `title_defense_bonus` + `streak_adjustment` (both computed off `raw_base_points` independently, summed, scaled once by `perf_scale`) → `higher_rated_opponent_bonus` (trailing-3-fight smoothed rating gap, tanh-bounded, `perf_scale` applied internally) → `age_adjustment` / `own_age_adjustment` (invariant) → `rematch_adjustment` (invariant) → `opponent_quality_adjustment` (invariant). Final swing clamped to `±ABSOLUTE_SWING_CAP` (60 pts).

**Loss symmetry:** `perf_scale` extends to losses (0.20–1.0), so a bad stoppage loss is penalized more than a tight decision loss. `opponent_quality_adjustment` is asymmetric on losses: elite opposition gives no cushion (1.0x floor), weak opposition amplifies the penalty (up to 1.5x).

**Upset/rating-gap bonus:** continuous tanh curve (no discrete step tiers), fed by a trailing-3-fight rolling average of both fighters' ratings rather than the instantaneous pre-fight value, to avoid single-fight rating whiplash feeding back into the gap calculation.

**Age adjustment — two different sign conventions by design, not oversight:**
- `opponent` side (`age_adjustment`): signed slope — an opposition-quality discount. A declining opponent is objectively worth less to beat, worse to lose to, in whichever direction the calibration finds.
- `own` side (`own_age_adjustment`): `abs(slope)` — an achievement-under-adversity reward. Every other UDE component (`higher_rated_opponent_bonus`, `opponent_quality_adjustment`) already rewards clearing a harder bar with *more* credit, never less; own-age matches that convention rather than penalizing a lower a-priori win probability.
- The own-age bonus is gated by opponent age via `_own_age_gate_scale`, a sigmoid (not a hard cutoff) centered on `reference_age`, 3-year width: full credit well below reference, ~0 well above.
- Both `_age_multiplier` code paths explicitly guard `pd.isna(reference_age)` (the "neutral calibration" state for years with insufficient prior history) and return a true 1.0x no-op — this guard is load-bearing, not decorative; see `project_history.md` #11 for what happens without it.

**Calibration:** age-decline curve (piecewise-logistic, BIC-selected breakpoint) and method×PDI residual (binomial GLM, forward 5-fight win rate) are each fit per calendar year on a trailing 5-year window (`CALIBRATION_ROLLING_WINDOW_YEARS`), falling back to full expanding history only below `CALIBRATION_MIN_FIGHT_OBSERVATIONS` (1,000) fights — keeps calibration reflective of the contemporary era. Strictly temporal: no fit uses data on/after its own cutoff. Fallback engages 1999–2011 in the current dataset; rolling window active 2012 onward.

**Diagnostics persisted per fight** (`<metric>_fighter_1`/`_fighter_2` columns): `method_pdi_residual`, `performance_scaling_factor`, `higher_rated_opponent_bonus`, `opponent_quality_adjustment`, `title_defense_bonus`, `streak_bonus`, `age_adjustment`, `own_age_adjustment`, `rematch_adjustment`, `absolute_swing_cap_triggered`, `quality_score`, `quality_multiplier`. Every bonus component's marginal contribution is independently auditable from the output CSV without re-running with ablation. `df.attrs` also carries `absolute_swing_cap_bind_count`/`_total_observations`/`_bind_rate` — currently 0/16,904 (0%), the cap is dormant, not load-bearing.

### Accepted tradeoffs (current decisions, not open questions)
- **PDI signal reuse across `dominance_adjustment` and `perf_scale`:** both derive from the identical `pdi_margin / PDI_MARGIN_SCALE`, so they move in the same direction on every fight. Accepted because `perf_scale` is shrink-only (0.20–1.0) and structurally cannot generate points on its own — it dampens other components, it doesn't duplicate `dominance_adjustment`'s output. No unclaimed orthogonal signal exists elsewhere in the architecture for `perf_scale` to use instead without creating a new collision (method → residual, opponent record → OQ adjustment, rating gap → upset bonus are all already owned).
- **Rolling/expanding calibration boundary (2011→2012):** the regime switch could in principle produce a calibration jump unrelated to any real change in the sport. Measured and found statistically indistinguishable from ordinary year-to-year BIC refit noise elsewhere in the series — left unaddressed.
- **Smoothing constants** (`OWN_AGE_GATE_WIDTH_YEARS=3.0`, `HIGHER_RATED_GAP_SCALE=30.0`, `HIGHER_RATED_GAP_FLOOR=15.0`, `TITLE_DEFENSE_CAP`/`DECAY`) are hand-tuned, not empirically fit — consistent across the codebase, not a gap specific to any one component.

---

## 2a. Locked GOAT Ranking

`rank_fighters_by_shrunk_ude_rate(df, prior_strength=10.0, min_fights=10)` on `latest_fights_up_to_islam_garry_with_ude_points_calculated_v2_6.csv`. `min_fights=10` excludes small-sample careers that Bayesian shrinkage alone doesn't stabilize (`population_mean_rate` is still computed over the full unfiltered population, so the floor doesn't bias the shrinkage target for anyone). Stability-checked: 18–19/20 top-20 membership overlap when `prior_strength` is swept 5→20 with the floor applied.

| Rank | Fighter | Record | Fights | Career Gain | Shrunk Rate |
|---|---|---|---|---|---|
| 1 | Georges St-Pierre | 20-2-0 | 22 | 154.8 | 4.428 |
| 2 | Jon Jones | 22-1-0 | 24 | 158.4 | 4.275 |
| 3 | Islam Makhachev | 18-1-0 | 19 | 110.9 | 3.373 |
| 4 | Demetrious Johnson | 15-2-1 | 18 | 89.0 | 2.712 |
| 5 | Amanda Nunes | 16-2-0 | 18 | 82.9 | 2.495 |
| 6 | Valentina Shevchenko | 15-3-1 | 19 | 81.2 | 2.350 |
| 7 | Khabib Nurmagomedov | 13-0-0 | 13 | 67.0 | 2.343 |
| 8 | Alexander Volkanovski | 15-3-0 | 18 | 76.0 | 2.247 |
| 9 | Justin Gaethje | 11-5-0 | 16 | 57.1 | 1.693 |
| 10 | Merab Dvalishvili | 14-3-0 | 17 | 56.4 | 1.605 |
| 11 | Dricus Du Plessis | 10-1-0 | 11 | 44.5 | 1.494 |
| 12 | Daniel Cormier | 11-3-0 | 15 | 50.0 | 1.476 |
| 13 | Ilia Topuria | 9-1-0 | 10 | 42.5 | 1.473 |
| 14 | Francis Ngannou | 12-2-0 | 14 | 46.4 | 1.390 |
| 15 | Alex Pereira | 10-3-0 | 13 | 44.0 | 1.345 |
| 16 | Alexandre Pantoja | 14-4-0 | 18 | 49.5 | 1.300 |
| 17 | Movsar Evloev | 10-0-0 | 10 | 38.9 | 1.290 |
| 18 | Aljamain Sterling | 18-5-0 | 23 | 54.0 | 1.239 |
| 19 | Benson Henderson | 11-3-0 | 14 | 42.7 | 1.232 |
| 20 | Petr Yan | 12-4-0 | 16 | 44.6 | 1.214 |

619 fighters clear the `n_fights >= 10` floor. Topuria, Evloev, Dricus Du Plessis, and Alex Pereira sit closest to it (10–13 fights).

---

## 3. Core Data & Architecture

The principal dataset contains **8,564 UFC fights** through the Islam Makhachev vs. Ian Machado Garry timeframe.

### Fundamental Dataframe Structure
The fight dataframe utilizes a standardized two-sided convention:
```text
fighter_1
fighter_2

<feature>_fighter_1
<feature>_fighter_2
```
This naming convention is strictly enforced across utility transformations. `ude_points_utils.extract_fighter_details_programmatically`/`extract_opponent_details_programmatically` discover columns generically by substring match on `fighter_1`/`fighter_2` — any new `<metric>_fighter_1`/`_fighter_2` column is automatically picked up by `create_fighter_career_dataset` with no utils changes needed.

### `ude_points_feature_engineering_pipeline.py` — PDI computation

`pdi_margin` (the single input driving `dominance_adjustment`, `perf_scale`, and the method×PDI residual, all at once) is built in `calculate_phase_magnitude_and_pdi` from five phase magnitudes: striking, control time, takedowns, submission attempts, knockdowns. Takedown and submission magnitude use `_dominance_magnitude`, a smooth blend (soft-AND of two sigmoids, over landed-count and proportion-of-total-landed) between a "close" curve (capped 0.35) and a "decisive" curve (0.36+) — replacing a hard `count > N and proportion >= P` gate that produced a real value-gap at the boundary. Landed counts are integers, so the count-axis width (`count_width=0.10`) is deliberately tight: low counts (1-3 landed) stay within ~0.004 of their old capped value, preserving the original small-sample protection; the actual smoothing benefit is in the proportion axis (genuinely continuous) and in closing the 0.35→0.36 value-gap itself.

`map_weight_class` classifies by substring containment on the raw weight-class string, checked longest-name-first — required because `"Heavyweight"` is a literal substring of `"Light Heavyweight"`; matching shortest-first (or dict insertion order) would misclassify.

Two computed-but-unconsumed-by-scoring pathways are retained deliberately, not dead code: `who_won_*`/`dominant_fighter`/`phases_won` (categorical phase-win detection, for narrative/visualization use), and `rematch_column`/`is_rematch` (0-indexed meeting count per fighter pair — 0=first meeting, 1=first rematch, 2=second rematch, etc. — for visualization; `ude_points_algorithm.py`'s own `rematch_adjustment` tracks pair history incrementally during scoring instead of reading these columns).

`engineer_all_features` runs `add_standing_sig_strikes_columns` before `add_time_and_per_min_features` (not after) — the latter scans `df.columns` for every `*_landed`/`*_attempted` column present *at that point* to generate a per-minute variant, so `standing_sig_strikes_landed_per_min_*` would silently never be generated if its source column didn't exist yet. The raw landed/attempted stat columns consumed by the cumulative-sum trackers (`update_career_means`, `add_dynamic_strike_accuracy`/`_defence`, `add_dynamic_td_accuracy`/`_defence`) are `fillna(0)`'d once at the top of `engineer_all_features`, before any of those trackers run — each accumulates via unchecked `+=`, so a single NaN would otherwise poison that fighter's running total (and every derived column built from it) for the rest of their career; zero NaN currently present in these columns, so this is a defensive guard, not a live fix.

**Column hygiene:** `v2_6.csv` is rebuilt end-to-end from its 75 genuinely-raw columns on every regeneration — `engineer_all_features` → `calculate_ude_points_with_ablation` → `add_ude_points_difference_columns` — not patched incrementally. The feature-engineering functions add columns via `pd.concat`, which silently creates a duplicate-named column (not an overwrite) if the input already contains that name; re-running any stage against an already-processed file reintroduces exactly this. Full column accounting verified empirically (not by manual enumeration): 75 raw → 151 feature-engineering-derived → 28 UDE-scoring-derived → 2 diff columns = 256 total, zero duplicates at every stage.

---

## 4. Career Trajectories & Utilities
`ude_points_utils.create_fighter_career_dataset` converts the two-sided fight dataframe into fighter-specific career trajectories.
```text
fight-level dataset 
        ↓
fighter-level career trajectory 
        ↓
ranking / historical analysis / visualization
```

### 4a. Rankings
`rank_fighters_by_shrunk_ude_rate` is the locked GOAT ranking (§2a). `rank_fighters_by_shrunk_ude_rate_by_weight_class(df, weight_class=None, start_year=None, end_year=None, prior_strength=10.0, min_fights=None)` is a thin wrapper — `filter_by_weight_class`/`filter_by_year` pre-filter, then the unchanged ranking function runs on the filtered population, so the shrinkage target (`population_mean_rate`) is the *division's own* mean rate, not the promotion-wide one. `filter_by_weight_class` deliberately does not also require both fighters in a fight to individually clear a fight-count floor in that division — an earlier version of this idea did, and silently undercounted a fighter's real fight total by dropping fights against one-off opponents; the floor belongs to the ranking function's own `min_fights`, applied post-scoring on the fighter actually being ranked.

### 4b. Fighter status
`create_fighter_status_dataset(df, as_of=None)` — active/inactive per fighter (fought within 730 days of `as_of`). `as_of` defaults to `datetime.now()`, making the result non-deterministic across runs by design (unlike everything else in this project, "is this fighter still active" is genuinely an as-of-today question) — pass `as_of` explicitly for a reproducible cutoff.

### 4c. Finishing rate, power, and durability-adjusted power — three deliberately separate metrics
None is blended into the others; each answers a different question:
- **`calculate_overall_potency`** ("who finishes fights") — win-conditioned: `finishes / wins` per fighter/weight-class (striking and grappling computed separately, combined by geometric mean), shrunk toward each division's own baseline. Excludes injury-cause stoppages from the finish count (checks `details` for "injury").
- **`calculate_striking_power`** ("who hits hardest") — NOT win-conditioned: `kd / head_strikes_landed` per fighter/weight-class, same shrinkage machinery and injury filter. A knockdown the opponent survives counts exactly as much as one that ends the fight.
- **`calculate_durability_adjusted_power`** — `calculate_striking_power`, reweighted per-fight by `add_opponent_durability_multiplier`: a bounded `[0.5, 2.0]` multiplier from the OPPONENT's own pre-fight standing-KO/TKO-loss rate, shrunk toward *that fight's own weight class* baseline (not one dataset-wide number — this rate spans 8.4x across divisions, HW to WSW). Neither `calculate_striking_power` nor `calculate_overall_potency` incorporates this adjustment; only `calculate_durability_adjusted_power` does.
- Shared helper: `_shrink_rate(count, total, prior_strength, prior_rate)` — same shape as `ude_points_algorithm.shrunk_win_rate`, generalized to an arbitrary count/total pair. `RATE_SHRINKAGE_PRIOR_STRENGTH=5.0` (potency & power) and `DURABILITY_SHRINKAGE_PRIOR_STRENGTH=15.0` are both sensitivity-checked: potency's top-10 rankings hold 7–10/10 stable across a 1→30 sweep, power's hold 9–10/10 (sturdier because its floor, `min_head_strikes_landed`, gates directly on the ratio's own denominator rather than a looser fight-count proxy).
- **Known limitation, not a bug:** `add_opponent_durability_multiplier`'s bound binds for ~63% of fighter-fight observations even with correct division-scoping — a structural consequence of shrinking a right-skewed, rare-event rate toward its mean (a majority of any population legitimately sits below the mean for a rare event), not a miscalibration. Left as-is; revisit only if a specific downstream use needs finer discrimination among highly-durable opponents specifically (see `project_history.md` for the fuller mean-vs-median tradeoff discussion).

### 4d. Rematch history
`process_rematch_data(df, exclude_no_contests=False)` (+ `find_same_winner_rematches`) — every fighter-pair rematch, whether each meeting was immediate (no intervening fight for either side since their last meeting), and the winner. `assign_winner` returns distinct `'draw'`/`'no_contest'` sentinels rather than a collapsed `None`, so two draws between the same pair can't spuriously register as a repeat win. `filter_invalid_rematches` reuses `ude_points_algorithm.is_no_score_fight` (checks both `fight_result == 'NC'` and `method in {'DQ', 'Overturned'}`) instead of a narrower method-string-only check.

### 4e. Opponent-similarity matching
`find_most_similar_past_opponents(df, fighter_name, future_opponent_name, exclude_future_opponent=True)` — for a fighter's upcoming opponent, ranks their own past opponents by similarity to that opponent. **Physical** (age, height, reach) and **style** (`dynamic_*` striking/TD accuracy & defense) similarity are reported separately, not blended into one score — a fighter can be a close physical match while fighting a completely different style. Every compared column is min-max scaled before differencing (unscaled differences let whichever column has the largest raw numeric range dominate the ranking). `total_difference` is the mean of the *available* `|diff|` values per row, not a sum with missing columns filled to 0 — filling-to-0 would let a past opponent with zero real comparison data score a false "perfect match." `stance_match` (`'same'`/`'different'`/`'unknown'`) is attached as a flag alongside the score, not folded into the numeric distance, since stance is categorical, not subtractable. `exclude_future_opponent=True` (default) drops the future opponent's own past meeting(s) with the fighter from their own comparison set — without it, a fighter who's already fought their upcoming opponent trivially ranks that real fight as the "most similar" match to itself.

---

## 5. Planned Next Steps (v3 candidates)

**v2.6 is locked as production** (System Integrity 9.5/10, Theoretical Alignment 9/10 — see audit history). These are candidate directions for a future v3, not scheduled work:

1. **Era/division-strength normalization.** Nothing in the current architecture accounts for how strong the competition was in a given era or weight class — a title run in a shallow field scores identically to one in a stacked one. This is the largest gap against the project's own primary goal (GOAT status) and the most likely point of public/critical pushback. **Feasibility already assessed — see §6, don't redo from scratch.** Re-derive against the then-current dataset before acting on it, though: §6's numbers (874/2,544 bridge fighters, per-division roster sizes, 0% stat-coverage gaps) will have shifted by the time this is picked up, and could change the feasibility verdict.
2. **Extend empirical calibration to the remaining hand-tuned constants.** Age and method×PDI are data-fit; `dominance_adjustment`'s anchors, `perf_scale`'s pivot, `title_defense_bonus`'s saturation curve, and the upset-bonus gap/scale are still hand-picked — as are `_dominance_magnitude`'s own thresholds/widths (§3), even after its hard-cliff fix. Fitting those the same way would close the gap and would likely surface any remaining instances of this project's one confirmed bug class (a neutral/fallback state silently resolving to a clip boundary instead of a true no-op) before they ship. Note: the *cliff* in PDI's takedown/submission magnitude specifically is already fixed (§3) — this item is about the still-hand-picked constants generally, not a repeat of that.
3. **Decouple `perf_scale` from `dominance_adjustment`** (lower priority) — once a genuinely non-PDI-derived signal exists to drive `perf_scale`. Not worth forcing before then; see §2's "Accepted tradeoffs."

---

## 6. Era/Division-Strength Normalization — Feasibility

**Partially possible now, but it's a new subsystem, not a quick fix — and part of the ceiling is unfixable.**

What's available: fight stats (PDI, significant strikes, etc.) have 0% missing data back to 1999, so there's no raw-coverage blocker. 34.4% of all fighters (874 of 2,544) have fought in 2+ weight classes — enough cross-division bridges to plausibly identify relative division/era strength via a paired-comparison model (Bradley-Terry-style), the same way this project already fits age and method effects, rather than a circular "average the division's own UDE ratings" shortcut (which would just feed the ranking back into itself).

What's genuinely hard:
- **Identification is real modeling work.** A new calibration function is needed — analogous to `calibrate_age_effects` — that estimates weight-class × era strength offsets from the bridge-fighter network, with the same strict-temporal, no-future-leakage discipline already enforced elsewhere, and an explicit NaN-safe fallback for under-connected cells (the exact class of bug found and fixed in `_age_multiplier` this session must be designed against from the start here).
- **Some cells have no data by construction, not by gap.** Featherweight and bantamweight didn't exist in the UFC before ~2011 (0 unique fighters in 2001–2003 in this dataset, 129–140 by 2022–2024) — there's no "true" era-strength value to estimate for a division that didn't exist yet, only a defined "not applicable" state.
- **Measurement validity, not just volume.** Rules, judging criteria, and round/format standards changed materially over 1999–2026; a "dominant" PDI performance under one era's judging isn't necessarily comparable to another's, and no amount of additional fight-count data fixes that — it's a property of the sport's history, not an engineering gap.
- **No ground truth.** There's no objective answer to "was 2003 heavyweight weaker than 2024 lightweight," so any model needs face-validity checks against combat-sports historical consensus before being trusted, not just a clean fit statistic.

Net: buildable, and the data supports a first version — but it's a v3-scale project (new calibration subsystem + validation framework), not something to fit into v2.6's maintenance.

---

## 7. Current Source Files

**Folder layout note (26 Aug 2026, outside pipeline tracking):** notebooks and several older dataset snapshots (`v2_6.csv`, `v2_6_with_phase_profiles.csv`, plus the raw ETL inputs used for the stance regeneration below) were moved into an `ipynb and old datasets/` subfolder. `fights_up_to_islam_garry_ready_for_features.csv` and the current production file stayed in the main folder.

**Pipeline order:** `dataset_processing_pipeline.py` (raw scrape → 1-row-per-fight) → `ude_points_feature_engineering_pipeline.py` (→ PDI/chronological features) → `ude_points_algorithm.py` (→ UDE points) → `ude_points_utils.py` (→ rankings/career views).

* ```text dataset_processing_pipeline.py ``` — ETL: merges raw scraped fight/event/fighter-bio data into one row per fight (`run_etl_pipeline`). Drops and reports (does not silently discard) any row where `event_date` is null — `drop_rows_with_null_event_date`, called twice inside `run_etl_pipeline`: right after column standardization (catches a failed event-date join on the freshly-scraped data) and again right after the optional `current_dataset` merge (catches null/unparseable dates already sitting in the historical data being appended to — a fresh scrape's own check can't see those, since they enter the pipeline later). `ude_points_feature_engineering_pipeline.engineer_all_features` calls the same function again defensively at its own entry point, in case its input didn't come through this ETL step at all. `bio_cols` (step 6) now also carries `STANCE` through from the raw fighter-bio scrape into the merged fight dataset. The final column-ordering step (step 12) *appends* any column not named in its `ordered_columns` list instead of dropping it — a strict whitelist there previously discarded `STANCE` silently even after the bio merge succeeded, and (once switched to append) immediately surfaced a second, unrelated pre-existing issue: `convert_to_one_fight_one_row`'s `.nth(0)`/`.nth(1)` calls didn't drop their pre-groupby row index, so a bare `.reset_index()` was creating a meaningless `index_fighter_1`/`_fighter_2` column that the old whitelist had been silently swallowing too. Both are fixed (`ordered_columns` appends now; `.reset_index(drop=True)` on both `.nth()` calls).

* ```text fights_up_to_islam_garry_ready_for_features.csv ``` — Output of `run_etl_pipeline`; input to `engineer_all_features`. 75 raw columns. Verified: running the full pipeline (`engineer_all_features` → `calculate_ude_points_with_ablation` → `add_ude_points_difference_columns`) on this file reproduces `v2_6.csv` bit-for-bit (same 8,564 fight URLs, zero difference in any UDE point) — confirms the pipeline is fully reproducible from genuinely raw data, not just self-consistent under incremental patching.

* ```text ude_points_algorithm.py ``` — Authoritative UDE scoring implementation. Unchanged since v2.6's lock.

* ```text ude_points_feature_engineering_pipeline.py ``` — Generates chronological state and PDI fight-performance features. Unchanged since v2.6's lock.

* ```text ude_points_utils.py ``` — Peak/career/shrunk-rate rankings and career dataset conversions, plus (added 26 Aug 2026, see §4a–4e) division/era-scoped ranking, fighter active/inactive status, finishing-rate and power metrics (with a durability-adjusted variant), rematch history, and opponent-similarity matching. Now imports `is_no_score_fight` from `ude_points_algorithm.py`.

* ```text latest_fights_up_to_islam_garry_with_ude_points_calculated_v2_6_with_stance.csv ``` — **Current production file.** 8,564 fights × 258 columns. Verified bit-identical to `v2_6.csv` (below) on all 255 shared, non-`STANCE` columns — this is an additive enrichment (2 new `STANCE_fighter_1`/`_fighter_2` columns, ~98% coverage after backfilling from the fighter-bio source by `fighter_url`), not a scoring change; v2.6's lock still applies. Same naming pattern as `_with_phase_profiles` below — an additive variant, not a version bump.

* ```text latest_fights_up_to_islam_garry_with_ude_points_calculated_v2_6 ``` — 8,564 fights × 256 columns, no `STANCE`. Reflects the full scoring mechanics in §2, rebuilt end-to-end from raw columns per §3's "Column hygiene" note. Superseded by `_with_stance` above for anything needing physical/stance data; otherwise equivalent. Moved to `ipynb and old datasets/` in the 26 Aug reorg.

* ```text latest_fights_up_to_islam_garry_with_ude_points_calculated_v2_6_with_phase_profiles ``` — 8,564 fights × ~400 columns, adds 12-phase skill percentiles (full-history + 3-year era-windowed). Does not have `STANCE`; cross-reference by `fight_url` against `_with_stance` if a use needs both. Moved to `ipynb and old datasets/`.

* ```text latest_fights_up_to_islam_garry_with_ude_points_calculated_v2_5 ``` — Superseded by v2_6. Retained, not deleted, as a pre-fix historical snapshot. Still in the main folder (not part of the 26 Aug reorg).

* ```text project_history.md ``` — Chronological record of how the current state was reached; not required reading to continue the project.
