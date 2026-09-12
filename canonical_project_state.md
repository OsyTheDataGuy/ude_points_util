# UDE Points — Canonical Project State

**Baseline date: 21 August 2026 — v2.6 locked as production.** The scoring engine (`ude_points_algorithm.py`, `ude_points_feature_engineering_pipeline.py`) is unchanged since this lock; `ude_points_utils.py` and `dataset_processing_pipeline.py` have grown since (§4, §7) without touching scoring.

This document is the **authoritative compact context** for continuing the UDE Points project — architecture, decisions in force, known tradeoffs. It holds current state only, not a history of how it was reached. For data-integrity corrections and load-bearing invariants that aren't obvious from the code, see `data_integrity_and_invariants.md`.

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
- The own-age bonus is gated by opponent age via `_own_age_gate_scale`, a sigmoid (not a hard cutoff) centered on `anchor_age` (fixed at `AGE_ANCHOR_YEARS` = 32.0 — the data identifies no threshold; 32 sits at the top of the defensible range so the adjustment engages only past prime, not at the ~30.3 median fighter age; see `age_calibration_validation.md`), 3-year width: full credit well below the anchor, ~0 well above.
- `_age_logit_offset` explicitly guards `pd.isna(anchor_age)` (the "neutral calibration" state for years with insufficient prior history) before any arithmetic and returns 0.0, so `_age_multiplier` is a true 1.0x no-op — this guard is load-bearing, not decorative; see `data_integrity_and_invariants.md` for what happens without it.

**Calibration:** the age-decline curve is a fixed-anchor smooth logistic model (age effect measured as deviation from `AGE_ANCHOR_YEARS`, biting only above it), model order — `flat` / `linear` / `quadratic` — chosen per window by BIC on the fight count. There is no searched breakpoint: the fight data does not identify one (see `age_calibration_validation.md`). Every rolling window from 2008 selects `linear`. The method×PDI residual (binomial GLM, forward 5-fight win rate) is fit the same way. Both are fit per calendar year on a trailing 5-year window (`CALIBRATION_ROLLING_WINDOW_YEARS`), falling back to full expanding history only below `CALIBRATION_MIN_FIGHT_OBSERVATIONS` (1,000) fights — keeps calibration reflective of the contemporary era. Strictly temporal: no fit uses data on/after its own cutoff. Fallback engages 1999–2011 in the current dataset; rolling window active 2012 onward.

**Diagnostics persisted per fight** (`<metric>_fighter_1`/`_fighter_2` columns): `method_pdi_residual`, `performance_scaling_factor`, `higher_rated_opponent_bonus`, `opponent_quality_adjustment`, `title_defense_bonus`, `streak_bonus`, `age_adjustment`, `own_age_adjustment`, `rematch_adjustment`, `absolute_swing_cap_triggered`, `quality_score`, `quality_multiplier`. Every bonus component's marginal contribution is independently auditable from the output CSV without re-running with ablation. `df.attrs` also carries `absolute_swing_cap_bind_count`/`_total_observations`/`_bind_rate` — currently 0/16,904 (0%), the cap is dormant, not load-bearing.

### Accepted tradeoffs (current decisions, not open questions)
- **PDI signal reuse across `dominance_adjustment` and `perf_scale`:** both derive from the identical `pdi_margin / PDI_MARGIN_SCALE`, so they move in the same direction on every fight. Accepted because `perf_scale` is shrink-only (0.20–1.0) and structurally cannot generate points on its own — it dampens other components, it doesn't duplicate `dominance_adjustment`'s output. No unclaimed orthogonal signal exists elsewhere in the architecture for `perf_scale` to use instead without creating a new collision (method → residual, opponent record → OQ adjustment, rating gap → upset bonus are all already owned).
- **Rolling/expanding calibration boundary (2011→2012):** the regime switch could in principle produce a calibration jump unrelated to any real change in the sport. On the method×PDI side it was measured and found statistically indistinguishable from ordinary year-to-year refit noise. On the new `age_gap_linear` series the 2011→2012 step (−0.046 → −0.060) is the largest single adjacent move, ~2× the typical year-to-year delta — modest in absolute terms and within the era-trend direction, left unaddressed but worth knowing.
- **Smoothing constants** (`OWN_AGE_GATE_WIDTH_YEARS=3.0`, `HIGHER_RATED_GAP_SCALE=30.0`, `HIGHER_RATED_GAP_FLOOR=15.0`, `TITLE_DEFENSE_CAP`/`DECAY`) are hand-tuned, not empirically fit — consistent across the codebase, not a gap specific to any one component.

---

## 2a. Locked GOAT Ranking

`rank_fighters_by_shrunk_ude_rate(df, prior_strength=10.0, min_fights=10)` on `current_df.csv` (§7) — reproduced directly from that live file, not a fixed historical snapshot; re-run this against `current_df.csv` after any refresh to regenerate the table below, rather than trusting it to stay current on its own. `min_fights=10` excludes small-sample careers that Bayesian shrinkage alone doesn't stabilize (`population_mean_rate` is still computed over the full unfiltered population, so the floor doesn't bias the shrinkage target for anyone). Stability-checked: 18–19/20 top-20 membership overlap when `prior_strength` is swept 5→20 with the floor applied. See §4a's note below the table for why this rate/shrinkage approach is used at all, over ranking on raw cumulative career points.

| Rank | Fighter | Record | Fights | Career Gain | Shrunk Rate |
|---|---|---|---|---|---|
| 1 | Jon Jones | 22-1-0 | 24 | 160.4 | 4.377 |
| 2 | Georges St-Pierre | 20-2-0 | 22 | 151.2 | 4.363 |
| 3 | Islam Makhachev | 18-1-0 | 19 | 118.0 | 3.670 |
| 4 | Demetrious Johnson | 15-2-1 | 18 | 94.7 | 2.969 |
| 5 | Amanda Nunes | 16-2-0 | 18 | 91.4 | 2.853 |
| 6 | Valentina Shevchenko | 15-3-1 | 19 | 87.8 | 2.628 |
| 7 | Khabib Nurmagomedov | 13-0-0 | 13 | 69.5 | 2.519 |
| 8 | Alexander Volkanovski | 15-3-0 | 18 | 80.2 | 2.451 |
| 9 | Merab Dvalishvili | 14-3-0 | 17 | 63.8 | 1.937 |
| 10 | Alex Pereira | 10-3-0 | 13 | 55.2 | 1.899 |
| 11 | Ilia Topuria | 9-1-0 | 10 | 48.5 | 1.848 |
| 12 | Dricus Du Plessis | 10-1-0 | 11 | 49.5 | 1.809 |
| 13 | Daniel Cormier | 11-3-0 | 15 | 55.9 | 1.774 |
| 14 | Francis Ngannou | 12-2-0 | 14 | 51.4 | 1.659 |
| 15 | Justin Gaethje | 11-5-0 | 16 | 53.3 | 1.607 |
| 16 | Kamaru Usman | 16-4-0 | 20 | 56.7 | 1.505 |
| 17 | Khamzat Chimaev | 9-1-0 | 10 | 40.8 | 1.465 |
| 18 | Movsar Evloev | 10-0-0 | 10 | 40.8 | 1.462 |
| 19 | Benson Henderson | 11-3-0 | 14 | 45.8 | 1.428 |
| 20 | Alexandre Pantoja | 14-4-0 | 18 | 50.8 | 1.401 |

623 fighters clear the `n_fights >= 10` floor. Topuria, Chimaev, and Evloev sit right at it (10 fights); Du Plessis (11) and Pereira (13) aren't far behind. Jones's #1 spot, and Gaethje's fall out of the top 10, both trace to `is_champion_fighter_1`/`_2` and `title_defenses_fighter_1`/`_2` correctly distinguishing an interim title reign from an undisputed one (`create_is_title_bout_column` previously coded every interim title bout identically to an undisputed one, so `update_title_defenses` and every scoring component keyed on champion/defense status treated the two as equivalent). Kamaru Usman and Khamzat Chimaev are new entrants to the top 20; Aljamain Sterling and Petr Yan are the two fighters this pushed out.

---

## 3. Core Data & Architecture

The principal dataset (`current_df.csv`) contains **8,590 UFC fights** as of the last automated refresh. It is kept current on a weekly cycle by the automated pipeline (§8), not a static snapshot — the fight count grows each time a refresh PR is merged.

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
`rank_fighters_by_shrunk_ude_rate` is the locked GOAT ranking (§2a). `rank_fighters_by_shrunk_ude_rate_by_weight_class(df, weight_class=None, start_year=None, end_year=None, prior_strength=10.0, min_fights=None)` is a thin wrapper — `filter_by_weight_class`/`filter_by_year` pre-filter, then the unchanged ranking function runs on the filtered population, so the shrinkage target (`population_mean_rate`) is the *division's own* mean rate, not the promotion-wide one. `filter_by_weight_class` deliberately does not also require both fighters in a fight to individually clear a fight-count floor in that division: doing so would drop fights against one-off opponents entirely, undercounting a fighter's real fight total in that division. The fight-count floor lives solely in the ranking function's own `min_fights`, applied post-scoring on the fighter actually being ranked.

**Why shrunk per-fight rate, not raw cumulative career points:** `career_point_gain` (cumulative UDE points earned) rewards volume as much as quality — on `current_df.csv`, Dustin Poirier (32 fights) ranks #31 by raw cumulative total but only #56 by shrunk rate, because a below-average per-fight rate (1.06, vs. GSP's 7.04) compounds into a large total purely by fighting more times. Dividing by `n_fights` (`raw_rate`) fixes that but creates the opposite problem at the other end: with n=1, Frank Shamrock and Bas Rutten post `raw_rate` ≈ 5.0–5.1 (higher than GSP's career rate) off a single fight each — a sample far too small to trust as a "true" rate. Bayesian shrinkage resolves both at once: `shrunk_rate` is a weighted average of a fighter's own `raw_rate` and the full population's `population_mean_rate` (currently ≈ −1.30, dragged negative because 39.5% of all 2,555 scored fighters have 3 or fewer fights and average −2.43 — most short UFC careers end on a loss, not a win), weighted by `n_fights` vs. `prior_strength` (10.0, in equivalent-fights units). A 1-fight career gets pulled almost entirely to the population mean (Shamrock: 5.10 → −0.72); a 22-fight career like GSP's barely moves (7.04 → 4.43) because real evidence outweighs the prior 2-to-1. Shrinkage alone still isn't a complete fix at the margin, though — Shavkat Rakhmonov's 7-fight, `shrunk_rate`-only ranking (unfiltered) would place him #37 overall, ahead of many 15+ fight veterans — which is exactly why `min_fights=10` exists as a hard floor on top of the shrinkage, rather than relying on the prior alone to argue every small sample back into line.

### 4b. Fighter status
`create_fighter_status_dataset(df, as_of=None)` — active/inactive per fighter (fought within 730 days of `as_of`). `as_of` defaults to `datetime.now()`, making the result non-deterministic across runs by design (unlike everything else in this project, "is this fighter still active" is genuinely an as-of-today question) — pass `as_of` explicitly for a reproducible cutoff.

### 4c. Finishing rate, power, and durability-adjusted power — three deliberately separate metrics
None is blended into the others; each answers a different question:
- **`calculate_overall_potency`** ("who finishes fights") — win-conditioned: `finishes / wins` per fighter/weight-class (striking and grappling computed separately, combined by geometric mean), shrunk toward each division's own baseline. Excludes injury-cause stoppages from the finish count (checks `details` for "injury").
- **`calculate_striking_power`** ("who hits hardest") — NOT win-conditioned: `kd / head_strikes_landed` per fighter/weight-class, same shrinkage machinery and injury filter, plus the `is_no_score_fight` exclusion (DQ/Overturned as well as NC — matching `filter_invalid_rematches`). A knockdown the opponent survives counts exactly as much as one that ends the fight. Returns one unfiltered row per (fighter, weight class) — a building block; `.nlargest()` on it directly surfaces 1-strike careers and thin pseudo-divisions, so rank via `get_power_by_weight_class` (`min_head_strikes_landed`).
- **`calculate_durability_adjusted_power`** — `calculate_striking_power`, reweighted per-fight by `add_opponent_durability_multiplier`: a bounded `[0.5, 2.0]` multiplier from the OPPONENT's own pre-fight standing-KO/TKO-loss rate, shrunk toward *that fight's own weight class* baseline (not one dataset-wide number — this rate spans ~8x across divisions, HW 0.64% to WSW 0.08%, global 0.33%). Neither `calculate_striking_power` nor `calculate_overall_potency` incorporates this adjustment; only `calculate_durability_adjusted_power` does.
- Shared helper: `_shrink_rate(count, total, prior_strength, prior_rate)` — same shape as `ude_points_algorithm.shrunk_win_rate`, generalized to an arbitrary count/total pair. Prior strengths are in the units of each ratio's own denominator, so they differ by metric: `RATE_SHRINKAGE_PRIOR_STRENGTH=5.0` for potency (denominator is *wins*), `POWER_SHRINKAGE_PRIOR_STRENGTH=200.0` for striking/durability-adjusted power (denominator is *head strikes landed*; a qualified fighter's median is ~200, so the old shared `5.0` shrank by ~1% and left the unfiltered top full of 1-strike careers), `DURABILITY_SHRINKAGE_PRIOR_STRENGTH=300.0` for the opponent-durability multiplier (denominator is *standing strikes absorbed*; ~3x the ~90-strike median opponent history).
- **`add_opponent_durability_multiplier` — how much the bound binds is a calibration choice, tracked here.** For the ~72% of fighter-fights whose opponent has no prior standing-KO loss, the multiplier reduces algebraically to `1 + absorbed / prior_strength` and the division baseline cancels out — so `prior_strength` alone sets how fast it ramps to the `2.0` ceiling. At the former `15` it hit the ceiling after 15 absorbed strikes (~63% of all observations pinned at a bound, division baseline inert for three-quarters of the data). At `300` the ceiling needs a genuinely long clean record: ~11% pin, and the division baseline drives the majority of fights. Mean-based shrinkage (not median) is deliberate — there is no clean Bayesian formulation for median-based shrinkage.

### 4d. Rematch history
`process_rematch_data(df, exclude_no_contests=False)` (+ `find_same_winner_rematches`) — every fighter-pair rematch, whether each meeting was immediate (no intervening fight for either side since their last meeting), and the winner. `assign_winner` returns distinct `'draw'`/`'no_contest'` sentinels rather than a collapsed `None`, so two draws between the same pair can't spuriously register as a repeat win. `filter_invalid_rematches` reuses `ude_points_algorithm.is_no_score_fight` (checks both `fight_result == 'NC'` and `method in {'DQ', 'Overturned'}`) instead of a narrower method-string-only check.

### 4e. Opponent-similarity matching
`find_most_similar_past_opponents(df, fighter_name, future_opponent_name, exclude_future_opponent=True, as_of=None, min_columns=None)` — for a fighter's upcoming opponent, ranks their own past opponents by similarity to that opponent. **Physical** (age, height, reach) and **style** similarity are reported separately, not blended into one score — a fighter can be a close physical match while fighting a completely different style. `total_difference` is the mean of the *available* `|diff|` values per row; because mean-of-available grades a data-poor candidate on an easier subset, each row also carries `n_columns_compared`, and any row below `min_columns` (default: all columns for physical, `len − 1` for style — currently 26 of 27) gets a `NaN` `total_difference` and sorts last rather than winning on a thin subset; checked directly against a real call (Makhachev/Della Maddalena) rather than assumed to still hold as the style-column count grew from 6 to 25 to 27 — 16 of 18 past opponents clear the floor, unchanged across both re-checks, the 2 that don't are opponents with no prior cumulative data at all (a UFC debut), the intended behavior, not the floor being too strict. `as_of` restricts every input — the future opponent's profile and the past-opponent set — to fights strictly before that date; without it a retrospective call silently uses the future opponent's stats as they stand today. `stance_match` (`'same'`/`'different'`/`'unknown'`) and `fight_date` are attached alongside the score, not folded into the distance. `exclude_future_opponent=True` (default) drops the future opponent's own past meeting(s) with the fighter from their own comparison set — without it, a fighter who's already fought their upcoming opponent trivially ranks that real fight as the "most similar" match to itself.

**Scaling (2026-09): population-referenced robust z-score, division-scoped, not per-call min-max.** `_compute_robust_scale_reference(df, columns, weight_class=None, min_division_observations=300)` gives each compared column a scale of `1.4826 × MAD` (median absolute deviation) computed from real fighter-fight observations — from `weight_class`'s own division when it has ≥300 non-null observations for that column and a non-degenerate (nonzero) MAD, promotion-wide otherwise. `find_most_similar_past_opponents` passes the **future opponent's own division** — the relevant context is "similar by the standards of the fight being prepped for." A past-vs-future difference is then `(past_value − future_value) / scale`; the median cancels out of a plain difference, so only the spread matters here. Computed fresh from `df` every call (never cached), so a retrospective (`as_of`-filtered) call automatically scales against the population as of that date, with no separate temporal-scoping logic needed.

Both the scope and the statistic were decided from the data, not guessed. **Scope:** per-column SD checked by division vs. promotion-wide across all 28 comparison columns — physical measures barely differ by division (1.25–1.53× spread), overturning the initial assumption that they would; `dynamic_kd_rate` differs 7.35× (heavyweight power variance dwarfs bantamweight's) and the grappling-usage columns differ ~2×. Division-scoping costs nothing where it doesn't matter and fixes real distortion where it does, rather than special-casing which columns get it. **Statistic:** skew and outlier rate checked on all 28 columns — physical columns are clean (|skew| < 0.3, <1% outliers), but most style columns are not (`dynamic_kd_rate` skew 13.3; several usage columns skew 3–6). A robust (median/MAD) scale applied *uniformly* — rather than plain z-score for the clean columns and robust for the skewed ones — gives the needed outlier resistance without reintroducing a mixed-footing average across columns; verified this costs nothing on the clean columns (age's plain SD, 4.14, and its robust estimate, 4.21, are functionally identical). One real edge case found and guarded: MAD is exactly 0 in 4 division×column cells (`dynamic_kd_rate` in three lower-power divisions, `dynamic_sub_att_rate` in HW, where most fighters share the same near-zero rate) — falls back to that column's promotion-wide scale rather than dividing by zero.

This replaces per-call min-max scaling (over just `{this fighter's own past opponents} ∪ {the future opponent}`, 5–20 values), which had three confirmed failures, each re-verified fixed against the exact real cases that first exposed them: scores weren't comparable across different calls; removing one unrelated candidate from Makhachev's real candidate set used to shift every other score by up to 30% and reorder 8 of 19 ranks — re-run after the fix, removing a candidate now changes every other score by exactly 0.00%; and a 2-past-opponent career (Aaron Rosa) used to force every scaled difference to a degenerate 0 or ±1 — now produces real, non-trivial values (0.61, 0.97).

**Style similarity is grouped into named axes (`STYLE_SIMILARITY_AXES`), not one flat list.** `calculate_style_similarity`/`calculate_similarity_differences`' `axis_map` parameter additionally reports `axis_<name>_difference`/`_n_compared` for each axis, alongside the unchanged flat `total_difference`/`n_columns_compared` (still what the ranking sorts on) — added once the style column count grew from 6 to 25, since a single flat mean at that size implicitly overweights whichever theme happens to own the most columns and collapses "similar in power, different in grappling" into one number that can't say which. Five axes, `STYLE_SIMILARITY_COLUMNS` is the flattened union:
- **`power`** (1 col) — `dynamic_kd_rate`: cumulative knockdowns / head strikes landed, the same formula `calculate_striking_power` uses, reshaped into a per-fighter snapshot.
- **`output_pace`** (2 cols) — `dynamic_sig_strikes_attempt_rate`, `dynamic_td_attempt_rate`.
- **`striking_placement`** (10 cols) — `dynamic_{head,body,leg,clinch,distance}_strikes_accuracy`/`_defence`. Target mix (head/body/leg — *where on the body*) and phase mix (clinch/distance — *how well while free to strike*) grouped together, since both answer "where and how well do they land."
- **`phase_preference`** (6 cols) — `dynamic_{ground,clinch,distance}_strikes_share` and the matching `_attempt_rate`. Pure usage — how much of a fighter's output happens in each position — deliberately separate from position *accuracy* above: share and rate answer different questions (rate is volume-sensitive, share isolates preference from pace) and neither substitutes for the other, so both are kept.
- **`grappling_control_and_finishing`** (8 cols) — `dynamic_td_accuracy`/`_defence`, `dynamic_ctrl_time_share`, `dynamic_sub_att_rate`, `dynamic_ground_strikes_accuracy`/`_defence`, `dynamic_ground_strikes_per_control_minute`, `dynamic_sub_attempts_per_control_minute`. Ground-strike quality sits here, not in `striking_placement` — "what a fighter does once it's on the ground" (takedown entries, control, strikes, submissions) is one story, and ground-strike output specifically is what separates real ground-and-pound (high control time AND high ground-strike accuracy/output) from a wrestler who controls without striking much once there. `dynamic_ctrl_time_share` is a fraction of total cage time (bounded [0,1]), not a per-time rate like the others in this axis — control time is itself bounded by the fight's own duration, so it's framed as share, not frequency. The two `_per_control_minute` columns (2026-09) answer a different question again from either share or the total-fight-time rate columns: "how busy is this fighter once they actually have someone controlled." Khabib Nurmagomedov and Jailton Almeida have a similar total-fight-time `dynamic_ground_strikes_attempt_rate` (3.08 vs. 2.79) despite Almeida's much higher `dynamic_ground_strikes_share` (0.80 vs. 0.38), but Khabib's `dynamic_ground_strikes_per_control_minute` (5.54) is ~60% higher than Almeida's (3.51) — Khabib is the higher-output ground striker once on top; Almeida holds control longer (higher `dynamic_ctrl_time_share`) and converts more of it into finish-hunting (higher `dynamic_sub_att_rate`) than volume striking. Floored at `MIN_CONTROL_MINUTES_FOR_RATE = 5.0` cumulative control-minutes (below it, both columns are `NaN`) — picked by checking the full population's max/p95 rate by control-minute bucket, not guessed; comfortably excludes the case that surfaced the need for it (Alex Pereira, 2.6 minutes of career control, had produced an unstable 30.000 reading). Caveat that remains, not fixable from the columns available: above the floor, the numerator still isn't strictly "while in control" — a strike or submission attempt can happen from the bottom or mid-scramble, since UFCStats' box score doesn't separate the two. Full detail in `data_dictionary.md`.

Validated against a 7-fighter face-validity test before shipping (Israel Adesanya, Michael Page, Petr Yan, Merab Dvalishvili, Khabib Nurmagomedov, Jailton Almeida, Alex Pereira) — the axes separate all seven along their real reputations: Almeida is the extreme outlier on ground share (0.80) and control share (0.79); Merab leads every fighter on output pace (10.2 strikes/min, 16.9 TD attempts/15min) while posting the lowest power reading (0.001); Khabib and Almeida both show high ground-strike accuracy (0.73, 0.72) alongside their high control shares, distinguishing genuine ground-and-pound from a stalling profile that would show high control with low ground-strike output; Pereira and Adesanya are the two highest on power (0.022, 0.021) with Pereira lowest on grappling engagement. One confirmed data-scope limitation, not a metric defect: Michael Page reads `dynamic_kd_rate = 0.000` despite a real highlight-reel knockout reputation — his 6 fights in this (UFC-only) dataset are all recent (2024–2026) and none ended in a knockdown; his finishing reputation is almost entirely pre-UFC (Bellator) and isn't in this data at all.

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
- **Identification is real modeling work.** A new calibration function is needed — analogous to `calibrate_age_effects` — that estimates weight-class × era strength offsets from the bridge-fighter network, with the same strict-temporal, no-future-leakage discipline already enforced elsewhere, and an explicit NaN-safe fallback for under-connected cells (the exact bug class already found and fixed once in `_age_multiplier` — see `data_integrity_and_invariants.md` — must be designed against from the start here).
- **Some cells have no data by construction, not by gap.** Featherweight and bantamweight didn't exist in the UFC before ~2011 (0 unique fighters in 2001–2003 in this dataset, 129–140 by 2022–2024) — there's no "true" era-strength value to estimate for a division that didn't exist yet, only a defined "not applicable" state.
- **Measurement validity, not just volume.** Rules, judging criteria, and round/format standards changed materially over 1999–2026; a "dominant" PDI performance under one era's judging isn't necessarily comparable to another's, and no amount of additional fight-count data fixes that — it's a property of the sport's history, not an engineering gap.
- **No ground truth.** There's no objective answer to "was 2003 heavyweight weaker than 2024 lightweight," so any model needs face-validity checks against combat-sports historical consensus before being trusted, not just a clean fit statistic.

Net: buildable, and the data supports a first version — but it's a v3-scale project (new calibration subsystem + validation framework), not something to fit into v2.6's maintenance.

---

## 7. Current Source Files

**What's usually shared alongside this document:** `dataset_processing_pipeline.py`, `ude_points_feature_engineering_pipeline.py`, `ude_points_algorithm.py`, and `current_df.csv`. These four are documented in full below. Everything else exists only in the project's GitHub repo (`OsyTheDataGuy/ude_points_util`) and is listed in one line each at the end of this section — detailed elsewhere in this document (§4, §8), not repeated here.

**Folder layout:** the main folder holds the active pipeline files (below) plus `current_df.csv` and `fighters_df.csv`, both live files kept current by the automated pipeline (§8). No other dataset snapshots or notebooks remain locally — earlier superseded snapshots (`v2_6.csv`, `v2_6_with_phase_profiles.csv`, the raw ETL-stage `..._ready_for_features.csv`, the old `v2_5.csv`) have been deleted, not archived — `current_df.csv` is rebuilt end-to-end from raw columns on every regeneration (§3), so no intermediate snapshot is authoritative.

**Pipeline order:** `fighter_scrape_new.py` + `ude_scrape_new.py` (acquisition, incremental) → `dataset_processing_pipeline.py` (raw scrape → 1-row-per-fight) → `ude_points_feature_engineering_pipeline.py` (→ PDI/chronological features) → `ude_points_algorithm.py` (→ UDE points) → `ude_points_utils.py` (→ rankings/career views). `run_refresh.py` orchestrates the ETL-through-scoring half of this chain as one call (§8); the two scrapers run as separate steps before it.

* ```text dataset_processing_pipeline.py ``` — ETL: merges raw scraped fight/event/fighter-bio data into one row per fight (`run_etl_pipeline`). Drops and reports (does not silently discard) any row where `event_date` is null — `drop_rows_with_null_event_date`, called twice inside `run_etl_pipeline`: right after column standardization (catches a failed event-date join on the freshly-scraped data) and again right after the optional `current_dataset` merge (catches null/unparseable dates already sitting in the historical data being appended to — a fresh scrape's own check can't see those, since they enter the pipeline later). `ude_points_feature_engineering_pipeline.engineer_all_features` calls the same function again defensively at its own entry point, in case its input didn't come through this ETL step at all. `bio_cols` (step 6) carries `Height (m)`/`Weight (lbs)`/`Reach (in)`/`Stance` through from the raw fighter-bio scrape into the merged fight dataset. The final column-ordering step (step 12) appends any column not named in its `ordered_columns` list rather than dropping it, since by that point every column present was already deliberately constructed by an earlier step — the list's job is establishing a readable order, not deciding what belongs in the output. `convert_to_one_fight_one_row`'s `.nth(0)`/`.nth(1)` calls use `reset_index(drop=True)`, so no stray row-index column survives the pivot to one-row-per-fight. `validate_transformed_data` (called inside `run_etl_pipeline`, on the freshly-scraped batch before `current_dataset` is merged in) checks primary-key uniqueness, landed≤attempted, age/control-time bounds, join-leakage nulls, and — when `current_dataset` is passed in — that none of the batch's `fight_url`s already exist there. `validate_dataset_regeneration(old_df, new_df, key_col='fight_url', columns_expected_to_change=None)` is a separate function for a different pipeline stage: it diffs a prior fully-processed/scored dataset against a freshly regenerated one, raising on any lost key or any changed column not explicitly listed as expected to change.

* ```text ude_points_feature_engineering_pipeline.py ``` — Generates chronological state and PDI fight-performance features. Scoring inputs (`pdi_margin` and everything derived from it) are unchanged since v2.6's lock. The file itself is not: `calculate_phase_magnitude_and_pdi` was edited post-lock to fix `decisive_wins`/`close_wins`/`ties` misclassification (`data_integrity_and_invariants.md`) by rounding a separate copy of the phase magnitudes before classification bucketing — verified to leave `pdi_margin` and every scored output bit-identical.

* ```text ude_points_algorithm.py ``` — Authoritative UDE scoring implementation. Unchanged since v2.6's lock.

* ```text current_df.csv ``` — **Current production file**, and the file the automated weekly pipeline (§8) reads and overwrites in place. 8,590 fights × 258 columns as of the last refresh; includes `Stance_fighter_1`/`Stance_fighter_2` (~98% coverage, backfilled from `fighters_df.csv` by `fighter_url`) — an additive enrichment, not a scoring change; v2.6's lock still applies. There is only ever one file under this name: each merged refresh PR replaces its content with that cycle's newly scored output (§8) — it is not a fixed snapshot and its row count grows over time.

**Also in the GitHub repo, not routinely shared here:**
* `ude_points_utils.py` — rankings/career/similarity utilities (§4a–4e); imports `is_no_score_fight` from `ude_points_algorithm.py`.
* `fighters_df.csv` — fighter bio source (Height/Weight/Reach/Stance/DOB) keyed by `URL`, 4,614 fighters as of the last refresh, kept current by the automated pipeline; carries a `bio_scrape_attempts` retry-cap column (§8).
* `fighter_scrape_new.py`, `ude_scrape_new.py`, `run_refresh.py` — the automated acquisition/refresh scripts (§8).
* `requirements.txt`, `.github/workflows/refresh_dataset.yml` — CI dependency list and the workflow definition (§8).

---

## 8. Automated Production Pipeline

The weekly refresh described in earlier sections (`fighter_scrape_new.py` → `ude_scrape_new.py` → ETL → scoring, previously run by hand) is now automated via GitHub Actions, in the project's repo (`OsyTheDataGuy/ude_points_util`). Live as of 31 August 2026.

**Trigger:** `.github/workflows/refresh_dataset.yml` fires on the manual "Run workflow" button (`workflow_dispatch`) or automatically every Tuesday 06:00 UTC (`schedule`). The schedule only fires once this file is on the repo's default branch — a schedule defined on an unmerged branch/PR is never registered.

**Pipeline, in order:** fetch the latest `ufc_fighter_tott.csv`/`ufc_fight_details.csv`/`ufc_event_details.csv` from greco1899's GitHub → `ude_scrape_new.py` (incremental: only fights not already in `current_df.csv`) → `fighter_scrape_new.py` (incremental: only new fighters, fighters who just fought — the "active" trigger, since UFCStats' listed Weight tracks current division — or fighters with an incomplete bio profile still under their retry cap) → `run_refresh.py` (chains `run_etl_pipeline` → Stance backfill → `engineer_all_features` → `calculate_ude_points_with_ablation` → `add_ude_points_difference_columns` → `validate_dataset_regeneration`) → open a PR.

**Incremental-scrape design (`fighter_scrape_new.py`):** a fighter with a genuinely permanent bio gap (e.g. UFCStats never measured an older/retired fighter's Reach — true for ~43% of the roster) is capped at `MAX_INCOMPLETE_RESCRAPE_ATTEMPTS = 3` re-scrape attempts via a persisted `bio_scrape_attempts` column in `fighters_df.csv`, so the "incomplete" trigger doesn't re-visit the same few thousand permanently-incomplete fighters on every single run.

**Validation gate:** `validate_dataset_regeneration` compares the newly regenerated file against the prior production file and raises (failing the job, before any PR is opened) on any lost `fight_url` or any changed column not explicitly allow-listed. A run that fails here produces no PR at all — check the failed Actions run's log directly in that case, there's nothing to approve.

**PR as the human-confirmation gate:** on success, the workflow opens a PR (`peter-evans/create-pull-request`) with the validation summary as its body. Merging the PR is the only manual step in the loop — and merging **atomically updates `current_df.csv` and `fighters_df.csv` in place**: `run_refresh.py --output` writes directly to `current_df.csv` (safe because it's fully loaded into memory before that write happens), so there is no separate "rename the output forward" step anymore. Steady-state loop: PR appears → read the report in its body → merge if it looks sane.
