# Data Dictionary — Columns Used in Content

Reference for every column touched while building `content_bank.md`, so using one again means looking it up here instead of re-reading the pipeline and trusting memory. Not exhaustive over all 200+ dataset columns — only what's actually been used. Add an entry the first time a new column gets used in content, before it's used a second time.

**Verification tiers, stated per entry — don't round one up to the next without doing the work:**
- **Numerically verified** — reconstructed from more primitive inputs and matched to a real row's stored value.
- **Verified vs. known fact** — checked against an external ground truth (public record, UFCStats.com) rather than rebuilt from other columns.
- **Read from code only** — understood from the source function, not yet independently rebuilt or checked. Treat as provisional.

---

## Raw, single-fight columns (safe to use directly — no pre/post ambiguity)

| Column | Meaning | Source | Verification |
|---|---|---|---|
| `fight_result_fighter_1`/`_2` | W/L/D/NC for *this* fight | `dataset_processing_pipeline.clean_modified_df` | Numerically verified (used as ground truth throughout) |
| `sig_strikes_landed`/`_attempted_fighter_1`/`_2` | **Total** significant strikes this fight — standing + clinch + **ground combined**. Not the same as `standing_sig_strikes_landed` below; conflating the two is exactly the error made mid-session. | Raw UFCStats scrape, split by `apply_split` | Numerically verified — Hendricks/Lawler II: 111/228 vs 116/201 |
| `standing_sig_strikes_landed`/`_attempted_fighter_1`/`_2` | Significant strikes landed **standing only** (distance + clinch), **excludes ground strikes**. This is the field `pdi_margin`'s striking phase actually reads. | `add_standing_sig_strikes_columns` = `distance_strikes + clinch_strikes` ([ude_points_feature_engineering_pipeline.py:565](ude_points_feature_engineering_pipeline.py:565)) | Numerically verified — Hendricks 110 vs Lawler 90, reconstructed `pdi_margin` from it and matched stored value exactly |
| `ctrl_in_secs_fighter_1`/`_2` | Total control time, this fight, in seconds | `process_control_time` ([dataset_processing_pipeline.py:179](dataset_processing_pipeline.py:179)), MM:SS → seconds | Numerically verified — Hendricks 620s vs Lawler 62s, part of the pdi_margin reconstruction |
| `td_landed`/`_attempted_fighter_1`/`_2` | Takedowns, this fight | Raw scrape | Numerically verified |
| `sub_att_fighter_1`/`_2` | Submission attempts, this fight | Raw scrape | Numerically verified |
| `kd_fighter_1`/`_2` | Knockdowns scored, this fight. **A discrete box-score event, not a continuous power measurement** — a fight-ending strike doesn't necessarily log a `kd` (Pereira's Oct 2025 finish of Ankalaev shows `kd=0` despite ending the fight by strikes). | Raw scrape | Verified vs. known fact (Pereira/Ankalaev case) |
| `fight_day_age (yrs)_fighter_1`/`_2` | Fighter's exact age on this fight's date | `calculate_age` ([dataset_processing_pipeline.py:163](dataset_processing_pipeline.py:163)), computed once at ETL from DOB + event_date | Read from code only — no ambiguity to verify (deterministic date math) |
| `STANCE_fighter_1`/`_2` | Orthodox/Southpaw/Switch/Open Stance/Sideways — a **roster-level bio property**, not fight-specific. Backfilled by `fighter_url` for historical fights (97.8% coverage). | `process_fighter_bio`, backfilled by `fighter_url` from `fighters_df.csv` | Verified vs. known fact for one spot-check (Thiago Santos = Orthodox, confirmed live against UFCStats.com) — not a general audit of the field |
| `Reach (in)_fighter_1`/`_2`, `Height (m)_fighter_1`/`_2` | Bio properties, same roster-level nature as STANCE | `process_fighter_bio` | Read from code only |
| `event_date` | Date of the fight (shared column, no fighter suffix) | ETL merge against `events_df` | Numerically verified (used for all chronological sorting) |
| `weight_class` / `weight_class_cleaned` | Raw string / short code (LW, WW, etc.) | `map_weight_class`, longest-substring-first match ([ude_points_feature_engineering_pipeline.py:30](ude_points_feature_engineering_pipeline.py:30)) | Read from code only |
| `method` / `method_mapped` | Raw result detail / collapsed to `{Finish, UD, MD, SD}` | `map_fight_method` | Read from code only |
| `is_title_bout` | 0 = no title bout, 1 = interim title bout, 2 = undisputed title bout | `create_is_title_bout_column` | Numerically verified — every interim-titled bout (raw `weight_class` containing "Interim") confirmed coded `1`, distinct from undisputed's `2`. Spot-checked: Gaethje–Pimblett (Jan 2026 interim LW) = 1, Topuria–Gaethje (Jun 2026 undisputed unification) = 2. |

## Engineered, pre-fight snapshots (exclude the current row's own fight — the recurring trap)

Sort chronologically by `event_date` **per fighter** before taking a "final" value with `.last()` on any of these, or you get whichever row happens to fall last in file order, not the fighter's most recent fight.

| Column | Meaning | Source | Verification |
|---|---|---|---|
| `career_sig_striking_accuracy_fighter_1`/`_2` | Cumulative sig-strike accuracy **entering** this fight | `update_career_means` ([ude_points_feature_engineering_pipeline.py:279](ude_points_feature_engineering_pipeline.py:279)) | Numerically verified — Gunnar Nelson's full chronological accuracy sequence reproduced exactly from raw sums |
| `dynamic_td_defence_fighter_1`/`_2` (and the `sig`/`head`/`body`/`leg` accuracy/defence variants) | Cumulative rate **entering** this fight, career-to-date | `add_dynamic_strike_accuracy`/`_defence`, `add_dynamic_td_accuracy`/`_defence` | Numerically verified for `dynamic_td_defence` (matched independent raw sum for Jon Jones, 95%) |
| `title_defenses_fighter_1`/`_2` | Cumulative defenses of the *undisputed* title in the current reign, **entering** this fight — an interim-title fight neither counts as a defense nor resets it. **Undercounts a reign's true final total by 1** whenever the fighter's last fight in that division was itself a successful defense — there's no later row to record the post-fight increment. | `update_title_defenses` | Numerically verified as **unreliable for "final/total" claims** — GSP's true WW total is 10, this column's max value gives 9. Always recompute directly from title-bout results for any "most defenses" claim; see the multi-division-champion and title-defense entries in `content_bank.md`. |
| `is_champion_fighter_1`/`_2` | 0 = not champion, 1 = interim champion, 2 = undisputed champion, **entering** this fight, per weight class | `update_champion_status` | Numerically verified — correctly distinguishes an interim reign from an undisputed one (depends on `is_title_bout` above). E.g. Justin Gaethje reads `1` entering his Jun 2026 unification fight against Ilia Topuria's `2`, not two simultaneous undisputed champions of the same division. |
| `pre_fight_record_fighter_1`/`_2_(W-L-D NC)` | Record string **entering** this fight | `update_fight_records` | Numerically verified (internally consistent across every Level-2 thread built so far) |
| `W/L_streak_fighter_1`/`_2` | Signed streak **entering** this fight (+ = win streak, − = loss streak) | `update_win_streaks` | Read from code only for the column itself — for any "current streak" claim, recompute directly from the full chronological result sequence instead (done for Islam/Silva; matches the public record of 17 and 16) |
| `quality_score_fighter_1`/`_2` | Pre-fight opponent-quality score (shrunk win rate + champion/defense bonus) | `add_quality_score_columns` → `ude_points_algorithm.quality_score` | Read from code only |

## Engineered, single-fight only (do not confuse with the "dynamic" cumulative versions above)

| Column | Meaning | Source | Verification |
|---|---|---|---|
| `sig_strikes_defense_fighter_1`/`_2`, `td_defense_fighter_1`/`_2` | Defense rate in **this one fight only** — despite the naming similarity to `dynamic_td_defence`, this is not cumulative. | `add_defense_columns` | Numerically verified — not cumulative (contrast `dynamic_*_defence`) |

## Engineered composite — drives actual scoring

| Column | Meaning | Source | Verification |
|---|---|---|---|
| `pdi_fighter_1`/`_2`, `pdi_margin_fighter_1`/`_2` | Sum of 5 independently-computed **signed** phase magnitudes for this fight — striking (standing only), control, takedowns, submissions, knockdowns — each roughly bounded to [-1, 1] (tighter under low-volume floors). Range: -5 to +5. **A positive margin does not mean every phase was won** — wins just have to outweigh losses in the sum. | `calculate_phase_magnitude_and_pdi` ([ude_points_feature_engineering_pipeline.py:661](ude_points_feature_engineering_pipeline.py:661)) | Numerically verified — reconstructed all 5 phase magnitudes independently for 5 real fights and matched the stored `pdi_margin` exactly every time (see `content_bank.md`'s robbery-leaderboard entry) |
| `decisive_wins`/`close_wins`/`close_losses`/`decisive_losses_fighter_1`/`_2` | Per-phase classification of the same 5 magnitudes: `>0.35` decisive, `0`–`0.35` close, mirrored for losses, `0` = tie | Same function | Numerically verified the classification rule itself — but confirmed via `grep` that **these columns are computed and stored, never read anywhere in `ude_points_algorithm.py`'s actual scoring**. Diagnostic only. |
| `dominant_fighter`, `phases_won` | Categorical: wins ≥3 of a **different** 5-category split (striking/wrestling/grappling/control/standing-danger) and more than the opponent | `add_dominance_columns`/`add_who_won_col` | Read from code only. Used in scoring **only** as `pdi_margin`'s fallback when it's missing/NaN — otherwise unused, kept for narrative/visualization. Don't conflate with `pdi_margin`'s own phase magnitudes — different inputs entirely. |

## Phase-percentile columns (`..._with_phase_profiles.csv` only)

| Column | Meaning | Source | Verification |
|---|---|---|---|
| `phase_{metric}_pre_fight`/`_post_fight_fighter_1`/`_2` (12 metrics) | Cumulative raw value entering / including this fight | `add_phase_profile_raw_columns` | Read from code only |
| `phase_{metric}_..._pctile_full`/`_era_fighter_1`/`_2` | Percentile rank vs. full history, or vs. same-weight-class contemporaries in a 3-year window | `add_phase_profile_percentiles` | **Read from code only / face-validity checked** (Edson Barboza's leg-kick-volume era percentile "looked right" for a known leg-kicker; the percentile math itself was not independently rebuilt). Lighter verification tier than the raw-count nuggets in the same batch — say so if this is ever the sole basis of a claim. |

## UDE scoring output (referenced, not itself part of the growth-content dataset work)

| Column | Meaning | Source |
|---|---|---|
| `ude_points_pre_fight`/`_post_fight_fighter_1`/`_2` | Running UDE rating | `calculate_ude_points_with_ablation`, locked per `canonical_project_state.md` §2a |
