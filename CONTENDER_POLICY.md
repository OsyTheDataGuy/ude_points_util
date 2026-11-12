# Contender qualification and matchmaking

The public API is re-exported by `ude_points_utils`. Its implementation lives in
`ude_contender_utils` so the policy can be used without importing plotting code.
No network access, source-specific ranking acquisition or private data is bundled.

1. `current_division_win_run(history, division, as_of)` extracts a complete current
   winning run. Supply a single-fighter ledger with date, division, result and
   fight_url. NC preserves the run; L/D and another division interrupt it. Future
   fights are ignored. Fighter identities must be stable IDs upstream. The row
   that ended the run is in `result.attrs['stopped_by']`.
   This is a newer, lower-verification construct for contender content. It does
   not replace `update_win_streaks` (`ude_points_feature_engineering_pipeline.py`),
   which stays the central streak logic. A test checks that the two agree for
   every single-division career with no catchweight or open-weight bout
   (1,693 careers at the 7 November 2026 data).
1a. `attribute_catchweight_divisions(fights)` assigns catchweight bouts to a
   division. The data does not record the intended division, so this is an
   inference: take each fighter's nearest non-catchweight fight before and after
   the bout (up to 4 fights). If all are in one division, the bout is `resolved`
   to it; otherwise it is `ambiguous`. Set the history's division from this before
   calling the run function: a resolved catchweight counts in its division and the
   run continues; an ambiguous one keeps the catchweight label, interrupts the run
   and should be flagged for review via `stopped_by`. The "after" fight looks ahead
   in time; that is acceptable for deciding which division a past bout belonged
   to, not for anything presented as known on fight night. At the 7 November 2026
   data, 58 of 84 catchweight bouts resolve (2006-05-27 to 2026-09-12), including
   26 of the 30 since 2023. The 4 ambiguous ones since 2023 are Rebecki vs. Orolbai
   `919e3c5cc6999cb3` (LW/WW), Musayev vs. Orolbai `36ec204f47e4d613` (LW/WW),
   Castaneda vs. Vologdin `552f7cdaf93e1055` (BW/FW) and Fairn vs. Cavalcanti
   `2a48d0a98c21e736` (WBW/WFW).
2. `assess_contender_qualification(wins, as_of, policy=ContenderPolicy())` evaluates
   the entire run. Each win needs date, opponent_rank, rank_status and rank_source.
   Rank 0 is champion; 1–15 are incoming ranks; null is unknown unless explicitly
   unranked. Verify identity, division and the pre-bout ranking edition upstream.
   Never substitute today's ranks. Missing data can produce `ranking_review`.
3. `prioritize_title_contenders(assessments, criteria=...)` orders only qualified
   fighters. Default descending criteria: recent top-five wins, all current-run
   top-five wins, current-run ranked wins, run length, latest win date. This is an
   explicit editorial order, not an objective deservingness score. Equal evidence
   shares priority; stable fighter ID only orders display. Zero/many qualifiers
   are valid. Rankings/bookings do not change qualification here.
4. `fight_down_pool_decision(qualification, booking_status=..., ...)` excludes
   every title-qualified fighter regardless of priority, and independently checks
   booked/unchecked status, champion status and unavailable/departed roster status.
   It returns all reasons. Unresolved title cases are held for review. A successful
   no-booking listing check is not proof of medical availability.

Defaults: title qualification requires 3+ current division wins including an
incoming top-five/champion win within the past 12 calendar months. A top-five
test requires 2+ division wins and an incoming top-15/champion win in that window.
Only the ranked win must be within the window. Both windows include the boundary
date and use the explicit local fight-data cutoff, never the execution date.
These are creator preferences, not official UFC rules.

An expired top-five-win window is flagged without automatically extending it.
The flag does not establish why the fighter waited or that the fighter had once
qualified. Reigning-champion cross-division exceptions require explicit editorial
review; passing `cross_division_champion=True` does not grant a shot automatically.

The ranking adapter should remain separate: it validates incoming ranks and
provides evidence; these functions apply the rule. Matchmaking/rendering code
should consume the returned decisions instead of reimplementing the thresholds.
Caller-supplied current ranks, bookings and their observation dates should be
retained alongside the report, separately from the fight-data cutoff.

Run `python -m pytest test_contender_policy.py -q`. Rank evidence in the tests is
synthetic; the catchweight and streak cross-checks read the public `current_df.csv`.
Private historical ranking exports must not be committed.
