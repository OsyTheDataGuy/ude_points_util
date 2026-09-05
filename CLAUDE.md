# Project instructions — UDE Points

## Reuse existing pipeline/utils logic before writing new analysis code

Before hand-rolling any transformation, aggregation, or metric for a research/analysis task, check whether `ude_points_utils.py`, `ude_points_feature_engineering_pipeline.py`, or `ude_points_algorithm.py` already computes it (or something close enough to adapt) — and reuse or call the real function rather than reimplementing its logic from scratch.

**Why:** reimplementing logic that already exists risks silently diverging from how the real code actually behaves, and that risk is not hypothetical — it happened twice in one session (2026-09-05):
- Reusing the real `update_champion_status`/`update_title_defenses` (instead of reimplementing their logic) is what correctly identified two separate bugs and confirmed their exact downstream impact.
- Hand-simulating `multi_division_championship_bonus`'s logic instead of calling it with its real input produced a false "14 confirmed false positives" claim — the real function reads `weight_class_cleaned`, the simulation used raw `weight_class`. Caught only because the user demanded the claim be reverified against the actual code.

**Where to look first:**
- `ude_points_utils.py` — fighter-career trajectories (`create_fighter_career_dataset` and its helpers, including the generic fighter/opponent column-extraction pattern), rankings (`rank_fighters_by_shrunk_ude_rate*`), fighter status, potency/power/durability metrics, rematch history, opponent-similarity matching.
- `ude_points_feature_engineering_pipeline.py` — champion status, title defenses, streaks, career means, dynamic accuracy/defense, PDI/phase magnitudes.
- `ude_points_algorithm.py` — the actual per-fight scoring components (championship/title-defense/quality bonuses, dominance/age/rematch adjustments).

**The one boundary:** this applies unconditionally to *engineered/feature* functions (champion status, streaks, defenses, dynamic accuracy — factual "what happened"). For the actual *scoring* functions (`quality_score`, `title_defense_bonus`, `dominance_adjustment`, etc.), reuse them when the content is explicitly about UDE Points itself — but for plain factual/comparison content that isn't meant to invoke the model (see `mma_content_strategy.md`'s Level 1/2 vs. Level 3 staging), pulling raw stats directly is often more correct than routing through the scoring layer, since that layer bakes in hand-tuned UDE-specific constants the audience hasn't been introduced to yet.

If no existing function fits and something new must be built (e.g. a metric with no ready-made column), say so explicitly and flag it as a fresh, lower-verification-tier construct rather than silently presenting it at the same confidence as an existing, already-verified column.

## Reuse is the default, not blind trust — check the existing logic too

Reusing existing logic is about not *reimplementing* something that already works, not about accepting whatever it produces without question. When calling an existing function as part of a research task, sanity-check its actual output (real spot-checks against known facts, value_counts that look wrong, a result that contradicts something already verified) the same way any other claim gets verified before it's used. If the existing logic itself looks flawed, say so explicitly and flag it, rather than silently propagating a wrong result because "the code already does this."

This is exactly how the `is_title_bout` bug (`project_history.md` #54 / `canonical_project_state.md` §9) was found in the first place — a routine `value_counts()` check while scoping an unrelated research task showed only `{0, 2}` where the column's own documented meaning implied three values, and that discrepancy was chased down rather than assumed to be fine because the function was already-built, existing code. Reuse the real function *and* keep questioning what it produces — those aren't in tension.
