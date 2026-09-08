# Age calibration — validation of the breakpoint-selection procedure

Scope: the empirical age calibration in `ude_points_algorithm.py`
(`calibrate_age_effects` and the functions it feeds). Question asked: is the
BIC-based breakpoint selection clean, given that the fit uses two
complementary observations per fight?

## Verdict

The two-observation design is **not** the problem — it is fine for estimating
the age effect and does not bias which breakpoint is chosen. The problems are
that (a) the "BIC" selection was not doing what its name implies, (b) the
quantity it selected — a breakpoint age — is not identifiable from this data,
and (c) the likelihood it scored was inflated ~2×. The procedure has been
replaced with a fixed-anchor smooth model whose order (none / linear /
quadratic) is chosen by BIC on the fight count.

## Shortlist

| Verdict | Finding |
|---|---|
| fix-now | #1 Breakpoint age is not identified — selected value swings 26.5↔33.5 between windows sharing 80% of their fights |
| fix-now | #2 "BIC selection" reduced to maximum-likelihood — the penalty was identical across all candidates and cancelled |
| decide-and-document | #3 Two mirrored rows per fight scored as independent → log-likelihood (hence every BIC/deviance gap) inflated ~2× |
| minor | #4 13 weight-class dummies structurally pinned at ~0 by the mirror design, padding the parameter count |
| minor | #5 Breakpoint estimated but not counted as a parameter; also a non-regular parameter |
| already-handled | #6 `pd.isna` guards on the neutral-calibration state — preserved in the rewrite |

---

### 1. The breakpoint age is not identified — fix-now

- **What:** `calibrate_age_effects` searched a 0.5-year grid of breakpoints and
  kept the one minimising BIC. The profiled log-likelihood is flat across the
  whole grid, so the winner is set by noise.
- **Breaks:** Across the full sample the log-likelihood spread over the entire
  26.5–34.5 grid is 3.9 nats; the selected breakpoint (31.0) beats a breakpoint
  2.5 years away (28.5) by 0.33 nats. Within ~2 nats of the optimum — i.e. not
  distinguishable — is essentially the entire candidate range.
- **Scale:** Every calibrated window (2008–2026 in the current dataset).
- **Example:** Selected `reference_age` by rolling window, old procedure:
  2012 → 32.0, 2013 → 27.0, 2014 → 26.5 (grid floor), 2015 → 30.5, 2016 → 33.0;
  later 2023 → 32.5, 2024 → 32.0, 2025 → 28.0, 2026 → 33.5. Consecutive windows
  share ~80% of their fights. The dependent `own_post_slope` ranged −0.039
  (2017) to −0.106 (2026), a 2.7× swing driven mostly by where the breakpoint
  happened to land.
- **Fix + verdict:** fix-now. There is no breakpoint to find — the age effect
  is a smooth bend, not a hinge (see "What the data supports" below). Removed
  the search; the age term is now measured relative to a fixed anchor age
  (`AGE_ANCHOR_YEARS = 32.0`).

### 2. "BIC selection" was maximum-likelihood selection — fix-now

- **What:** Every candidate breakpoint produced a design matrix with the same
  column count (6 numeric + 13 weight-class dummies = 19), so the BIC penalty
  `k·ln(n)` was identical for all candidates and `argmin BIC ≡ argmax
  log-likelihood`. `calibration_method` was labelled `piecewise_logistic_bic`.
- **Breaks:** `_fit_age_model` (old) — for the full sample, BIC-selected
  breakpoint = 31.0 = maximum-log-likelihood breakpoint, necessarily. No
  complexity trade-off was applied.
- **Scale:** All calibrated windows.
- **Example:** Old candidate grid, full sample: `k = 19` at every step; BIC
  spread across the grid (7.72) equals `2 × log-likelihood` spread (7.72)
  exactly. Separately, BIC *does* reject the piecewise specification when
  actually asked: break@31 BIC 22902 vs a plain-linear age term 22895 — a
  proper BIC comparison prefers **no** breakpoint (ΔBIC +6.9 on the doubled n,
  +12.9 on the honest n).
- **Fix + verdict:** fix-now. BIC now chooses model *order* — `flat` vs
  `linear` vs `quadratic` — which is what BIC is for. On every rolling window
  in the current data it selects `linear` (or `flat` before 2008); the pooled
  full-history fit narrowly selects `quadratic` (ΔBIC 1.4).

### 3. Mirrored observations double the log-likelihood — decide-and-document

- **What:** `_age_calibration_observations` emits two rows per fight — one per
  fighter's perspective — with mirrored covariates and `win` flipped. They are
  deterministically linked (`y_B = 1 − y_A`; win mean = exactly 0.5000) but
  were fit as independent Bernoulli trials.
- **Breaks:** Old `_fit_age_model` used `n = len(y)` = 16,834 when the
  independent-unit count is 8,417 fights. Measured log-likelihood ratio
  (doubled fit vs one-side fit) = 2.007, so `−2·logLik` and every deviance/BIC
  gap between specifications was ~2× too large; any slope standard error is
  ~√2 too small.
- **Does not affect:** the selected breakpoint (the doubled objective is the
  own/opponent-symmetrised profile likelihood — same argmax, confirmed 31.0
  both ways) or the slope point estimates (doubled vs one-side agree to
  ~0.006). This is why the design is kept — it correctly enforces an
  antisymmetric fighter-level effect.
- **Where it bit:** the `canonical_project_state.md` note that the 2011→2012
  regime switch is "indistinguishable from ordinary year-to-year BIC refit
  noise" — that comparison was on the inflated scale.
- **Fix + verdict:** decide-and-document. BIC now uses `n_fights = len(y) // 2`.
  The design matrix is also reformulated so own and opponent age enter only as
  the mirrored difference `d_own − d_opp` (and its square), making the
  antisymmetry exact rather than approximate.

### 4. Weight-class dummies pinned at zero — minor

- **What:** `pd.get_dummies(..., drop_first=True)` added 13 weight-class
  columns.
- **Breaks:** The mirror construction enters one win and one loss in the same
  weight class for every fight, so each weight-class coefficient is forced
  toward 0 (fitted |coef| < 0.065, most < 0.03). They contributed 13 of 19
  parameters and only inflated the reported BIC.
- **Fix + verdict:** minor. Dropped. A weight-class × age *interaction* (do
  heavier fighters age differently?) is a real question but out of scope here.

### 5. Breakpoint not counted as a parameter — minor / latent

- **What:** Old BIC used `n_parameters = len(beta)` and ignored the profiled
  breakpoint, which is both an estimated parameter and a non-regular one
  (likelihood non-smooth in it), so BIC's asymptotics did not cover it.
- **Fix + verdict:** minor — moot now that there is no breakpoint.

### 6. Neutral-calibration guards — already-handled

- The `pd.isna` short-circuits that make the "insufficient prior history"
  state a true 1.0× no-op (see `canonical_project_state.md` §9b) are preserved:
  `_age_logit_offset` guards `anchor_age` being NaN explicitly before any
  arithmetic, and `_neutral_age_calibration` sets `anchor_age = np.nan`,
  `age_gap_linear = age_gap_curvature = 0.0`.

---

## What the data supports about aging

Fit on the two-observation set, opponent quality controlled, honest `n` =
fights. "Decline" = slope of the win-probability logit per year of age.

**The age effect strengthened across eras** (identifiable, monotone, ~2 SE
between first and last era):

| Era | Linear decline per year | Fights |
|---|---|---|
| 2003–2012 | −0.051 ± 0.010 | 1,724 |
| 2013–2018 | −0.061 ± 0.008 | 2,726 |
| 2019–2026 | −0.074 ± 0.006 | 3,833 |

This is the **opposite** of the common "modern fighters sustain their prime
longer" claim — a year of age costs more in the current era, not less.

**The decline also accelerates with age** (concave curve), and more so in the
modern era. Implied per-year decline from the era quadratic fits:

| Era | at age 25 | at age 30 | at age 37 | curvature /yr² |
|---|---|---|---|---|
| 2003–2012 | −0.045 | −0.051 | −0.059 | −0.0006 |
| 2013–2018 | −0.046 | −0.060 | −0.078 | −0.0013 |
| 2019–2026 | −0.052 | −0.068 | −0.091 | −0.0016 |

The curvature is consistent in direction across every flexible fit (spline and
quadratic, every era) but is **not** decisive against a straight line by BIC in
any single window — only the pooled full-history fit clears it, by ΔBIC 1.4.
So: the "past-peak cliff" intuition is directionally real as a smooth bend, but
there is no single age at which it triggers, which is exactly why a
one-breakpoint model had nothing stable to lock onto.

---

## Current procedure (after the change)

`calibrate_age_effects` returns:

| field | meaning |
|---|---|
| `model_order` | `flat` \| `linear` \| `quadratic`, chosen by BIC on `n_fights` |
| `anchor_age` | fixed at `AGE_ANCHOR_YEARS` (32.0); NaN under neutral calibration |
| `age_gap_linear` | per-year logit slope of the age effect at the anchor |
| `age_gap_curvature` | quadratic term; 0.0 unless `quadratic` selected **and** concave |
| `opponent_quality_slope` | opponent-quality control coefficient |
| `bic_by_order` | the three BIC values, for audit |

- Own and opponent age enter the design only as `d_own − d_opp` and
  `d_own² − d_opp²` where `d = age − anchor` — antisymmetric by construction.
- No weight-class dummies. No breakpoint search.
- `_age_logit_offset(age, cal)` gives the signed hit to a person's win-logit
  from being older than the anchor: `0` at/below the anchor, monotone below 0
  above it (a concave quadratic is clamped so it cannot turn back upward).
- Multiplier semantics in `_age_multiplier` are unchanged: opponent side is a
  signed opposition-quality discount, own side is an `abs()` achievement
  reward gated by opponent age, both clamped to `[0.50, 1.50]`.
- BIC per rolling window in the current dataset selects `linear` every year
  from 2008; `flat` 1999–2007 (unchanged fallback). The `quadratic` branch is
  correct but dormant at current per-window sample sizes.

### Rolling-window stability, before → after

| | old `reference_age` / `own_post_slope` | new `age_gap_linear` |
|---|---|---|
| 2012 | 32.0 / −0.093 | −0.060 |
| 2013 | 27.0 / −0.058 | −0.053 |
| 2014 | 26.5 / −0.062 | −0.054 |
| 2015 | 30.5 / −0.063 | −0.047 |
| 2016 | 33.0 / −0.048 | −0.053 |
| 2023 | 32.5 / −0.081 | −0.057 |
| 2024 | 32.0 / −0.084 | −0.064 |
| 2025 | 28.0 / −0.073 | −0.070 |
| 2026 | 33.5 / −0.103 | −0.078 |

The new slope moves smoothly and roughly monotonically (−0.046 → −0.078 over
the full series), tracking the era trend instead of the breakpoint's position.

### Scoring impact

Measured as the change from the pre-recalibration model (searched breakpoint)
to the fixed anchor at 32, on the dataset as it stood then (8,590 fights). It
touches only the age adjustments and their direct downstream — `age_adjustment`,
`own_age_adjustment`, `higher_rated_opponent_bonus`, `upset_rating_gap_smoothed`,
`ude_points_pre_fight` / `_post_fight` / `_diff`. No record, streak, PDI,
method, or dominance column moves.

- Net opponent-age adjustment across the dataset roughly **halves** — the
  anchor at 32 gates out the entire 30–32 band (~18% of fighter-fights) and
  shortens the lever arm above it.
- `age_adjustment` changes on ~3,900 of 8,590 fighter rows, mean \|Δ\| ~0.5
  pts, max ~2.7; the career-total columns move a median ~1 pt (`corr` ≥ 0.99
  with the old values on every column).

**GOAT ranking** (`rank_fighters_by_shrunk_ude_rate`, `min_fights = 10`),
pre-recalibration → current `current_df.csv` (which also carries the later
`is_title_bout` interim-title fix — the two changes push the same direction):

| | pre-recal | current |
|---|---|---|
| #1 | Georges St-Pierre (4.43) | **Jon Jones (4.38)** |
| #2 | Jon Jones (4.27) | Georges St-Pierre (4.37) |
| entered top 20 | — | **Kamaru Usman** (#16), **Khamzat Chimaev** (#17) |
| left top 20 | — | Aljamain Sterling, Petr Yan |
| largest fall | — | Justin Gaethje #9 → #15 (interim-title credit removed) |

Every top-30 fighter gains shrunk rate as the net age penalty lightens — a
level shift, not individual re-rating. Usman and Chimaev enter because they
fought unusually old slates (Usman: 8 of 20 opponents 33+; Chimaev: 6 of 10)
during 2019–2022 when the old breakpoint had drifted to 28.5–31; their mid-30s
opponents were scored 5–7 years past prime and each win discounted 2–3 points,
which the anchor at 32 roughly halves. The current authoritative table is
`canonical_project_state.md` §2a.

### The anchor age (`AGE_ANCHOR_YEARS = 32.0`)

This is the age below which no age adjustment applies, and the reference point
the effect is measured from. It is **not** an empirically located threshold —
the whole reason the searched breakpoint was removed is that the fight data
does not identify one; win-probability decline is smooth and continuous from
the early 20s (≈ −0.036/yr) through 40 (≈ −0.094/yr), with no plateau or knee.

The anchor is a modelling choice about *where the UDE correction should engage*.
Bounds on the defensible range:

- Median fighter-fight age is 30.3; p75 is 33.3. An anchor below ~29 starts
  docking fighters at the population centre; an anchor at 35 sits at the ~87th
  percentile and silences the adjustment for 86% of fighter-fights, including
  nearly every 31–34-year-old championship-era bout.
- The old searched `reference_age` ranged 26.5–33.5 across 2008–2026 (mean
  ≈ 30.5).

**32** was chosen at the top of that range: the adjustment engages only once a
fighter is clearly past prime, not merely older than the median. Scoring is
not knife-edge in the anchor — moving it a year changes `age_adjustment` by
~0.15 pts per fight on average — but the higher the anchor, the more the
scores depart from the pre-recalibration baseline (30 would have moved the
top-20 set less than 32 does).

For the linear model actually selected every window, the fitted
`age_gap_linear` is invariant to the anchor (`d_own − d_opp` = `own_age −
opp_age` regardless of centering); the anchor only sets the gate and the
lever arm.

### To reconsider later

- If per-window samples grow enough that BIC starts selecting `quadratic`, the
  curvature path is already live and clamped for monotonicity. (Only the
  quadratic decomposition — not the fitted curve or the selection — depends on
  the anchor.)
