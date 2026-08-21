"""
UDE Points — Temporal-Calibrated UFC Fighter Scoring

Scores each fighter-side of each fight into a running UDE rating (starting at
500). Calibration (age-decline curve, method x PDI residual) is fit
separately from scoring: production callers pass frozen calibration objects
learned only from data available by their declared cutoff date;
calculate_ude_points_with_ablation() builds a temporal calibration cache
itself when none is supplied, so a fight at time T only ever scores against
parameters learned from fights strictly before T.

Per-fight scoring components, applied in order (see calculate_ude_points_with_ablation):
1. Base result: +/-3 points for W/L, 0 for a draw/no-contest.
2. Championship bonus: 2.0x on title bouts (win or loss).
3. Multi-division bonus: 1.25x for winning a title in a new weight class.
4. Dominance adjustment: continuous PDI-margin-driven multiplier (0.75x-1.30x).
5. Method x PDI residual: small calibrated bonus/penalty for method of
   victory, UD as the reference method (+/-0.75 cap).
6. Title-defense and streak bonuses: each computed off the raw base result
   (not the multiplied total), summed, then jointly scaled by perf_scale.
7. Higher-rated-opponent bonus: additive upset bonus from a smoothed
   trailing rating gap, perf_scale applied internally.
8. Age adjustment (opponent's age, opposition-quality discount) and own-age
   adjustment (fighter's own age, achievement reward) — both empirically
   calibrated, invariant (not perf_scale-scaled).
9. Rematch/revenge adjustment: multipliers for avenging a prior loss and for
   rematch series depth.
10. Opponent-quality adjustment: multiplier from the opponent's shrunk
    pre-fight win rate, champion status, and title defenses.

The final per-fight swing is clamped to +/-ABSOLUTE_SWING_CAP.
"""

import logging
from collections import defaultdict, deque
import pandas as pd
import numpy as np
import statsmodels.api as sm

logger = logging.getLogger(__name__)

# Window (in fights) for the trailing rolling average used by
# higher_rated_opponent_bonus's rating-gap calculation. See that function's
# docstring for why a smoothed rating is used instead of the raw
# instantaneous pre-fight UDE snapshot.
UPSET_RATING_WINDOW = 3

def noop(points, *args, **kwargs):
    return points

ALL_BONUSES = [
    'title_defense',
    'championship',
    'multi_division',
    'higher_rated',
    'streak',
    'age',
    'own_age',
    'rematch',
    'dominance',
    'opponent_quality'
]

# Outcomes that result in no points movement, regardless of who is on which side
NO_SCORE_RESULTS = {'NC'}
NO_SCORE_METHODS = {'DQ', 'Overturned'}

def is_no_score_fight(result, method):
    return result in NO_SCORE_RESULTS or method in NO_SCORE_METHODS

# Unused by the current scoring path (no phases_won interpolation is wired
# in) -- kept in case that interpolation is reintroduced.
TOTAL_PHASES = 5
MIN_PHASES_FOR_DOMINANCE = 3
PHASES_LOW_ANCHOR = 0.85   
PHASES_HIGH_ANCHOR = 1.0   

# Title defense bonus variables. 
# Uses a saturating curve: 1 + CAP*(1 - DECAY^n) to reward additional defenses
# with diminishing returns toward a real ceiling, preventing unbounded exponential growth.
TITLE_DEFENSE_CAP = 0.6884
TITLE_DEFENSE_DECAY = 0.85377

# Circuit breaker cap on total per-fight point swings.
# Prevents uncapped multipliers (championship, dominance, multi_division) from stacking infinitely.
ABSOLUTE_SWING_CAP = 60

def base_points_from_result_method(result, method_mapped):
    """
    Method-neutral fight-night base.  The empirical method x PDI calibration
    supplies the method-specific residual as a separate, bounded component.
    UD is therefore the reference method at the base-score level.
    """
    if result == 'W':
        return 3.0
    if result == 'L':
        return -3.0
    return 0.0

def is_championship_fight_row(row):
    if 'is_title_bout' in row:
        return row.get('is_title_bout') == 2
    return row.get('is_championship', False)

def championship_bonus(points, result, is_championship):
    if is_championship and result in ('W', 'L'):
        return points * 2.0
    return points

def multi_division_championship_bonus(points, result, is_title_bout, fighter_url, weight_class, fighter_title_weight_classes):
    """
    Rewards a fighter for capturing championship titles across multiple different weight classes.
    When a fighter wins a title bout in a new weight class they haven't held a title in before,
    applies a multiplicative milestone bonus (1.25x) to reflect historic multi-division greatness.
    """
    if is_title_bout == 2 and result == 'W':
        if weight_class not in fighter_title_weight_classes[fighter_url]:
            fighter_title_weight_classes[fighter_url].add(weight_class)
            # If this is their 2nd (or subsequent) distinct weight class championship win:
            if len(fighter_title_weight_classes[fighter_url]) > 1:
                return round(points * 1.25, 2)
    return points


def streak_adjustment(points, result, opponent_streak):
    if result == 'W':
        if opponent_streak > 2:
            multiplier = 1.0 + ((opponent_streak - 2) * 0.10)
            return round(points * multiplier, 2)
        elif opponent_streak < -2:
            multiplier = max(0.5, 1.0 - ((abs(opponent_streak) - 2) * 0.10))
            return round(points * multiplier, 2)
    elif result == 'L':
        if opponent_streak < -2:
            multiplier = 1.0 + ((abs(opponent_streak) - 2) * 0.15)
            return round(points * multiplier, 2)
    return points

# ---------------------------------------------------------------------------
# Empirical age calibration
# ---------------------------------------------------------------------------
# Age parameters are deliberately NOT hardcoded.  The calibration is derived
# from the fight dataset passed to calculate_ude_points_with_ablation().
#
# The calibration models current-fight win probability as a function of:
#   * fighter's own age
#   * opponent's age
#   * opponent quality
#   * weight class
#
# A piecewise-linear age specification is selected empirically by BIC.  The
# selected breakpoint and post-breakpoint slopes are then passed into the two
# scoring functions below.  This keeps statistical calibration separate from
# the UDE scoring mechanism while ensuring the scoring mechanism never embeds
# a manually chosen age threshold/slope.

AGE_MIN_MULTIPLIER = 0.50
AGE_MAX_MULTIPLIER = 1.50
AGE_CALIBRATION_MIN_OBSERVATIONS = 1000
AGE_CALIBRATION_RIDGE = 1e-6


def _age_calibration_observations(df):
    """Build two fighter-side observations from each decisive fight."""
    records = []
    side_specs = [
        (1, 2, 'fight_result_fighter_1',
         'fight_day_age (yrs)_fighter_1', 'fight_day_age (yrs)_fighter_2',
         'is_champion_fighter_2', 'title_defenses_fighter_2',
         'pre_fight_record_fighter_2_(W-L-D NC)'),
        (2, 1, 'fight_result_fighter_2',
         'fight_day_age (yrs)_fighter_2', 'fight_day_age (yrs)_fighter_1',
         'is_champion_fighter_1', 'title_defenses_fighter_1',
         'pre_fight_record_fighter_1_(W-L-D NC)'),
    ]

    for _, row in df.iterrows():
        if is_no_score_fight(row.get('fight_result_fighter_1'), row.get('method', row.get('method_mapped'))):
            continue
        if is_no_score_fight(row.get('fight_result_fighter_2'), row.get('method', row.get('method_mapped'))):
            continue

        weight_class = row.get('weight_class_cleaned')
        if pd.isna(weight_class):
            continue

        for _, _, result_col, own_age_col, opp_age_col, champ_col, defenses_col, record_col in side_specs:
            own_age = row.get(own_age_col)
            opponent_age = row.get(opp_age_col)
            opponent_record = row.get(record_col)
            opponent_champion = row.get(champ_col)
            opponent_defenses = row.get(defenses_col)

            if pd.isna(own_age) or pd.isna(opponent_age):
                continue
            if pd.isna(opponent_record) or pd.isna(opponent_champion) or pd.isna(opponent_defenses):
                continue

            result = row.get(result_col)
            if result not in {'W', 'L'}:
                continue

            records.append({
                'own_age': float(own_age),
                'opponent_age': float(opponent_age),
                'opponent_quality': quality_score(
                    opponent_record, opponent_champion, opponent_defenses
                ),
                'weight_class': str(weight_class),
                'win': 1.0 if result == 'W' else 0.0,
            })

    return pd.DataFrame.from_records(records)


def _fit_logistic_irls(X, y, ridge=AGE_CALIBRATION_RIDGE,
                       max_iter=100, tolerance=1e-8):
    """Small dependency-free ridge logistic regression via IRLS/Newton steps."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    beta = np.zeros(X.shape[1], dtype=float)

    penalty = np.eye(X.shape[1], dtype=float) * ridge
    penalty[0, 0] = 0.0  # do not penalize intercept

    for _ in range(max_iter):
        eta = np.clip(X @ beta, -30.0, 30.0)
        p = 1.0 / (1.0 + np.exp(-eta))
        w = np.maximum(p * (1.0 - p), 1e-8)
        z = eta + (y - p) / w

        lhs = X.T @ (w[:, None] * X) + penalty
        rhs = X.T @ (w * z)
        try:
            new_beta = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            new_beta = np.linalg.pinv(lhs) @ rhs

        if np.max(np.abs(new_beta - beta)) < tolerance:
            beta = new_beta
            break
        beta = new_beta

    eta = np.clip(X @ beta, -30.0, 30.0)
    p = 1.0 / (1.0 + np.exp(-eta))
    eps = 1e-12
    log_likelihood = np.sum(
        y * np.log(np.clip(p, eps, 1.0 - eps))
        + (1.0 - y) * np.log(np.clip(1.0 - p, eps, 1.0 - eps))
    )
    return beta, log_likelihood


def _age_design_matrix(observations, breakpoint):
    """Create the piecewise-linear age/OQ/weight-class design matrix."""
    own_linear = observations['own_age'].to_numpy() - breakpoint
    own_post = np.maximum(own_linear, 0.0)
    opponent_linear = observations['opponent_age'].to_numpy() - breakpoint
    opponent_post = np.maximum(opponent_linear, 0.0)
    quality = observations['opponent_quality'].to_numpy()

    weights = pd.get_dummies(
        observations['weight_class'].astype(str),
        drop_first=True,
        dtype=float,
    )

    numeric = np.column_stack([
        np.ones(len(observations)),
        own_linear,
        own_post,
        opponent_linear,
        opponent_post,
        quality - observations['opponent_quality'].mean(),
    ])

    if len(weights.columns):
        return np.column_stack([numeric, weights.to_numpy(dtype=float)])
    return numeric


def _fit_age_model(observations, breakpoint):
    X = _age_design_matrix(observations, breakpoint)
    y = observations['win'].to_numpy(dtype=float)
    beta, log_likelihood = _fit_logistic_irls(X, y)
    n = len(y)
    n_parameters = len(beta)
    bic = -2.0 * log_likelihood + n_parameters * np.log(n)
    return beta, bic


def calibrate_age_effects(df):
    """
    Empirically derive the age parameters used by UDE's age adjustments.

    The breakpoint is selected by BIC over a data-derived candidate range
    (the middle 60% of observed fighter ages, evaluated in 0.5-year steps).
    The two slopes before the breakpoint and the two incremental post-break
    slopes are estimated jointly with opponent quality and weight class.

    Returns a plain dict so it can be passed explicitly into the scoring
    functions and easily logged/tested/serialized.
    """
    observations = _age_calibration_observations(df)
    if len(observations) < AGE_CALIBRATION_MIN_OBSERVATIONS:
        raise ValueError(
            f'Insufficient observations for empirical age calibration: '
            f'{len(observations)} < {AGE_CALIBRATION_MIN_OBSERVATIONS}'
        )

    ages = pd.concat([
        observations['own_age'],
        observations['opponent_age'],
    ], ignore_index=True)
    lower = float(ages.quantile(0.20))
    upper = float(ages.quantile(0.80))
    candidates = np.arange(
        np.floor(lower * 2.0) / 2.0,
        np.ceil(upper * 2.0) / 2.0 + 0.001,
        0.5,
    )

    fitted = []
    for breakpoint in candidates:
        beta, bic = _fit_age_model(observations, float(breakpoint))
        fitted.append((bic, float(breakpoint), beta))

    _, breakpoint, beta = min(fitted, key=lambda item: item[0])

    # Matrix order is fixed by _age_design_matrix:
    # intercept, own_linear, own_post, opponent_linear, opponent_post, quality, ...
    own_pre_slope = float(beta[1])
    own_post_increment = float(beta[2])
    opponent_pre_slope = float(beta[3])
    opponent_post_increment = float(beta[4])

    return {
        'reference_age': breakpoint,
        'own_pre_slope': own_pre_slope,
        'own_post_slope': own_pre_slope + own_post_increment,
        'opponent_pre_slope': opponent_pre_slope,
        'opponent_post_slope': opponent_pre_slope + opponent_post_increment,
        'own_post_increment': own_post_increment,
        'opponent_post_increment': opponent_post_increment,
        'n_observations': int(len(observations)),
        'calibration_method': 'piecewise_logistic_bic',
    }


def _age_multiplier(age, calibration, side, result):
    """Convert calibrated age effect into the existing UDE multiplier semantics."""
    if result not in {'W', 'L'} or pd.isna(age):
        return 1.0

    reference_age = calibration['reference_age']
    # calibration['reference_age'] is NaN under neutral calibration (see
    # _neutral_age_calibration) -- comparisons against NaN are always False,
    # so `age <= reference_age` would silently fail to short-circuit and let
    # slope * NaN propagate into the multiplier. Must return early explicitly.
    if pd.isna(reference_age):
        return 1.0
    if age <= reference_age:
        return 1.0

    slope = calibration[f'{side}_post_slope']
    years_after_reference = float(age) - reference_age

    if side == 'opponent':
        # Opposition-quality discount: signed, so a declining opponent is
        # worth less to beat and a worse loss to fall to -- whichever
        # direction the calibration actually finds for opponent_post_slope.
        effect = np.exp(slope * years_after_reference)
        multiplier = 1.0 / effect if result == 'W' else effect
        return float(max(AGE_MIN_MULTIPLIER, min(AGE_MAX_MULTIPLIER, multiplier)))

    # Own side: an achievement-under-adversity reward, not an
    # opposition-quality discount, so magnitude (not sign) scales the bonus --
    # every other UDE component (higher_rated_opponent_bonus,
    # opponent_quality_adjustment) already rewards clearing a harder bar with
    # more credit, never less. The signed own_post_slope is negative
    # (own-age win probability declines past the breakpoint); using it
    # directly would shrink this bonus instead of rewarding it.
    effect = np.exp(abs(slope) * years_after_reference)
    multiplier = effect if result == 'W' else 1.0
    return float(max(AGE_MIN_MULTIPLIER, min(AGE_MAX_MULTIPLIER, multiplier)))


def age_adjustment(points, result, opponent_age, weight_class, calibration):
    multiplier = _age_multiplier(opponent_age, calibration, 'opponent', result)
    return round(points * multiplier, 2)


OWN_AGE_GATE_WIDTH_YEARS = 3.0

def _own_age_gate_scale(opponent_age, reference_age, width=OWN_AGE_GATE_WIDTH_YEARS):
    """
    Smooth replacement for a hard opponent_age > reference_age cutoff: 1.0
    well below reference_age (full credit), 0.5 exactly at reference_age,
    ->0.0 well above it (fully gated). Avoids a discontinuous jump in the
    fighter's own-age bonus from a marginal, sub-year change in opponent age.
    """
    if pd.isna(opponent_age) or pd.isna(reference_age):
        return 0.0
    return float(1.0 / (1.0 + np.exp((float(opponent_age) - float(reference_age)) / width)))

def own_age_adjustment(points, result, fighter_age, opponent_age, weight_class, calibration):
    # Gated by opponent age so a fighter isn't rewarded for beating another
    # fighter who is also past their prime -- gate_scale (see
    # _own_age_gate_scale) scales continuously rather than gating on/off.
    if result == 'W':
        reference_age = calibration['reference_age']
        gate_scale = _own_age_gate_scale(opponent_age, reference_age)
        raw_multiplier = _age_multiplier(fighter_age, calibration, 'own', result)
        multiplier = 1.0 + (raw_multiplier - 1.0) * gate_scale
        return round(points * multiplier, 2)

    multiplier = _age_multiplier(fighter_age, calibration, 'own', result)
    return round(points * multiplier, 2)
    
def _get_previous_fights(fighter, opponent, df, current_fight_date):
    """
    Shared lookup for all previous meetings between `fighter` and `opponent`.
    Filters on the fighter_url_fighter_1/2 columns because `fighter` and 
    `opponent` variables pass URL data.
    """
    return df[(df['fighter_url_fighter_1'].isin([fighter, opponent])) &
              (df['fighter_url_fighter_2'].isin([fighter, opponent])) &
              (df['event_date'] < current_fight_date)]

def _revenge_bonus(points, result, fighter, opponent, previous_fights):
    if not previous_fights.empty:
        first_fight = previous_fights.iloc[0]
        if first_fight['fighter_url_fighter_1'] == fighter:
            first_fight_result_fighter = first_fight['fight_result_fighter_1']
        else:
            first_fight_result_fighter = first_fight['fight_result_fighter_2']
        if first_fight_result_fighter == 'L' and result == 'W':
            return round(points * 1.15, 2)
    return points

def _rematch_count_bonus(points, result, previous_fights):
    num_rematches = len(previous_fights)
    if num_rematches == 0:
        return points
    if result == 'L':
        penalty_multiplier = 1.0 + (min(num_rematches, 5) * 0.10)
        return round(points * penalty_multiplier, 2)
    return points

def rematch_adjustment(points, result, fighter, opponent, df, current_fight_date, pair_history_cache=None):
    """Apply revenge/rematch effects from strictly prior meetings.

    With the incremental pair-history state supplied by the scoring loop, this
    is equivalent to the historical dataframe-filter implementation without
    repeatedly scanning the full fight dataframe.
    """
    if pair_history_cache is None:
        previous_fights = _get_previous_fights(fighter, opponent, df, current_fight_date)
        points = _revenge_bonus(points, result, fighter, opponent, previous_fights)
        return _rematch_count_bonus(points, result, previous_fights)

    key = tuple(sorted((fighter, opponent)))
    state = pair_history_cache.get(key)
    if state is None or state['count'] == 0:
        return points

    if result == 'W' and state['first_result_by_fighter'].get(fighter) == 'L':
        points = round(points * 1.15, 2)
    if result == 'L':
        points = round(points * (1.0 + min(state['count'], 5) * 0.10), 2)
    return points

def title_defense_bonus(raw_base_points, fighter_defenses, is_champion, is_title_bout, result):
    """
    Calculates title defense scaling off raw base-result points. 
    This decouples the bonus from championship (2.0x) and dominance (up to 1.5x) multipliers 
    to prevent excessive compounding on every title fight. Uses a saturating curve.
    """
    if is_champion > 0 and is_title_bout == 2 and result == 'W':
        defense_multiplier = 1.0 + TITLE_DEFENSE_CAP * (1 - TITLE_DEFENSE_DECAY ** max(0, fighter_defenses))
        return round(raw_base_points * defense_multiplier, 2)
    return raw_base_points

# Max possible |pdi_margin|: 5 phases, each capped at magnitude 1.0
PDI_MARGIN_SCALE = 5.0  

def _interpolate_dominance_multiplier(t, high_anchor, low_anchor):
    """
    t in [-1, 1]: fighter's own normalized pdi_margin (+1 = fully dominant,
    -1 = fully outclassed). high_anchor = multiplier at t=+1, low_anchor =
    multiplier at t=-1. Piecewise-linear through (t=0 -> 1.0), reducing
    to plain base points when fighters are statistically even.
    """
    t = max(-1.0, min(1.0, t))
    if t >= 0:
        return 1.0 + t * (high_anchor - 1.0)
    else:
        return 1.0 + t * (1.0 - low_anchor)

def dominance_adjustment(points, fighter_name, opponent_name, result, dominant_fighter, pdi_margin=None):
    """
    Method-neutral continuous PDI adjustment. Method is handled separately by
    the empirically calibrated method_x_pdi residual (see method_pdi_residual_points).
    """
    if result not in ('W', 'L'):
        return points

    if result == 'W':
        high_anchor, low_anchor = 1.30, 0.80
    else:
        high_anchor, low_anchor = 0.75, 1.20

    if pdi_margin is None or (isinstance(pdi_margin, float) and pd.isna(pdi_margin)):
        if dominant_fighter == fighter_name:
            t = 1.0
        elif dominant_fighter == opponent_name:
            t = -1.0
        else:
            return points
    else:
        t = pdi_margin / PDI_MARGIN_SCALE

    multiplier = _interpolate_dominance_multiplier(t, high_anchor, low_anchor)
    return round(points * multiplier, 2)

HIGHER_RATED_GAP_FLOOR = 15.0     # rating gaps at/under this earn zero bonus/penalty
HIGHER_RATED_GAP_SCALE = 30.0     # controls how quickly tanh approaches its asymptote
HIGHER_RATED_MAX_MAGNITUDE = 9.0  # asymptotic cap on the bonus/penalty magnitude

def higher_rated_opponent_bonus(points, result, diff, perf_scale=1.0):
    """
    Additive upset bonus/penalty from a rating gap, continuous and bounded:
    zero at/under HIGHER_RATED_GAP_FLOOR, saturating toward
    +/-HIGHER_RATED_MAX_MAGNITUDE via tanh as the gap widens.

    The tanh magnitude is scaled by `perf_scale` before being added, so the
    bonus/penalty is damped when the underlying in-fight performance
    (PDI-derived) wasn't itself decisive, full strength when it was. Because
    perf_scale is applied here internally, the caller must add this bonus
    directly and must NOT also multiply it by perf_scale again (that would
    double-apply the performance coupling).

    `diff` must be a SMOOTHED rating gap (trailing UPSET_RATING_WINDOW-fight
    average for both fighters -- see _trailing_rating_for_upset), not the raw
    instantaneous pre-fight UDE difference. UDE's ledger is self-referential
    and can jump 10+ points off a single fight (championship 2x, dominance,
    this bonus itself all stacking), so using the literal snapshot as an
    underdog-ness proxy would make the very next fight's gap mechanically
    shrink or inflate from scoring-order artifacts rather than a fighter's
    actual standing. The trailing average keeps the underdog signal while
    damping that single-fight whiplash.
    """
    if result == 'W' and diff > 0:
        raw_bonus = HIGHER_RATED_MAX_MAGNITUDE * np.tanh(
            max(0.0, (diff - HIGHER_RATED_GAP_FLOOR) / HIGHER_RATED_GAP_SCALE)
        )
        return points + raw_bonus * perf_scale
    elif result == 'L' and diff < 0:
        raw_penalty = -HIGHER_RATED_MAX_MAGNITUDE * np.tanh(
            max(0.0, (abs(diff) - HIGHER_RATED_GAP_FLOOR) / HIGHER_RATED_GAP_SCALE)
        )
        return points + raw_penalty * perf_scale
    return points
# Opponent-quality adjustment
# Deliberately evaluates the PRE-FIGHT signal: the opponent's record, championship
# status, and title-defense count are taken from the state immediately before the bout.
#
# Shrinkage prior pulls small-sample records toward a .500 win rate so that an
# opponent arriving at 1-0 is not mathematically treated as a 100% winner.
OQ_PRIOR_STRENGTH = 5.0
OQ_PRIOR_WIN_RATE = 0.50
OQ_QUALITY_CENTER = 0.56
OQ_CHAMPION_BONUS = 0.05
OQ_DEFENSE_BONUS = 0.01
OQ_MAX_DEFENSES_FOR_QUALITY = 5
OQ_DEFAULT_K = 2.5
OQ_MIN_MULTIPLIER = 0.50
OQ_MAX_MULTIPLIER = 1.50

def _parse_pre_fight_record(record):
    """Return (wins, losses) from the pipeline's W-L-D NC record string."""
    try:
        record = str(record)
        wl = record.split()[0]
        wins, losses, *_ = wl.split('-')
        return int(wins), int(losses)
    except (ValueError, IndexError, AttributeError):
        return 0, 0

def shrunk_win_rate(pre_fight_record, prior_strength=OQ_PRIOR_STRENGTH,
                    prior_win_rate=OQ_PRIOR_WIN_RATE):
    """
    Empirical-Bayes-style shrinkage of an opponent's pre-fight win rate toward
    a .500 prior. Only fights completed before the current bout are included.
    """
    wins, losses = _parse_pre_fight_record(pre_fight_record)
    fights = wins + losses
    return (wins + prior_strength * prior_win_rate) / (fights + prior_strength)

def quality_score(opponent_pre_fight_record, opponent_is_champion,
                           opponent_title_defenses):
    """Return the pre-fight quality score of the opponent before k scaling."""
    win_rate = shrunk_win_rate(opponent_pre_fight_record)
    champion_signal = (OQ_CHAMPION_BONUS if opponent_is_champion == 2 else 0.0)
    try:
        defenses = min(float(opponent_title_defenses), OQ_MAX_DEFENSES_FOR_QUALITY)
    except (TypeError, ValueError):
        defenses = 0.0
    defense_signal = max(0.0, defenses) * OQ_DEFENSE_BONUS
    return win_rate + champion_signal + defense_signal

def opponent_quality_multiplier(opponent_pre_fight_record, opponent_is_champion,
                                opponent_title_defenses, k=OQ_DEFAULT_K):
    """Convert pre-fight opponent quality into the clipped multiplier."""
    quality_score_var = quality_score(
        opponent_pre_fight_record, opponent_is_champion, opponent_title_defenses
    )
    multiplier = 1.0 + k * (quality_score_var - OQ_QUALITY_CENTER)
    return max(OQ_MIN_MULTIPLIER, min(OQ_MAX_MULTIPLIER, multiplier))

def opponent_quality_adjustment(points, result, opponent_pre_fight_record,
                                opponent_is_champion, opponent_title_defenses,
                                k=OQ_DEFAULT_K):
    """Adjust fight points for pre-fight opponent quality.

    Wins retain the original quality multiplier. For losses, elite/high-quality
    opposition provides no cushion (1.0x). Losses to below-center opponents
    are amplified according to ``1 + k * (OQ_QUALITY_CENTER - OQ)`` and bounded at
    1.50x. This makes loss sensitivity asymmetric: elite losses remain fully
    informative, while bad losses become more costly.
    """
    if result not in ('W', 'L'):
        return points

    win_multiplier = opponent_quality_multiplier(
        opponent_pre_fight_record, opponent_is_champion, opponent_title_defenses, k=k
    )

    if result == 'W':
        multiplier = win_multiplier
    else:
        # Elite/high-quality opponents do not cushion a loss.
        # Weak opponents amplify the loss in proportion to how far their
        # OQ falls below the empirical quality center.
        oq = quality_score(
            opponent_pre_fight_record, opponent_is_champion, opponent_title_defenses
        )
        if oq >= OQ_QUALITY_CENTER:
            multiplier = 1.0
        else:
            multiplier = 1.0 + k * (OQ_QUALITY_CENTER - oq)
        multiplier = max(1.0, min(OQ_MAX_MULTIPLIER, multiplier))

    multiplier = max(OQ_MIN_MULTIPLIER, min(OQ_MAX_MULTIPLIER, multiplier))
    return round(points * multiplier, 2)

# ---------------------------------------------------------------------------
# Empirical method x PDI residual calibration
# ---------------------------------------------------------------------------
METHOD_RESIDUAL_MAX_POINTS = 0.75  # structural UDE cap, not a fitted parameter
METHOD_RESIDUAL_REFERENCE = 'UD'

def _build_future_performance_observations(df, min_future_fights=5, cutoff_date=None, cutoff_year=None, strict_cutoff=True):
    """Create one fighter-fight observation with a subsequent 5-fight record."""
    d = df.sort_values('event_date').copy()
    d['event_date'] = pd.to_datetime(d['event_date'])
    rows = []
    # chronological fight history per fighter URL
    histories = defaultdict(list)
    for idx, row in d.iterrows():
        for side in (1, 2):
            f = row[f'fighter_url_fighter_{side}']
            r = row[f'fight_result_fighter_{side}']
            histories[f].append((idx, row['event_date'], r))
    # Training observations must be strictly before the calibration cutoff.
    # cutoff_date is preferred; cutoff_year is retained only for backwards compatibility.
    if cutoff_date is not None:
        cutoff_date = pd.Timestamp(cutoff_date)
    elif cutoff_year is not None:
        cutoff_date = pd.Timestamp(f'{int(cutoff_year) + 1}-01-01')
    else:
        cutoff_date = d['event_date'].max() + pd.Timedelta(days=1)

    # score each fight from the perspective of each fighter
    for idx, row in d.iterrows():
        if row['event_date'] >= cutoff_date:
            continue
        method = row.get('method_mapped')
        if method not in {'Finish','UD','MD','SD'}:
            continue
        for side in (1,2):
            f = row[f'fighter_url_fighter_{side}']
            own_result = row[f'fight_result_fighter_{side}']
            if own_result not in {'W','L'}:
                continue
            opp = 2 if side == 1 else 1
            pdi = row.get(f'pdi_margin_fighter_{side}', np.nan)
            oq = row.get(f'quality_score_fighter_{opp}', np.nan)
            wc = row.get('weight_class_cleaned', row.get('weight_class'))
            future = []
            # use scored future bouts only, matching the five-fight performance target
            for j, dt, res in histories[f]:
                if dt <= row['event_date'] or j == idx:
                    continue
                # Strict temporal calibration: training observations may only use
                # future fights that themselves occurred on/before the calibration cutoff.
                if strict_cutoff and dt >= cutoff_date:
                    break
                if res in {'W','L'}:
                    future.append(res)
                if len(future) >= min_future_fights:
                    break
            if len(future) < min_future_fights:
                continue
            rows.append({
                'event_date': row['event_date'],
                'fighter_url': f,
                'method': method,
                'pdi_margin': float(pdi) if pd.notna(pdi) else np.nan,
                'opponent_quality': float(oq) if pd.notna(oq) else np.nan,
                'weight_class': wc,
                'future_wins': sum(x == 'W' for x in future[:min_future_fights]),
                'future_fights': min_future_fights,
            })
    # Explicit schema, even when `rows` is empty (e.g. the calibration cutoff
    # falls at/near the start of fight history, so there simply is no prior
    # data yet). Without this, pd.DataFrame([]) has zero columns and the
    # dropna(subset=...) below raises KeyError instead of yielding a
    # correctly-shaped empty frame -- which then defeats the ValueError
    # handling that calibrate_method_pdi_effects()/_build_temporal_calibration_cache()
    # already rely on for gracefully falling back to neutral calibration.
    columns = ['event_date', 'fighter_url', 'method', 'pdi_margin',
               'opponent_quality', 'weight_class', 'future_wins', 'future_fights']
    out = pd.DataFrame(rows, columns=columns)
    return out.dropna(subset=['pdi_margin','opponent_quality','weight_class'])

def _fit_method_pdi_glm(obs):
    """Fit the residual future-performance model; UD is the reference method."""
    x = obs.copy()
    x['pdi2'] = x['pdi_margin'] ** 2
    x['oq_centered'] = x['opponent_quality'] - x['opponent_quality'].mean()
    x['oq2'] = x['oq_centered'] ** 2
    # method dummies and PDI interactions; use fixed effect weight class.
    method_d = pd.get_dummies(x['method'], prefix='method', drop_first=False, dtype=float)
    # ensure all expected methods exist
    for m in ['Finish','UD','MD','SD']:
        col=f'method_{m}'
        if col not in method_d: method_d[col]=0.0
    method_d=method_d[['method_Finish','method_MD','method_SD','method_UD']]
    X = pd.concat([
        pd.DataFrame({
            'const':1.0, 'pdi':x['pdi_margin'], 'pdi2':x['pdi2'],
            'oq':x['oq_centered'], 'oq2':x['oq2']
        }, index=x.index),
        method_d.drop(columns=['method_UD']),
        pd.DataFrame({
            'Finish_x_pdi': method_d['method_Finish']*x['pdi_margin'],
            'MD_x_pdi': method_d['method_MD']*x['pdi_margin'],
            'SD_x_pdi': method_d['method_SD']*x['pdi_margin'],
        }, index=x.index),
        pd.get_dummies(x['weight_class'], prefix='wc', drop_first=True, dtype=float)
    ], axis=1)
    X = X.astype(float)
    model = sm.GLM(x['future_wins']/x['future_fights'], X,
                   family=sm.families.Binomial(),
                   freq_weights=x['future_fights']).fit()
    return model, X.columns.tolist(), x

def calibrate_method_pdi_effects(df, cutoff_date=None, cutoff_year=None, strict_temporal=True):
    """
    Empirically calibrate the residual value of method conditional on PDI,
    opponent quality and weight class. The fitted model predicts future
    five-fight win probability. Calibration is strictly temporal when
    strict_temporal=True: each training observation can only use its next
    five scored fights if all five occurred on/before cutoff_year. UD is the reference method.

    The resulting residual effect is mapped to UDE points by preserving its
    empirical shape and normalising it to a structural +/- METHOD_RESIDUAL_MAX_POINTS cap.
    """
    obs = _build_future_performance_observations(df, cutoff_date=cutoff_date, cutoff_year=cutoff_year, strict_cutoff=strict_temporal)
    if len(obs) < 500:
        raise ValueError(f'Insufficient observations for method/PDI calibration: {len(obs)}')
    model, cols, x = _fit_method_pdi_glm(obs)
    # Evaluate method-vs-UD effect at each observed PDI/OQ/WC.
    pred = []
    base_cols = cols
    for i, r in x.iterrows():
        common = r.copy()
        # make_row (unused below) would rebuild a full design-matrix row per
        # method and re-predict; the contrast is computed directly from the
        # fitted coefficients instead, which is equivalent and cheaper.
        def make_row(method):
            rr = common.copy()
            rr['method']=method
            rr['method_Finish']=float(method=='Finish'); rr['method_MD']=float(method=='MD'); rr['method_SD']=float(method=='SD')
            rr['Finish_x_pdi']=rr['method_Finish']*rr['pdi_margin']
            rr['MD_x_pdi']=rr['method_MD']*rr['pdi_margin']
            rr['SD_x_pdi']=rr['method_SD']*rr['pdi_margin']
            rr['method_UD']=float(method=='UD')
            return rr
        b=model.params
        # linear predictor contrast method - UD
        contrasts={
            'Finish': b.get('method_Finish',0)+b.get('Finish_x_pdi',0)*r['pdi_margin'],
            'MD': b.get('method_MD',0)+b.get('MD_x_pdi',0)*r['pdi_margin'],
            'SD': b.get('method_SD',0)+b.get('SD_x_pdi',0)*r['pdi_margin'],
            'UD': 0.0
        }
        for m,c in contrasts.items(): pred.append((i,m,c))
    contrast_df=pd.DataFrame(pred, columns=['obs_idx','method','logit_contrast'])
    # Empirical 95th percentile of absolute residual contrast defines the mapping denominator.
    scale=float(np.quantile(np.abs(contrast_df.loc[contrast_df.method!='UD','logit_contrast']),0.95))
    scale=max(scale,1e-9)
    # coefficient table and empirical crossover
    finish_b=float(model.params.get('method_Finish',0)); finish_i=float(model.params.get('Finish_x_pdi',0))
    crossover=float(-finish_b/finish_i) if abs(finish_i)>1e-12 else np.nan
    calibration={
        'model_type':'binomial_glm_future_5_fight_residual_temporally_frozen',
        'reference_method':'UD',
        'cutoff_date':str(pd.Timestamp(cutoff_date).date()) if cutoff_date is not None else (f'{int(cutoff_year)+1}-01-01' if cutoff_year is not None else None),
        'strict_temporal':bool(strict_temporal),
        'observations':int(len(obs)),
        'finish_logit_intercept':finish_b,
        'finish_pdi_slope':finish_i,
        'md_logit_intercept':float(model.params.get('method_MD',0)),
        'md_pdi_slope':float(model.params.get('MD_x_pdi',0)),
        'sd_logit_intercept':float(model.params.get('method_SD',0)),
        'sd_pdi_slope':float(model.params.get('SD_x_pdi',0)),
        'empirical_abs_logit_scale_95pct':scale,
        'max_method_residual_points':METHOD_RESIDUAL_MAX_POINTS,
        'finish_ud_crossover_pdi':crossover,
        'training_date_max':str(obs['event_date'].max().date()),
    }
    # also store the model for runtime prediction through compact coefficients
    calibration['coefficients']={k:float(v) for k,v in model.params.items()}
    calibration['weight_class_columns']=[c for c in cols if c.startswith('wc_')]
    return calibration

def method_pdi_residual_points(method_mapped, pdi_margin, calibration, result):
    if result not in {'W','L'} or method_mapped not in {'Finish','UD','MD','SD'}:
        return 0.0
    if method_mapped == 'UD':
        delta=0.0
    else:
        c=calibration['coefficients']
        intercept=c.get(f'method_{method_mapped}',0.0)
        slope=c.get(f'{method_mapped}_x_pdi',0.0)
        delta=intercept+slope*(0.0 if pd.isna(pdi_margin) else float(pdi_margin))
    scale=calibration['empirical_abs_logit_scale_95pct']
    mapped=METHOD_RESIDUAL_MAX_POINTS * np.tanh(delta/scale)
    return round(float(mapped if result=='W' else -mapped), 4)

def get_performance_scaling_factor(result, method_mapped, pdi_margin, dominant_fighter=None, fighter_name=None, opponent_name=None):
    """
    Method-neutral scaling of contextual bonuses from PDI.

    Wins: t is the fighter's own normalized pdi_margin. A dominant win
    (t near +1) scales contextual bonuses toward 1.0; a win despite being
    outperformed on the stats (t near -1) scales them down toward low_anchor.

    Losses: mirrored on how decisive the loss was, using the same curve
    shape and the same [0.20, 1.0] clip range as wins (no separate
    method-severity axis, so a bad loss's penalties are bounded by the same
    ceiling a great win's bonuses are). A thoroughly outclassed loss (t near
    -1) scales contextual penalties (bad-loss, cold-streak-loss) toward full
    strength; a loss despite winning the underlying stats (t positive -- a
    close/questionable decision loss) scales penalties down, since the
    performance itself wasn't actually bad.
    """
    if result not in ('W', 'L'):
        return 1.0
    if pdi_margin is None or (isinstance(pdi_margin, float) and pd.isna(pdi_margin)):
        if dominant_fighter == opponent_name:
            t = -1.0
        elif dominant_fighter == fighter_name:
            t = 1.0
        else:
            t = 0.0
    else:
        t = max(-1.0, min(1.0, float(pdi_margin) / PDI_MARGIN_SCALE))

    mid_pivot = 0.75
    low_anchor = 0.30
    high_anchor = 1.0
    s = t if result == 'W' else -t
    scale = (mid_pivot + s * (high_anchor - mid_pivot) if s >= 0
             else mid_pivot + s * (mid_pivot - low_anchor))
    return round(max(0.20, min(1.0, scale)), 4)

def _neutral_age_calibration(reason='insufficient_prior_history'):
    return {
        'reference_age': np.nan,
        'own_pre_slope': 0.0, 'own_post_slope': 0.0,
        'opponent_pre_slope': 0.0, 'opponent_post_slope': 0.0,
        'own_post_increment': 0.0, 'opponent_post_increment': 0.0,
        'n_observations': 0,
        'calibration_method': 'temporal_expanding_neutral',
        'calibration_reason': reason,
    }

def _neutral_method_pdi_calibration(reason='insufficient_prior_history'):
    return {
        'model_type': 'temporal_expanding_neutral',
        'reference_method': 'UD',
        'cutoff_date': None,
        'strict_temporal': True,
        'observations': 0,
        'empirical_abs_logit_scale_95pct': 1.0,
        'max_method_residual_points': METHOD_RESIDUAL_MAX_POINTS,
        'finish_ud_crossover_pdi': np.nan,
        'training_date_max': None,
        'coefficients': {},
        'weight_class_columns': [],
        'calibration_reason': reason,
    }

CALIBRATION_ROLLING_WINDOW_YEARS = 5
CALIBRATION_MIN_FIGHT_OBSERVATIONS = 1000


def _build_temporal_calibration_cache(df):
    """Build calibrations using a trailing rolling window of fight history.

    Each calendar year's calibration uses only fights strictly before that
    year's cutoff, drawn from a trailing CALIBRATION_ROLLING_WINDOW_YEARS-year
    window rather than the full expanding history. This keeps age-decline and
    method x PDI curves reflective of the contemporary era instead of being
    dragged toward legacy-era rule/scoring noise as history accumulates.

    If the rolling window does not contain at least
    CALIBRATION_MIN_FIGHT_OBSERVATIONS fights (early years, before enough
    rolling history exists), the window falls back to the full expanding
    history so the fit still has adequate statistical power.
    """
    d = df.sort_values('event_date').copy()
    d['event_date'] = pd.to_datetime(d['event_date'])
    years = sorted(d['event_date'].dt.year.unique())
    cache = {}
    for year in years:
        cutoff = pd.Timestamp(f'{int(year)}-01-01')
        window_start = cutoff - pd.DateOffset(years=CALIBRATION_ROLLING_WINDOW_YEARS)
        prior = d[(d['event_date'] < cutoff) & (d['event_date'] >= window_start)]

        if len(prior) < CALIBRATION_MIN_FIGHT_OBSERVATIONS:
            prior = d[d['event_date'] < cutoff]

        try:
            age_cal = calibrate_age_effects(prior)
        except ValueError:
            age_cal = _neutral_age_calibration()

        try:
            method_cal = calibrate_method_pdi_effects(prior, cutoff_date=cutoff, strict_temporal=True)
        except ValueError:
            method_cal = _neutral_method_pdi_calibration()

        cache[year] = (age_cal, method_cal)
    return cache


def calculate_ude_points_with_ablation(df, ablate=None, opponent_quality_k=OQ_DEFAULT_K, age_calibration=None, method_pdi_calibration=None):
    ablate = set(ablate or [])
    streak_fn = noop if 'streak' in ablate else streak_adjustment
    age_fn = noop if 'age' in ablate else age_adjustment
    own_age_fn = noop if 'own_age' in ablate else own_age_adjustment
    rematch_fn = noop if 'rematch' in ablate else rematch_adjustment
    dominance_fn = noop if 'dominance' in ablate else dominance_adjustment
    higher_rated_fn = noop if 'higher_rated' in ablate else higher_rated_opponent_bonus
    title_defense_fn = noop if 'title_defense' in ablate else title_defense_bonus
    championship_fn = noop if 'championship' in ablate else championship_bonus
    multi_division_fn = noop if 'multi_division' in ablate else multi_division_championship_bonus
    opponent_quality_fn = noop if 'opponent_quality' in ablate else opponent_quality_adjustment

    df = df.sort_values(by='event_date').copy()
    df['event_date'] = pd.to_datetime(df['event_date'])

    # UDE is a historical reconstruction: a fight at time T can only use
    # calibration parameters learned from fights strictly before T (see
    # _build_temporal_calibration_cache for the rolling-window mechanics).
    # User-supplied calibrations are supported as explicitly frozen
    # production objects instead, for callers scoring outside this dataset.
    temporal_calibration_cache = None
    if age_calibration is None and method_pdi_calibration is None:
        temporal_calibration_cache = _build_temporal_calibration_cache(df)
    elif age_calibration is None or method_pdi_calibration is None:
        raise ValueError(
            'Pass both age_calibration and method_pdi_calibration as explicitly '
            'frozen objects, or pass neither so UDE builds temporally expanding calibrations.'
        )

    df.attrs['age_calibration'] = age_calibration
    df.attrs['method_pdi_calibration'] = method_pdi_calibration
    fighter_ude_points = {}
    fighter_title_weight_classes = defaultdict(set)

    # Incremental chronological pair history for rematch/revenge scoring.
    pair_history_cache = {}

    # ABSOLUTE_SWING_CAP diagnostics: how often the circuit-breaker cap on
    # total per-fight point swings actually binds, out of every scored
    # fighter-fight observation.
    swing_cap_triggered_count = 0
    total_scored_fighter_fights = 0

    # Trailing post-fight UDE ratings per fighter, capped to the last
    # UPSET_RATING_WINDOW fights. Used only to smooth the rating gap fed
    # into higher_rated_opponent_bonus (see that function's docstring).
    # Left empty for a fighter's first fight, which correctly falls back to
    # their instantaneous pre-fight rating (500, the default) below.
    fighter_rating_history = defaultdict(lambda: deque(maxlen=UPSET_RATING_WINDOW))

    def _trailing_rating_for_upset(fighter_url, instantaneous_pre_fight_rating):
        history = fighter_rating_history[fighter_url]
        if not history:
            return instantaneous_pre_fight_rating
        return sum(history) / len(history)

    for index, row in df.iterrows():
        weight_class = row['weight_class_cleaned']
        fighter_1_url = row['fighter_url_fighter_1']
        fighter_2_url = row['fighter_url_fighter_2']

        pre_fight_snapshot = {
            fighter_1_url: fighter_ude_points.get(fighter_1_url, 500),
            fighter_2_url: fighter_ude_points.get(fighter_2_url, 500),
        }
        post_fight_updates = {}

        if temporal_calibration_cache is not None:
            current_year = int(row['event_date'].year)
            current_age_calibration, current_method_pdi_calibration = temporal_calibration_cache[current_year]
        else:
            current_age_calibration = age_calibration
            current_method_pdi_calibration = method_pdi_calibration

        for fighter_url_col, fighter_col, result_col, ude_col, opponent_url_col, opponent_col, streak_col, own_age_col, opponent_age_col, champ_col, title_def_col, pdi_margin_col in [
            ('fighter_url_fighter_1', 'fighter_1', 'fight_result_fighter_1', 'ude_points_post_fight_fighter_1', 'fighter_url_fighter_2', 'fighter_2', 'W/L_streak_fighter_2', 'fight_day_age (yrs)_fighter_1', 'fight_day_age (yrs)_fighter_2', 'is_champion_fighter_1', 'title_defenses_fighter_1', 'pdi_margin_fighter_1'),
            ('fighter_url_fighter_2', 'fighter_2', 'fight_result_fighter_2', 'ude_points_post_fight_fighter_2', 'fighter_url_fighter_1', 'fighter_1', 'W/L_streak_fighter_1', 'fight_day_age (yrs)_fighter_2', 'fight_day_age (yrs)_fighter_1', 'is_champion_fighter_2', 'title_defenses_fighter_2', 'pdi_margin_fighter_2')
        ]:
            fighter = row[fighter_url_col]
            fighter_name = row[fighter_col]
            result = row[result_col]
            opponent = row[opponent_url_col]
            opponent_name = row[opponent_col]
            opponent_streak = row[streak_col]
            fighter_age = row[own_age_col]
            opponent_age = row[opponent_age_col]
            fighter_pdi_margin = row.get(pdi_margin_col, None)
            current_fight_date = row['event_date']
            is_champion = row[champ_col]
            title_defenses = row[title_def_col]
            is_title_bout = row['is_title_bout']
            opponent_record_col = (
                'pre_fight_record_fighter_2_(W-L-D NC)'
                if fighter_col == 'fighter_1'
                else 'pre_fight_record_fighter_1_(W-L-D NC)'
            )
            opponent_pre_fight_record = row[opponent_record_col]
            opponent_is_champion = row['is_champion_fighter_2'] if fighter_col == 'fighter_1' else row['is_champion_fighter_1']
            opponent_title_defenses = row['title_defenses_fighter_2'] if fighter_col == 'fighter_1' else row['title_defenses_fighter_1']

            pre_fight_ude = pre_fight_snapshot[fighter]
            opponent_pre_fight_ude = pre_fight_snapshot[opponent]

            # Bypass logic for outcomes that never yield point movements
            method_raw = row.get('method', row.get('method_mapped'))
            if is_no_score_fight(result, method_raw):
                post_fight_updates[fighter] = pre_fight_ude
                df.at[index, f'ude_points_pre_fight_{fighter_col}'] = pre_fight_ude
                df.at[index, f'ude_points_post_fight_{fighter_col}'] = pre_fight_ude
                continue

            # Base fight-night points
            raw_base_points = base_points_from_result_method(result, row['method_mapped'])
            points = championship_fn(raw_base_points, result, is_championship_fight_row(row))
            points = multi_division_fn(points, result, is_title_bout, fighter, weight_class, fighter_title_weight_classes)
            dominant_fighter = row['dominant_fighter']
            points = dominance_fn(points, fighter_name, opponent_name, result, dominant_fighter, fighter_pdi_margin)

            # Empirically calibrated method x PDI residual, bounded to a deliberately
            # small achievement-scale contribution. UD is the reference method.
            method_residual_pts = method_pdi_residual_points(
                row['method_mapped'], fighter_pdi_margin, current_method_pdi_calibration, result
            )
            points += method_residual_pts

            # Performance Scaling Factor: PDI-only; method is not counted again.
            perf_scale = get_performance_scaling_factor(result, row['method_mapped'], fighter_pdi_margin, dominant_fighter, fighter_name, opponent_name)

            # Marginal contributions of contextual bonuses
            df.at[index, f'method_pdi_residual_{fighter_col}'] = method_residual_pts
            # Derived by calculating the delta between the fn's output and the input running points,
            # making it ablation-safe.
            # Note: both title_defense_fn and streak_fn are seeded with `raw_base_points`
            # (pre-championship, pre-dominance) so neither contextual bonus compounds on
            # top of the other's or of the standard multipliers already active on the fight.
            t_def_bonus = round(title_defense_fn(raw_base_points, title_defenses, is_champion, is_title_bout, result) - raw_base_points, 2)
            streak_bonus = round(streak_fn(raw_base_points, result, opponent_streak) - raw_base_points, 2)
            df.at[index, f'title_defense_bonus_{fighter_col}'] = t_def_bonus
            df.at[index, f'streak_bonus_{fighter_col}'] = streak_bonus

            # Smoothed (trailing UPSET_RATING_WINDOW-fight average) rating
            # for BOTH fighters, not the raw instantaneous pre-fight UDE
            # snapshot -- see higher_rated_opponent_bonus docstring. Applied
            # symmetrically to both sides of the gap: the opponent's rating
            # is exactly as susceptible to single-fight whiplash as the
            # fighter's own, so smoothing only one side would just relocate
            # the artifact rather than remove it.
            own_rating_for_upset = _trailing_rating_for_upset(fighter, pre_fight_ude)
            opponent_rating_for_upset = _trailing_rating_for_upset(opponent, opponent_pre_fight_ude)
            upset_diff = opponent_rating_for_upset - own_rating_for_upset
            # higher_rated_fn (new tanh version) already applies perf_scale
            # internally -- it must NOT be re-scaled by the combined_bonuses
            # * perf_scale step below, or the performance coupling would be
            # applied twice (quadratically damped instead of linearly).
            upset_bonus = round(higher_rated_fn(points, result, upset_diff, perf_scale) - points, 2)

            df.at[index, f'performance_scaling_factor_{fighter_col}'] = perf_scale
            df.at[index, f'higher_rated_opponent_bonus_{fighter_col}'] = upset_bonus
            df.at[index, f'upset_rating_gap_smoothed_{fighter_col}'] = round(upset_diff, 4)

            combined_bonuses = t_def_bonus + streak_bonus
            scaled_bonuses = round(combined_bonuses * perf_scale, 2)

            # Invariant bonuses (not passed through perf_scale): opponent-age, own-age, rematch.
            age_pts = round(age_fn(points, result, opponent_age, weight_class, current_age_calibration) - points, 2)
            own_age_pts = round(own_age_fn(points, result, fighter_age, opponent_age, weight_class, current_age_calibration) - points, 2)
            rematch_pts = round(rematch_fn(
                points, result, fighter, opponent, df, current_fight_date, pair_history_cache
            ) - points, 2)
            df.at[index, f'age_adjustment_{fighter_col}'] = age_pts
            df.at[index, f'own_age_adjustment_{fighter_col}'] = own_age_pts
            df.at[index, f'rematch_adjustment_{fighter_col}'] = rematch_pts
            points_before_opponent_quality = points
            opponent_quality_pts = round(opponent_quality_fn(
                points, result, opponent_pre_fight_record, opponent_is_champion,
                opponent_title_defenses, k=opponent_quality_k
            ) - points, 2)

            # points already includes championship/dominance/method-residual;
            # upset_bonus is added directly (not through scaled_bonuses) since
            # higher_rated_fn already applied perf_scale internally above.
            points = points + scaled_bonuses + upset_bonus + age_pts + own_age_pts + rematch_pts + opponent_quality_pts

            # Bound the final point swing, tracking how often the cap actually binds.
            total_scored_fighter_fights += 1
            swing_cap_triggered = abs(points) > ABSOLUTE_SWING_CAP
            if swing_cap_triggered:
                swing_cap_triggered_count += 1
            df.at[index, f'absolute_swing_cap_triggered_{fighter_col}'] = swing_cap_triggered
            points = max(-ABSOLUTE_SWING_CAP, min(ABSOLUTE_SWING_CAP, points))

            # Persist opponent-quality diagnostics for audit/ablation analysis.
            oq_quality = quality_score(opponent_pre_fight_record, opponent_is_champion, opponent_title_defenses)
            oq_win_mult = opponent_quality_multiplier(opponent_pre_fight_record, opponent_is_champion, opponent_title_defenses, k=opponent_quality_k)
            # opponent_quality_adjustment() reflects this around 1.0 for losses
            # (2.0 - win_multiplier, reclipped) -- mirror that here so this
            # logged column matches the multiplier actually applied to points,
            # rather than always showing the un-reflected win-direction value.
            oq_mult = oq_win_mult if result == 'W' else max(OQ_MIN_MULTIPLIER, min(OQ_MAX_MULTIPLIER, 2.0 - oq_win_mult))
            df.at[index, f'quality_score_{opponent_col}'] = round(oq_quality, 6)
            df.at[index, f'quality_multiplier_{opponent_col}'] = round(oq_mult, 6)
            df.at[index, f'opponent_quality_adjustment_{fighter_col}'] = opponent_quality_pts

            post_fight_ude = pre_fight_ude + points
            post_fight_updates[fighter] = post_fight_ude

            df.at[index, f'ude_points_pre_fight_{fighter_col}'] = pre_fight_ude
            df.at[index, f'ude_points_post_fight_{fighter_col}'] = post_fight_ude

        fighter_ude_points.update(post_fight_updates)
        for furl, new_rating in post_fight_updates.items():
            fighter_rating_history[furl].append(new_rating)

        pair_key = tuple(sorted((fighter_1_url, fighter_2_url)))
        state = pair_history_cache.setdefault(
            pair_key, {'count': 0, 'first_result_by_fighter': {}}
        )
        if state['count'] == 0:
            state['first_result_by_fighter'][fighter_1_url] = row['fight_result_fighter_1']
            state['first_result_by_fighter'][fighter_2_url] = row['fight_result_fighter_2']
        state['count'] += 1

    bind_rate = (swing_cap_triggered_count / total_scored_fighter_fights
                 if total_scored_fighter_fights else 0.0)
    logger.info(
        'ABSOLUTE_SWING_CAP (%s pts) triggered %d/%d scored fighter-fight '
        'observations (%.4f%% bind rate).',
        ABSOLUTE_SWING_CAP, swing_cap_triggered_count, total_scored_fighter_fights,
        bind_rate * 100,
    )
    df.attrs['absolute_swing_cap_bind_count'] = swing_cap_triggered_count
    df.attrs['absolute_swing_cap_total_observations'] = total_scored_fighter_fights
    df.attrs['absolute_swing_cap_bind_rate'] = bind_rate

    return df.sort_values(by='event_date', ascending=False).reset_index(drop=True)


def add_ude_points_difference_columns(df):
    """
    Adds two columns to the dataframe for Ude points difference.
    One for fighter_1 and one for fighter_2.
    """
    df['ude_points_diff_fighter_1'] = df['ude_points_post_fight_fighter_1'] - df['ude_points_pre_fight_fighter_1']
    df['ude_points_diff_fighter_2'] = df['ude_points_post_fight_fighter_2'] - df['ude_points_pre_fight_fighter_2']

    return df
