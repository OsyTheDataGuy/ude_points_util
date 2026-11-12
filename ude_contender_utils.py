"""Offline contender policy: evidence, qualification, priority and matchmaking.

These are editorial rules, not UFC rules. Historical incoming ranks must be
verified by the caller; current ranks and bookings never determine eligibility.
All clocks use the supplied fight-data cutoff. No scraping or implicit today().
"""
from dataclasses import dataclass, asdict
import pandas as pd


@dataclass(frozen=True)
class ContenderPolicy:
    title_wins: int = 3
    title_opponent_rank: int = 5
    top_five_test_wins: int = 2
    top_five_test_opponent_rank: int = 15
    ranked_win_months: int = 12

    def __post_init__(self):
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if not self.title_opponent_rank <= self.top_five_test_opponent_rank <= 15:
            raise ValueError('Opponent rank thresholds must satisfy title <= test <= 15')


DEFAULT_CONTENDER_POLICY = ContenderPolicy()


CATCHWEIGHT = 'Catch Weight Bout'
NO_DIVISION = (CATCHWEIGHT, 'Open Weight Bout')


def attribute_catchweight_divisions(fights):
    """Give each catchweight bout a division when its neighbours agree.

    fights is the fight table (current_df shape): fight_url, event_date,
    weight_class_cleaned, fighter_url_fighter_1, fighter_url_fighter_2.
    For each catchweight bout, take each fighter's nearest non-catchweight
    fight before it and after it (up to 4 fights). If they are all in one
    division, the bout is 'resolved' to that division; otherwise 'ambiguous'
    and division is None. A fighter with no fight on one side adds nothing
    from that side.

    The 'after' fight looks ahead in time. That is acceptable for deciding
    which division a past bout belonged to, but the result must not be used
    as if it were known on the night of the bout.

    Newer, lower-verification construct (2026-11-12): the data does not record
    the intended division of a catchweight bout; this is an inference.
    Returns one row per catchweight bout: fight_url, division, status,
    neighbour_divisions.
    """
    f = fights.copy()
    f['event_date'] = pd.to_datetime(f.event_date)
    sides = pd.concat([f[['fight_url', 'event_date', 'weight_class_cleaned', c]].rename(columns={c: 'fighter_id'})
                       for c in ['fighter_url_fighter_1', 'fighter_url_fighter_2']])
    known = sides[~sides.weight_class_cleaned.isin(NO_DIVISION)].sort_values('event_date')
    by_fighter = dict(tuple(known.groupby('fighter_id')))
    rows = []
    for b in f[f.weight_class_cleaned == CATCHWEIGHT].itertuples(index=False):
        divisions = set()
        for fid in (b.fighter_url_fighter_1, b.fighter_url_fighter_2):
            h = by_fighter.get(fid)
            if h is None:
                continue
            divisions |= set(h[h.event_date < b.event_date].weight_class_cleaned.tail(1))
            divisions |= set(h[h.event_date > b.event_date].weight_class_cleaned.head(1))
        one = len(divisions) == 1
        rows.append(dict(fight_url=b.fight_url, division=next(iter(divisions)) if one else None,
                         status='resolved' if one else 'ambiguous', neighbour_divisions=sorted(divisions)))
    return pd.DataFrame(rows, columns=['fight_url', 'division', 'status', 'neighbour_divisions'])


def current_division_win_run(history, division, as_of):
    """Return winning rows in the current strict division run, chronologically.

    history is one fighter's complete ledger, with date, division, result,
    fight_url columns. NC preserves a run without adding wins. L/D and a bout
    in another division interrupt it. Future rows are ignored. Additional
    columns and original indexes are preserved.

    Catchweight bouts: set their division with attribute_catchweight_divisions
    before calling. A resolved catchweight then counts in its division; an
    ambiguous one keeps the catchweight label and interrupts the run. The row
    that ended the run is returned in result.attrs['stopped_by'] (None if the
    run covers the whole career), so callers can flag runs ended by an
    ambiguous catchweight for review.

    Newer, lower-verification construct (2026-11-12) for contender content. It
    does not replace update_win_streaks in ude_points_feature_engineering_pipeline.py,
    which stays the central streak logic; test_contender_policy.py checks that the
    two agree for single-division careers.
    """
    required = {'date', 'division', 'result', 'fight_url'}
    if not required <= set(history):
        raise ValueError(f'Missing run columns: {sorted(required - set(history))}')
    h = history.copy()
    cutoff = pd.Timestamp(as_of)
    if pd.isna(cutoff):
        raise ValueError('An explicit valid fight-data cutoff is required')
    if not h.index.is_unique:
        raise ValueError('Run history requires unique row indexes')
    h['date'] = pd.to_datetime(h.date, errors='raise')
    if h.date.isna().any() or h.fight_url.isna().any() or h.fight_url.duplicated().any():
        raise ValueError('Run history requires dated, uniquely identified bouts')
    if 'fighter_id' in h and h.fighter_id.nunique() > 1:
        raise ValueError('Run history must contain exactly one fighter')
    h = h[h.date <= cutoff].sort_values(['date', 'fight_url'])
    if not h.result.isin(['W', 'L', 'D', 'NC']).all():
        raise ValueError('Unknown result in completed fight history')
    indexes, stopped_by = [], None
    for ix, b in h.iloc[::-1].iterrows():
        if b.division != division or b.result not in ['W', 'NC']:
            stopped_by = dict(fight_url=b.fight_url, date=str(b.date.date()), division=b.division, result=b.result)
            break
        if b.result == 'W':
            indexes.append(ix)
    run = h.loc[indexes[::-1]].copy()
    run.attrs['stopped_by'] = stopped_by
    return run


def assess_contender_qualification(current_run_wins, as_of,
                                   policy=DEFAULT_CONTENDER_POLICY,
                                   cross_division_champion=False):
    """Assess every win in a current run, returning evidence and both gates.

    Each record needs date, opponent_rank (0 = champion, 1..15 = ranked,
    null = unknown/unranked), rank_status and rank_source. Null ranks count
    as proven unranked ONLY with rank_status='unranked'. Missing evidence
    remains unresolved. The caller must supply the ENTIRE current winning
    run, not just ranked wins. Cross-division champion exceptions are flagged
    for editorial review; this function never awards one automatically.
    """
    cutoff = pd.Timestamp(as_of)
    if pd.isna(cutoff):
        raise ValueError('An explicit valid fight-data cutoff is required')
    cutoff = cutoff.normalize()
    window = cutoff - pd.DateOffset(months=policy.ranked_win_months)
    records = (current_run_wins.to_dict('records') if isinstance(current_run_wins, pd.DataFrame)
               else [dict(r) for r in current_run_wins])
    ranked, unknown, evidence = [], [], []
    seen = set()
    for raw in records:
        r = dict(raw)
        date = pd.Timestamp(r['date'])
        if pd.isna(date) or date > cutoff:
            raise ValueError('Qualification evidence must be dated at/before cutoff')
        date = date.normalize()
        key = r.get('fight_url')
        if key and key in seen:
            raise ValueError('Duplicate win in qualification evidence')
        seen.add(key)
        if r.get('result', 'W') != 'W':
            raise ValueError('Qualification evidence must contain wins only')
        rank = r.get('opponent_rank')
        status = r.get('rank_status', 'unknown')
        if rank is not None and not pd.isna(rank):
            if isinstance(rank, bool) or float(rank) != int(float(rank)) or not 0 <= float(rank) <= 15:
                raise ValueError('Incoming opponent rank must be 0..15 or null')
            if status == 'unranked':
                raise ValueError('Contradictory ranked/unranked evidence')
            rank = int(float(rank))
            if not r.get('rank_source'):
                raise ValueError('A verified incoming rank needs a source')
            r['opponent_rank'] = rank
            ranked.append(r)
        else:
            r['opponent_rank'] = None
            if status != 'unranked':
                unknown.append(r)
        r['date'] = str(date.date())
        evidence.append(r)
    n = len(evidence)

    def gate(min_wins, max_rank):
        all_matches = [r for r in ranked if r['opponent_rank'] <= max_rank]
        recent = [r for r in all_matches if pd.Timestamp(r['date']) >= window]
        uncertain = [r for r in unknown if pd.Timestamp(r['date']) >= window]
        status = ('qualified' if n >= min_wins and recent else
                  'ranking_review' if n >= min_wins and uncertain else 'not_yet')
        return status, all_matches, recent

    title, top5, recent5 = gate(policy.title_wins, policy.title_opponent_rank)
    test, top15, recent15 = gate(policy.top_five_test_wins, policy.top_five_test_opponent_rank)
    ordinary = title
    if cross_division_champion and title != 'qualified':
        title = 'exception_review'
    expired = n >= policy.title_wins and bool(top5) and not recent5
    latest = max((pd.Timestamp(r['date']) for r in top5), default=None)
    return dict(title_status=title, ordinary_title_status=ordinary, top_five_test_status=test,
                target_division_wins=n, recent_top5_wins=len(recent5),
                current_run_top5_wins=len(top5), current_run_ranked_wins=len(top15),
                recent_ranked_wins=len(recent15), unknown_rank_wins=len(unknown),
                minimum_more_wins=max(max(0, policy.title_wins-n), int(not bool(recent5))),
                title_window_expired=expired,
                review_flags=(['top_five_win_window_expired'] if expired else []) +
                             (['cross_division_champion_exception'] if cross_division_champion else []),
                last_qualifying_win_valid_through=str((latest + pd.DateOffset(months=policy.ranked_win_months)).date()) if latest is not None else None,
                last_win=max((r['date'] for r in evidence), default=None),
                as_of=str(cutoff.date()), window_start=str(window.date()),
                policy=asdict(policy), evidence=evidence, title_qualifying_wins=recent5,
                top_five_test_qualifying_wins=recent15)


DEFAULT_TITLE_PRIORITY = ('recent_top5_wins', 'current_run_top5_wins',
                          'current_run_ranked_wins', 'target_division_wins', 'last_win')


def prioritize_title_contenders(assessments, criteria=DEFAULT_TITLE_PRIORITY):
    """Order qualified fighters only; retain merit ties rather than force a winner.

    Explicit lexicographic editorial policy, descending on each criterion.
    Stable fighter_id orders display within ties; it never breaks a merit tie.
    Does not consider bookings or change qualification. Empty pools are valid.
    """
    frame = pd.DataFrame(assessments).copy()
    if not criteria or len(set(criteria)) != len(criteria):
        raise ValueError('Priority criteria must be nonempty and unique')
    if frame.empty:
        result = pd.DataFrame(columns=['division', 'fighter_id', 'title_status', 'title_priority'])
        result.attrs['priority_policy'] = dict(criteria=list(criteria), descending=True, ties='shared priority')
        return result
    required = {'division', 'fighter_id', 'title_status'} | set(criteria)
    if not required <= set(frame):
        raise ValueError(f'Missing priority columns: {sorted(required - set(frame))}')
    if frame.duplicated(['division', 'fighter_id']).any():
        raise ValueError('Duplicate fighter/division assessment')
    q = frame[frame.title_status == 'qualified'].copy()
    if q[list(criteria)].isna().any().any():
        raise ValueError('Qualified contenders need complete priority evidence')
    q = q.sort_values(['division', *criteria, 'fighter_id'], ascending=[True]+[False]*len(criteria)+[True])
    q['title_priority'] = 0
    for _, group in q.groupby('division', sort=False):
        previous = None
        priority = 0
        for ix, row in group.iterrows():
            key = tuple(row[c] for c in criteria)
            if key != previous:
                priority += 1
                previous = key
            q.loc[ix, 'title_priority'] = priority
    q.attrs['priority_policy'] = dict(criteria=list(criteria), descending=True, ties='shared priority')
    return q


def fight_down_pool_decision(qualification, *, booking_status,
                            roster_status='current', is_champion=False):
    """Filter independently of merit order, returning every exclusion reason.

    not_listed means a dated successful check found no booking, NOT medical
    clearance. Expired-window flags do not silently create a title exemption.
    Unresolved title qualification is held for review, never treated as failed.
    """
    if booking_status not in ['booked', 'not_listed', 'unchecked']:
        raise ValueError('Unknown booking status')
    if roster_status not in ['current', 'signed', 'excluded', 'unavailable']:
        raise ValueError('Unknown roster status')
    status = qualification['title_status']
    if status not in ['qualified', 'not_yet', 'ranking_review', 'exception_review']:
        raise ValueError('Unknown title qualification status')
    reasons = []
    if is_champion: reasons.append('current_champion')
    if roster_status in ['excluded', 'unavailable']: reasons.append(roster_status)
    if booking_status == 'booked': reasons.append('booked')
    if booking_status == 'unchecked': reasons.append('booking_check_required')
    if status == 'qualified': reasons.append('title_qualified')
    elif status != 'not_yet': reasons.append('title_qualification_review')
    return dict(include=not reasons, reasons=reasons, review_flags=qualification.get('review_flags', []))
