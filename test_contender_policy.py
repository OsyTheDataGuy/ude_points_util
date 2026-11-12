"""Policy boundary tests. Rank evidence is synthetic (no private rankings); the
catchweight and streak cross-checks read the public current_df.csv."""
from pathlib import Path
import pandas as pd
import pytest
from ude_contender_utils import (
    ContenderPolicy, current_division_win_run, assess_contender_qualification,
    prioritize_title_contenders, fight_down_pool_decision, attribute_catchweight_divisions,
    NO_DIVISION,
)

# current_df.csv sits in the repo root; look upward so the tests also run from a subfolder.
DATA = next(p/'current_df.csv' for p in Path(__file__).resolve().parents if (p/'current_df.csv').exists())
COLS = ['event_date', 'fight_url', 'weight_class_cleaned', 'fighter_url_fighter_1', 'fighter_url_fighter_2',
        'fight_result_fighter_1', 'fight_result_fighter_2', 'W/L_streak_fighter_1', 'W/L_streak_fighter_2']


@pytest.fixture(scope='module')
def fights():
    return pd.read_csv(DATA, usecols=COLS)

def win(date, rank=None, status='unranked', key=None):
    return dict(date=date, opponent_rank=rank, rank_status='ranked' if rank is not None else status,
                rank_source='Synthetic incoming-rank fixture', fight_url=key or date)

def assess(*wins, **kwargs):
    return assess_contender_qualification(wins, '2026-11-07', **kwargs)

@pytest.mark.parametrize('rank,qualified', [(0,True),(1,True),(5,True),(6,False),(15,False),(None,False)])
def test_title_rank_boundary(rank,qualified):
    q=assess(win('2026-01-01'),win('2026-04-01'),win('2026-07-01',rank))
    assert (q['title_status']=='qualified') is qualified

def test_distinct_opportunity_gates():
    q=assess(win('2026-01-01'),win('2026-07-01',5))
    assert q['top_five_test_status']=='qualified'
    assert q['title_status']=='not_yet'
    assert q['minimum_more_wins']==1

def test_calendar_window_inclusive_and_expiry_flag():
    q=assess(win('2025-11-07',5),win('2026-04-01'),win('2026-07-01'))
    assert q['title_status']=='qualified'
    expired=assess(win('2025-11-06',5),win('2026-04-01'),win('2026-07-01'))
    assert expired['title_status']=='not_yet'
    assert expired['title_window_expired']
    assert expired['last_qualifying_win_valid_through']=='2026-11-06'
    assert fight_down_pool_decision(expired,booking_status='not_listed')['include']

def test_unknown_is_not_unranked_or_a_failure():
    q=assess(win('2026-01-01'),win('2026-04-01'),win('2026-07-01',status='stale_or_nonprior_rank'))
    assert q['title_status']=='ranking_review'
    assert not fight_down_pool_decision(q,booking_status='not_listed')['include']
    known=assess(win('2026-01-01'),win('2026-04-01'),win('2026-07-01'))
    assert known['title_status']=='not_yet'

def test_known_sufficient_win_still_qualifies_with_other_unknowns():
    q=assess(win('2026-01-01',status='unknown'),win('2026-04-01',5),win('2026-07-01'))
    assert q['title_status']=='qualified'
    assert q['unknown_rank_wins']==1

def test_champion_exception_requires_review():
    q=assess(cross_division_champion=True)
    assert q['ordinary_title_status']=='not_yet'
    assert q['title_status']=='exception_review'

@pytest.mark.parametrize('records', [
    [win('2026-11-08',5)], [win('2026-01-01'),win('2026-01-01')],
    [win('2026-01-01',16)], [dict(win('2026-01-01',5),rank_source='')],
    [dict(win('2026-01-01',5),rank_status='unranked')],
])
def test_invalid_evidence_fails_loudly(records):
    with pytest.raises(ValueError):assess(*records)

def test_run_nc_loss_draw_crossdivision_and_future():
    h=pd.DataFrame([dict(date=f'2026-0{i+1}-01',division='BW',result=r,fight_url=str(i))
                    for i,r in enumerate(['L','W','NC','W','W'])])
    original=h.copy(deep=True)
    assert len(current_division_win_run(h,'BW','2026-11-07'))==3
    assert len(current_division_win_run(h,'BW','2026-03-01'))==1
    pd.testing.assert_frame_equal(h,original)
    for result in ['L','D']:
        h.loc[3,'result']=result
        assert len(current_division_win_run(h,'BW','2026-11-07'))==1
    h=original.copy();h.loc[3,'division']='CW'
    assert len(current_division_win_run(h,'BW','2026-11-07'))==1

def test_independent_booking_filter_and_shared_priority_ties():
    q=assess(win('2026-01-01'),win('2026-04-01'),win('2026-07-01',5))
    rows=[dict(q,division='BW',fighter_id=fid) for fid in ['b','a']]
    rows.append(dict(q,division='BW',fighter_id='c',title_status='not_yet'))
    result=prioritize_title_contenders(rows)
    assert list(result.fighter_id)==['a','b']
    assert list(result.title_priority)==[1,1]
    assert fight_down_pool_decision(q,booking_status='booked')['reasons']==['booked','title_qualified']
    assert fight_down_pool_decision(q,booking_status='not_listed')['reasons']==['title_qualified']
    assert q['title_status']=='qualified'
    assert prioritize_title_contenders([rows[-1]]).empty
    assert prioritize_title_contenders([]).empty
    assert prioritize_title_contenders([]).attrs['priority_policy']['ties']=='shared priority'

def test_explicit_cutoff_and_unambiguous_history_required():
    h=pd.DataFrame([dict(date='2026-01-01',division='BW',result='W',fight_url='a')])
    with pytest.raises(ValueError):current_division_win_run(h,'BW',None)
    with pytest.raises(ValueError):assess_contender_qualification([],None)
    h=pd.concat([h,h.assign(fight_url='b')])
    with pytest.raises(ValueError):current_division_win_run(h,'BW','2026-11-07')

def test_priority_is_separate_configurable_and_rejects_duplicates():
    q=assess(win('2026-01-01'),win('2026-04-01'),win('2026-07-01',5))
    a=dict(q,division='BW',fighter_id='a',recent_top5_wins=2,target_division_wins=3)
    b=dict(q,division='BW',fighter_id='b',recent_top5_wins=1,target_division_wins=6)
    assert list(prioritize_title_contenders([a,b]).fighter_id)==['a','b']
    assert list(prioritize_title_contenders([a,b],criteria=('target_division_wins',)).fighter_id)==['b','a']
    with pytest.raises(ValueError):prioritize_title_contenders([a,a])

def test_policy_configuration_and_public_reexport():
    with pytest.raises(ValueError):ContenderPolicy(ranked_win_months=0)
    with pytest.raises(ValueError):ContenderPolicy(title_opponent_rank=16)
    import ude_points_utils as u
    assert u.assess_contender_qualification is assess_contender_qualification


def test_catchweight_resolved_and_ambiguous(fights):
    a = attribute_catchweight_divisions(fights).set_index('fight_url')
    url = 'http://ufcstats.com/fight-details/'
    # Ortega vs. Sterling, 2025-08-23: both fighters' neighbouring fights are featherweight.
    assert a.loc[url + '58080e8989927500', 'status'] == 'resolved'
    assert a.loc[url + '58080e8989927500', 'division'] == 'FW'
    # Rebecki vs. Orolbai, 2024-10-26: neighbours are lightweight and welterweight.
    assert a.loc[url + '919e3c5cc6999cb3', 'status'] == 'ambiguous'
    assert a.loc[url + '919e3c5cc6999cb3', 'division'] is None
    assert a.loc[url + '919e3c5cc6999cb3', 'neighbour_divisions'] == ['LW', 'WW']


def test_resolved_catchweight_continues_run_and_ambiguous_stops_it():
    h = pd.DataFrame([dict(date=f'2026-0{i+1}-01', division=d, result='W', fight_url=str(i))
                      for i, d in enumerate(['FW', 'FW', 'Catch Weight Bout', 'FW'])])
    stopped = current_division_win_run(h, 'FW', '2026-11-07')
    assert len(stopped) == 1 and stopped.attrs['stopped_by']['division'] == 'Catch Weight Bout'
    h.loc[2, 'division'] = 'FW'   # attributed by attribute_catchweight_divisions
    full = current_division_win_run(h, 'FW', '2026-11-07')
    assert len(full) == 4 and full.attrs['stopped_by'] is None


def test_run_matches_pipeline_streak_for_single_division_careers(fights):
    """For careers in one division with no catchweight or open-weight bouts, the
    run length must equal update_win_streaks' streak after the last fight."""
    f = fights.copy()
    f['event_date'] = pd.to_datetime(f.event_date)
    sides = pd.concat([f[['event_date', 'fight_url', 'weight_class_cleaned']].assign(
        fighter_id=f[f'fighter_url_fighter_{s}'], result=f[f'fight_result_fighter_{s}'],
        streak_in=f[f'W/L_streak_fighter_{s}']) for s in (1, 2)])
    divisions = sides.groupby('fighter_id').weight_class_cleaned.agg(set)
    single = divisions[divisions.map(lambda d: len(d) == 1 and not d & set(NO_DIVISION))].index
    checked = 0
    for fid, h in sides[sides.fighter_id.isin(single)].groupby('fighter_id'):
        h = h.sort_values(['event_date', 'fight_url'])
        last = h.iloc[-1]
        s, r = int(last.streak_in), last.result   # streak entering the last fight, then its result
        after = (s + 1 if s >= 0 else 1) if r == 'W' else 0 if r == 'D' else s if r == 'NC' else (s - 1 if s <= 0 else -1)
        run = current_division_win_run(h.rename(columns={'event_date': 'date', 'weight_class_cleaned': 'division'}),
                                       last.weight_class_cleaned, last.event_date)
        assert len(run) == max(after, 0), (fid, len(run), after)
        checked += 1
    assert checked > 1000
