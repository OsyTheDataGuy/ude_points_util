"""
Known-answer tests for UDE Points.

Run from the project root:   python -m pytest test_ude_points.py -q

What this file checks, by section:

  1. Dataset invariants   -- current_df.csv still has the shape every other
                             piece of code assumes (unique fights, storage
                             order, strike breakdowns that add up, scope).
  2. Scraper scope rule   -- Road to UFC is dropped; UFC cards whose name
                             merely contains "Road to" are kept.
  3. Fighter identity     -- fighters are matched by fighter_url, not name:
                             a renamed fighter's career stays whole, two
                             fighters sharing a name stay apart.
  4. Fighter profile      -- the profile's dynamic_* state equals what the
                             pipeline itself stores, and it includes the
                             fighter's latest fight.
  5. Opponent similarity  -- clean errors for a debut or a misspelled name,
                             and one pinned real matchup ranking.
  6. Archetype            -- pinned scores and labels for known fighters.
  7. Defensive vuln.      -- pinned per-area results for known fighters.
  8. GOAT ranking         -- the canonical_project_state.md section 2a table.

How the pinned numbers stay stable: every pinned check runs with
AS_OF = '2026-10-04', so it only sees fights up to 2026-10-03. New fights from
the weekly refresh don't change these results. They should only change when
historical data or the code that scores it changes. When that is deliberate
(e.g. a planned re-score), update the pinned value in the same commit and say
why. When it isn't, the failing test has found a regression.

Every example names its fighter_url / fight_url id so the record can be pulled
and checked by hand.
"""
import pandas as pd
import pytest

import ude_points_utils as u

AS_OF = '2026-10-04'  # pinned cut-off: data through 2026-10-03


@pytest.fixture(scope='module')
def df():
    """The live dataset, loaded once for the whole file."""
    return pd.read_csv('current_df.csv', low_memory=False)


def url_id(url):
    """Last path segment of a UFCStats URL -- the short id used in the docs."""
    return str(url).rstrip('/').split('/')[-1]


# ---------------------------------------------------------------------------
# 1. Dataset invariants
# ---------------------------------------------------------------------------

def test_one_row_per_fight(df):
    assert df['fight_url'].is_unique


def test_stored_newest_first(df):
    # Storage order is date-descending; code that takes "the latest row"
    # without sorting relies on this.
    assert pd.to_datetime(df['event_date']).is_monotonic_decreasing


@pytest.mark.parametrize('side', ['fighter_1', 'fighter_2'])
def test_strike_breakdowns_add_up_to_sig_strikes(df, side):
    # UFCStats splits significant strikes two ways: by POSITION
    # (distance/clinch/ground) and by TARGET (head/body/leg). Each split must
    # sum exactly to the total -- the vulnerability areas and the similarity
    # axes are designed around this.
    sig = df[f'sig_strikes_attempted_{side}']
    by_position = sum(df[f'{p}_strikes_attempted_{side}'] for p in ('distance', 'clinch', 'ground'))
    by_target = sum(df[f'{t}_strikes_attempted_{side}'] for t in ('head', 'body', 'leg'))
    assert (by_position == sig).all()
    assert (by_target == sig).all()


def test_no_out_of_scope_events_in_dataset(df):
    assert not df['event_name'].str.lower().str.contains('road to ufc').any()


# ---------------------------------------------------------------------------
# 2. Scraper scope rule
# ---------------------------------------------------------------------------

def test_road_to_ufc_dropped_but_ufc_18_kept():
    scrape = pytest.importorskip('ude_scrape_new')  # needs bs4 installed
    events = pd.DataFrame({'EVENT': [
        'UFC - Road to UFC 4.6',                       # out of scope
        'UFC 18: The Road to the Heavyweight Title',   # a real UFC card
        'UFC 300: Pereira vs. Hill',
    ]})
    kept = scrape.exclude_out_of_scope_events(events)['EVENT'].tolist()
    assert kept == ['UFC 18: The Road to the Heavyweight Title', 'UFC 300: Pereira vs. Hill']


# ---------------------------------------------------------------------------
# 3. Fighter identity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('name', ['Waldo Cortes-Acosta', 'Waldo Cortes Acosta'])
def test_renamed_fighter_career_is_whole(df, name):
    # fighter_url fc08099550072fe4 fought 9 times as 'Waldo Cortes-Acosta',
    # then 5 times as 'Waldo Cortes Acosta'. Either name returns all 14.
    career = u.create_fighter_career_dataset(df[pd.to_datetime(df['event_date']) < AS_OF], name)
    assert len(career) == 14
    assert career['fighter_url'].map(url_id).unique().tolist() == ['fc08099550072fe4']


def test_new_name_works_before_the_rename(df):
    # Montse Rendon (60193e707634e560) fought as 'Montserrat Rendon' until
    # 2024-03-23. Her new name must still find those earlier fights.
    profile = u.generate_fighter_profile(df, 'Montse Rendon', as_of='2025-09-13').iloc[0]
    assert url_id(profile['fighter_url']) == '60193e707634e560'
    assert str(profile['profile_event_date'])[:10] == '2024-03-23'


def test_shared_name_raises_and_ids_work(df):
    # Two different fighters are both 'Bruno Silva'. The name alone is
    # ambiguous; each fighter_url id gives that fighter's own career.
    before = df[pd.to_datetime(df['event_date']) < AS_OF]
    with pytest.raises(ValueError, match='2 different fighters'):
        u.create_fighter_career_dataset(before, 'Bruno Silva')
    assert len(u.create_fighter_career_dataset(before, '12ebd7d157e91701')) == 11
    assert len(u.create_fighter_career_dataset(before, '294aa73dbf37d281')) == 12


def test_rankings_keep_shared_name_fighters_apart(df):
    # Grouping by name used to merge the two Bruno Silvas into one row.
    names = set(u.with_one_name_per_fighter(df)['fighter_1'])
    assert {'Bruno Silva (12ebd7d157e91701)', 'Bruno Silva (294aa73dbf37d281)'} <= names
    assert 'Bruno Silva' not in names


# ---------------------------------------------------------------------------
# 4. Fighter profile
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('fighter, fight_id, fight_date', [
    # Islam Makhachev (275aca31f61ba28c) entering his fight vs Ian Machado Garry
    ('Islam Makhachev', '365fd759c03e93cf', '2026-08-15'),
    # Petr Yan (d661ce4da776fc20) entering his fight vs Merab Dvalishvili
    ('Petr Yan', '4a0db214d9721d6e', '2025-12-06'),
])
def test_profile_matches_pipeline_state(df, fighter, fight_id, fight_date):
    # With as_of = a fight's date, the profile is the state after the
    # fighter's previous fight -- which is exactly what the pipeline stored
    # on this fight's row as "state entering it". All style columns must match.
    profile = u.generate_fighter_profile(df, fighter, as_of=fight_date).iloc[0]
    row = df[df['fight_url'].str.endswith(fight_id)].iloc[0]
    side = 'fighter_1' if row['fighter_1'] == fighter else 'fighter_2'
    for col in u.STYLE_SIMILARITY_COLUMNS:
        stored = row[f'{col}_{side}']
        assert (pd.isna(profile[col]) and pd.isna(stored)) or profile[col] == pytest.approx(stored), col


def test_profile_includes_latest_fight(df):
    # Islam Makhachev (275aca31f61ba28c). His latest fight before AS_OF is
    # 365fd759c03e93cf (2026-08-15, 7 of 15 takedowns). Entering it his
    # stored td_accuracy was 0.517; the profile must count that fight too.
    profile = u.generate_fighter_profile(df, 'Islam Makhachev', as_of=AS_OF).iloc[0]
    assert url_id(profile['fighter_url']) == '275aca31f61ba28c'
    assert str(profile['profile_event_date'])[:10] == '2026-08-15'

    # Independent check: rebuild the shrunk career takedown accuracy from the
    # raw per-fight counts. Prior constants are add_dynamic_td_accuracy's.
    url = profile['fighter_url']
    before = df[pd.to_datetime(df['event_date']) < AS_OF]
    landed = attempted = 0
    for side in ('fighter_1', 'fighter_2'):
        own = before[before[f'fighter_url_{side}'] == url]
        landed += own[f'td_landed_{side}'].sum()
        attempted += own[f'td_attempted_{side}'].sum()
    expected = round(u._shrink_rate(landed, attempted, 21.74, 0.3684), 3)

    assert profile['dynamic_td_accuracy'] == pytest.approx(expected)
    assert profile['dynamic_td_accuracy'] != pytest.approx(0.517)


# ---------------------------------------------------------------------------
# 5. Opponent similarity
# ---------------------------------------------------------------------------

def test_similarity_debut_fighter_raises(df):
    # Charalampos Grigoriou (68b8ebdfce9dbb61) made his debut on 2024-03-16
    # (fight f1cbd73e7be64d28 vs Chad Anheliger): no prior fights to compare.
    with pytest.raises(ValueError, match='No fights found'):
        u.find_most_similar_past_opponents(df, 'Charalampos Grigoriou', 'Chad Anheliger', as_of='2024-03-16')


def test_similarity_misspelled_name_raises(df):
    with pytest.raises(ValueError, match='No fights found'):
        u.find_most_similar_past_opponents(df, 'Islam Makhachevv', 'Ilia Topuria')


def test_similarity_pinned_matchup(df):
    # Ilia Topuria's past opponents ranked by style similarity to
    # Islam Makhachev (275aca31f61ba28c).
    _, style = u.find_most_similar_past_opponents(df, 'Ilia Topuria', 'Islam Makhachev', as_of=AS_OF)
    top3 = style.head(3)
    assert top3['opponent'].tolist() == ['Youssef Zalal', 'Charles Oliveira', 'Bryce Mitchell']
    assert top3['total_difference'].round(3).tolist() == [0.964, 1.075, 1.090]


# ---------------------------------------------------------------------------
# 6. Archetype
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('fighter, fighter_id, orientation, style, label', [
    ('Khabib Nurmagomedov', '032cc3922d871c7f', 1.933, 4.746, 'ground-and-pound-leaning grappler'),
    ('Islam Makhachev', '275aca31f61ba28c', 1.233, -1.872, 'control/submission-leaning grappler'),
    ('Jailton Almeida', '41e83a89929d1327', 2.874, 0.003, 'balanced on the ground'),
    ('Israel Adesanya', '1338e2c7480bdf9e', -1.178, -0.127, 'primarily a striker'),
    ('Michael Page', 'a67d071163962af8', -1.063, -0.968, 'primarily a striker'),
])
def test_archetype_pinned(df, fighter, fighter_id, orientation, style, label):
    result = u.classify_fighter_archetype(df, fighter, as_of=AS_OF).iloc[0]
    assert url_id(result['fighter_url']) == fighter_id
    assert result['grappling_orientation'] == pytest.approx(orientation, abs=1e-3)
    assert result['ground_game_style'] == pytest.approx(style, abs=1e-3)
    assert result['archetype_label'] == label


# ---------------------------------------------------------------------------
# 7. Defensive vulnerability
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('fighter, fighter_id, area, landed, faced, z, label', [
    ('Jon Jones', '07f72a2a7591b409', 'td', 2, 40, -2.871, 'strong'),
    ('Michael Page', 'a67d071163962af8', 'distance', 85, 282, -3.094, 'strong'),
    ('Rob Font', '05339613bf8e9808', 'td', 56, 89, 2.018, 'weak'),
])
def test_vulnerability_pinned(df, fighter, fighter_id, area, landed, faced, z, label):
    result = u.assess_defensive_vulnerability(df, fighter, as_of=AS_OF)
    row = result[result['area'] == area].iloc[0]
    assert url_id(row['fighter_url']) == fighter_id
    assert (row['landed_against'], row['attempts_faced']) == (landed, faced)
    assert row['vulnerability_z'] == pytest.approx(z, abs=1e-3)
    assert row['vulnerability_label'] == label


# ---------------------------------------------------------------------------
# 8. GOAT ranking (canonical_project_state.md section 2a)
# ---------------------------------------------------------------------------

def test_goat_ranking_top3(df):
    before = df[pd.to_datetime(df['event_date']) < AS_OF]
    ranking = u.rank_fighters_by_shrunk_ude_rate(before, prior_strength=10.0, min_fights=10)
    assert len(ranking) == 627  # fighters clearing the 10-fight floor
    top3 = ranking.head(3)
    assert [url_id(x) for x in top3['fighter_url']] == [
        '6506c1d34da9c013',  # Georges St-Pierre
        '07f72a2a7591b409',  # Jon Jones
        '275aca31f61ba28c',  # Islam Makhachev
    ]
    assert top3['shrunk_rate'].round(3).tolist() == [4.351, 4.264, 3.672]
