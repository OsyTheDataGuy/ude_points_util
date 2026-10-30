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
                             fighters sharing a name stay apart. Title
                             defenses skip the fight that won the belt, and
                             off-cage strips/promotions are applied. Reign
                             date ranges agree with the champion flags and
                             with Wikipedia's reign lengths; the Wikipedia
                             parser keeps a note's text after a citation.
                             The UFC roster (ufc_roster.csv) reads every
                             name form and matches the right fighter.
  4. Fighter profile      -- the profile's dynamic_* state equals what the
                             pipeline itself stores, and it includes the
                             fighter's latest fight.
  5. Opponent similarity  -- clean errors for a debut or a misspelled name,
                             and one pinned real matchup ranking.
  6. Archetype            -- pinned scores and labels for known fighters.
  7. Defensive vuln.      -- pinned per-area results for known fighters.
  8. GOAT ranking         -- the canonical_project_state.md section 2a table.
  9. Fighter style table  -- fighter_style_table.csv is one row per fighter,
                             matches direct function calls, and refuses to
                             load against a different dataset.

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


@pytest.mark.parametrize('fight_id, fighter_id, record', [
    # Waldo Cortes Acosta's first fight under his new name: 7-2-0 carried
    # over, not a fresh 0-0-0.
    ('9d9390b550d77e06', 'fc08099550072fe4', '7-2-0 0'),
    # Bruno Silva 12ebd7d157e91701's UFC debut: 0-0-0, not the other Bruno
    # Silva's record.
    ('bb1ac382e0277b3f', '12ebd7d157e91701', '0-0-0 0'),
])
def test_pipeline_records_follow_fighter_url(df, fight_id, fighter_id, record):
    # The stored pre-fight record comes from the feature pipeline's state
    # machine, which must track fighters by fighter_url.
    row = df[df['fight_url'].str.endswith(fight_id)].iloc[0]
    side = 'fighter_1' if row['fighter_url_fighter_1'].endswith(fighter_id) else 'fighter_2'
    assert row[f'pre_fight_record_{side}_(W-L-D NC)'] == record


@pytest.mark.parametrize('fight_id, fighter_id, defenses', [
    # Georges St-Pierre entering Hendricks: 8. Winning the belt back from
    # Serra (67948da92e6c9bdc) as interim champion is not a defense.
    ('7ff036c99621eb1c', '6506c1d34da9c013', 8),
    # Jon Jones entering Reyes: 2. Winning the vacant belt from Gustafsson
    # (ceaae1db8ccde29c) while still flagged interim is not a defense.
    ('fae99a4089f8abb0', '07f72a2a7591b409', 2),
    # Off-cage title changes (title_changes_outside_cage.csv):
    # Ronda Rousey entering Holm: 6. She came from Strikeforce as champion,
    # so her first UFC title fight (Carmouche) is defense #1.
    ('08af25ef4252b138', '8bdac25ce0bb874d', 6),
    # Jose Aldo entering McGregor: 7. Same for his WEC transfer (Hominick #1).
    ('b302e12550278ffd', 'd0f3959b4a9747e6', 7),
    # Khabib Nurmagomedov entering Gaethje: 2. He retired after the fight; the
    # title was officially vacated on 2021-03-19.
    ('59eea37af7efb06c', '032cc3922d871c7f', 2),
])
def test_title_defenses_skip_belt_winning_fight(df, fight_id, fighter_id, defenses):
    # Only a win by the undisputed champion (is_champion == 2) in an
    # undisputed title bout counts as a defense.
    row = df[df['fight_url'].str.endswith(fight_id)].iloc[0]
    side = 'fighter_1' if row['fighter_url_fighter_1'].endswith(fighter_id) else 'fighter_2'
    assert row[f'title_defenses_{side}'] == defenses


def test_stripped_champion_not_flagged(df):
    # Charles Oliveira (07225ba28ae309b6) was stripped at the UFC 274
    # weigh-in, so he enters UFC 280 vs Makhachev (523815acf1bb2420) without
    # the belt.
    row = df[df['fight_url'].str.endswith('523815acf1bb2420')].iloc[0]
    side = 'fighter_1' if row['fighter_url_fighter_1'].endswith('07225ba28ae309b6') else 'fighter_2'
    assert row[f'is_champion_{side}'] == 0


def test_fight_day_strip_applies_before_the_card(df):
    # Oliveira (07225ba28ae309b6) was stripped on the day of UFC 274
    # (7d4e49d8a6678157), at the weigh-in, so he enters that fight without
    # the belt (fight_day_timing = before_card).
    row = df[df['fight_url'].str.endswith('7d4e49d8a6678157')].iloc[0]
    side = 'fighter_1' if row['fighter_url_fighter_1'].endswith('07225ba28ae309b6') else 'fighter_2'
    assert row[f'is_champion_{side}'] == 0


def test_fight_day_event_without_timing_stops(df):
    # An event on a day the fighter fights can't be ordered without
    # fight_day_timing: blank Oliveira's (UFC 274 weigh-in) and the pipeline must refuse.
    from ude_points_feature_engineering_pipeline import load_title_changes, check_fight_day_timing
    events = load_title_changes()
    events.loc[events['fighter'] == 'Charles Oliveira', 'fight_day_timing'] = ''
    with pytest.raises(ValueError, match='fight_day_timing'):
        check_fight_day_timing(df, events)


def test_reigns_agree_with_champion_flags(df):
    # undisputed_reigns: every undisputed title-bout side is entered flagged 2
    # exactly when it falls inside one of that fighter's reigns in that division.
    from ude_points_feature_engineering_pipeline import load_title_changes
    R = u.undisputed_reigns(df, load_title_changes(), '2026-10-30')
    t = df[df['is_title_bout'] == 2]
    bad = []
    for s in ('1', '2'):
        for r in t.itertuples():
            url, d = getattr(r, f'fighter_url_fighter_{s}'), pd.Timestamp(r.event_date)
            inside = ((R['fighter_url'] == url) & (R['weight_class_cleaned'] == r.weight_class_cleaned)
                      & (R['start'] < d) & (R['end'] >= d)).any()
            if inside != (getattr(r, f'is_champion_fighter_{s}') == 2):
                bad.append(r.fight_url)
    assert not bad, bad


@pytest.mark.parametrize('fighter_id, wc, start, days', [
    # Khabib Nurmagomedov (032cc3922d871c7f): Wikipedia lists 1,077 days, Apr 7,
    # 2018 to the official vacate on Mar 19, 2021 -- not his Oct 24, 2020 announcement.
    ('032cc3922d871c7f', 'LW', '2018-04-07', 1077),
    # Conor McGregor (f4c49976c75c5ab2): FW from Aldo (Dec 12, 2015) to the strip on
    # Nov 26, 2016 -- Wikipedia lists 350 days, 14 of them while also LW champion.
    ('f4c49976c75c5ab2', 'FW', '2015-12-12', 350),
])
def test_reign_lengths_match_wikipedia(df, fighter_id, wc, start, days):
    from ude_points_feature_engineering_pipeline import load_title_changes
    R = u.undisputed_reigns(df, load_title_changes(), '2026-10-30')
    r = R[R['fighter_url'].str.endswith(fighter_id) & (R['weight_class_cleaned'] == wc)
          & (R['start'] == pd.Timestamp(start))].iloc[0]
    assert (r['end'] - r['start']).days == days


def test_wikipedia_note_keeps_text_after_citation():
    # The parser used to cut a note at its first <ref>, losing "officially
    # vacated on ..." when it follows a citation (Khabib, Nunes on the live page).
    import title_changes_from_wikipedia as w
    page = '\n'.join([
        '===Lightweight Championship===',
        '! 10',
        '| {{flagicon|RUS}} [[Khabib Nurmagomedov]]',
        '| [[UFC 223]]',
        '| align=center| Apr 7, 2018',
        '| align=center| 1,077 days',
        '|',
        '|-',
        "! colspan=6 |''Nurmagomedov announced his retirement on October 24, 2020.<ref>{{cite web",
        '| url=https://example.com/a | title=t}}</ref> The title was officially vacated on March 19, 2021.\'\'',
        '|-',
    ])
    reigns, _, notes = w.parse_page(page)
    assert reigns['days'].tolist() == [1077]
    assert notes['text'].iloc[0].endswith('officially vacated on March 19, 2021.')
    events, _ = w.events_from_page(reigns, notes)
    assert events['date'].iloc[0] == pd.Timestamp('2021-03-19')


def test_title_changes_file_is_clean(df):
    # Every off-cage event names a fighter in the dataset, a known event type,
    # and a division that fighter has fought in.
    from ude_points_feature_engineering_pipeline import load_title_changes
    events = load_title_changes()
    assert set(events['event']) <= {'promoted', 'stripped', 'vacated', 'interim_stripped'}
    # Month-only dates stay marked as estimates (stored mid-month), apart from exact ones.
    assert set(events['date_precision']) <= {'day', 'month'}
    assert (events.loc[events['date_precision'] == 'month', 'date'].dt.day == 15).all()
    assert events['source_revision_url'].str.contains('oldid=').all()
    assert set(events['fight_day_timing']) <= {'', 'before_card', 'after_card'}
    pairs = set(zip(df['fighter_url_fighter_1'], df['weight_class_cleaned'])) | \
        set(zip(df['fighter_url_fighter_2'], df['weight_class_cleaned']))
    missing = [(e.fighter, e.weight_class_cleaned) for e in events.itertuples()
               if (e.fighter_url, e.weight_class_cleaned) not in pairs]
    assert not missing, missing


def test_roster_status_from_wikipedia(df):
    # ufc_roster.csv (Wikipedia, oldid=1382823554): Farid Basharat (a09ed60aa67a3f67)
    # was released on 2026-10-14 despite 7-0 in the UFC, so he is off the roster though
    # the 730-day rule still calls him active; Erik Perez (af62f99eb7606308) is not listed
    # at all; Sean O'Malley (b50a426a33da0012) and Song Yadong (efb96bf3e9ada36f) are current.
    roster = u.load_roster()
    basharat = roster[roster['fighter_url'].str.endswith('a09ed60aa67a3f67')].iloc[0]
    assert (basharat['status'], basharat['status_date'], basharat['reason']) == \
        ('released', '2026-10-14', 'Contract not renewed')
    status = u.create_fighter_status_dataset(df, as_of=pd.Timestamp('2026-10-30'), roster=roster)
    flags = status.set_index(status['fighter_url'].str[-16:])[['active', 'on_roster']]
    assert flags.loc['a09ed60aa67a3f67'].tolist() == [1, 0]
    assert flags.loc['af62f99eb7606308', 'on_roster'] == 0
    assert flags.loc[['b50a426a33da0012', 'efb96bf3e9ada36f'], 'on_roster'].tolist() == [1, 1]


def test_roster_file_is_clean(df):
    # One row per matched fighter, known statuses, every id a fighter in the dataset,
    # and the page revision cited.
    roster = u.load_roster()
    ids = roster.loc[roster['fighter_url'] != '', 'fighter_url']
    assert not ids.duplicated().any(), ids[ids.duplicated()].tolist()
    assert set(roster['status']) <= {'current', 'signed', 'released', 'retired'}
    known = set(df['fighter_url_fighter_1']) | set(df['fighter_url_fighter_2'])
    assert set(ids) <= known
    assert roster['source_revision_url'].str.contains('oldid=').all()


def test_roster_parser_reads_every_name_form():
    # Name cells come as {{sortname}} (any case), [[Target|Shown]] and plain text; a comment
    # after a heading must not hide the section; a release row carries date and reason.
    import roster_from_wikipedia as r
    page = '\n'.join([
        '===Bantamweights (135 lb, 61 kg)===<!-- note to editors -->',
        '{| class="wikitable"',
        '|-',
        '|{{#invoke:flag|icon|USA}}',
        "|{{Sortname|Sean|O'Malley|Sean O'Malley (fighter)}}",
        '|{{age nts|1994|10|24}}',
        '|',
        "|{{color|green|'''Win'''}} – [[UFC Freedom 250]] (Washington) – [[Aiemann Zahabi]]",
        '|<ref name="x"/>',
        '|{{ntsh|10.0}}11–3',
        '|{{ntsh|20.0}}20–3',
        '|-',
        '|{{#invoke:flag|icon|CHN}}',
        '|[[Song Yadong|Song Yadong]]',
        '|-',
        '|{{#invoke:flag|icon|CHN}}',
        '|Alateng Heili',
        '|}',
        '== Recent releases and retirements ==',
        '{| class="wikitable"',
        '|-',
        '|{{dts|2026|Oct|14}}',
        '|{{#invoke:flag|icon|AFG}}',
        '|{{sortname|Farid|Basharat}}',
        "|''Ferocious''",
        '|Contract not renewed',
        '|Bantamweight',
        '|}',
    ])
    p = r.parse_page(page).set_index('name')
    assert p.loc["Sean O'Malley", ['division', 'status', 'last_opponent', 'ufc_record']].tolist() == \
        ['BW', 'current', 'Aiemann Zahabi', '11-3']
    assert p.loc[['Song Yadong', 'Alateng Heili'], 'status'].tolist() == ['current', 'current']
    assert p.loc['Farid Basharat', ['status', 'status_date', 'reason', 'division']].tolist() == \
        ['released', '2026-10-14', 'Contract not renewed', 'BW']


@pytest.mark.parametrize('page_name, division, fighter_id', [
    # Word order, hyphens and spaces differ between the sites:
    ('Alateng Heili', 'BW', '1897b7b913736a7c'),          # UFCStats "Alatengheili"
    ('Lee Jeong-yeong', 'FW', 'd008d785f6347dc2'),
    ("Ode' Osbourne", 'FLW', '6d68c1afe954f121'),         # UFCStats "Ode Osbourne"
    # A middle name; the flyweight Bruno Silva, not the middleweight (12ebd7d157e91701):
    ('José Mauro Delano', 'FW', 'd7c4956d61ae76a9'),      # UFCStats "Jose Delano"
    ('Bruno Gustavo da Silva', 'FLW', '294aa73dbf37d281'),
    # A short first name, and a nickname via ROSTER_ALIASES:
    ('Cameron Rowston', 'MW', '8ebc3a8da015d70c'),        # UFCStats "Cam Rowston"
    ('Valesca Machado', 'WSW', 'a1a91dbf41a029c4'),       # UFCStats "Tina Black"
])
def test_roster_names_match_the_right_fighter(df, page_name, division, fighter_id):
    import roster_from_wikipedia as r
    page = pd.DataFrame([{'name': page_name, 'division': division, 'status': 'current', 'status_date': '',
                          'reason': '', 'last_opponent': '', 'ufc_record': '', 'suspended': False}])
    data = df.assign(event_date=pd.to_datetime(df['event_date']))
    assert r.match_names(page, data)['fighter_url'].iloc[0].endswith(fighter_id)

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
    ('Khabib Nurmagomedov', '032cc3922d871c7f', 1.933, 4.746, 'mostly a grappler, ground-and-pound-leaning'),
    ('Islam Makhachev', '275aca31f61ba28c', 1.233, -1.872, 'mostly a grappler, submission-leaning'),
    ('Jailton Almeida', '41e83a89929d1327', 2.874, 0.003, 'mostly a grappler, mixes submissions and ground strikes'),
    ('Israel Adesanya', '1338e2c7480bdf9e', -1.178, -0.127, 'mostly a striker'),
    ('Michael Page', 'a67d071163962af8', -1.063, -0.968, 'mostly a striker'),
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
    assert top3['shrunk_rate'].round(3).tolist() == [4.308, 4.246, 3.645]


# ---------------------------------------------------------------------------
# 9. Fighter style table (fighter_style_table.csv)
# ---------------------------------------------------------------------------
# The table is a stored copy of the functions' output on the CURRENT dataset,
# so these checks compare it with direct calls instead of pinning values.

@pytest.fixture(scope='module')
def style_table(df):
    return u.load_fighter_style_table(df)


def test_style_table_one_row_per_fighter(df, style_table):
    fighters = pd.concat([df['fighter_url_fighter_1'], df['fighter_url_fighter_2']]).unique()
    assert style_table['fighter_url'].is_unique
    assert set(style_table['fighter_url']) == set(fighters)


def test_style_table_refuses_other_dataset(df):
    with pytest.raises(ValueError, match='rebuild'):
        u.load_fighter_style_table(df.iloc[:-1])


# Islam Makhachev, Brendan Allen, Christian Leroy Duncan, plus 20 random fighters.
STYLE_TABLE_NAMED = ['275aca31f61ba28c', '2f181c0467965b98', 'a93f94c923c3a9cb']


def test_style_table_matches_direct_calls(df, style_table):
    sample = style_table[style_table['fighter_url'].map(url_id).isin(STYLE_TABLE_NAMED)]
    sample = pd.concat([sample, style_table.drop(sample.index).sample(20, random_state=0)])
    assert len(sample) == 23
    for _, row in sample.iterrows():
        arche = u.classify_fighter_archetype(df, row['fighter_url']).iloc[0]
        for col in ['grappling_orientation', 'ground_game_style', 'striking_style']:
            assert row[col] == pytest.approx(arche[col], abs=1e-9, nan_ok=True)
        assert (row['archetype_label'] if pd.notna(row['archetype_label']) else None) == arche['archetype_label']
        prof = u.generate_fighter_profile(df, row['fighter_url']).iloc[0]
        for col in u.STYLE_TABLE_PREFERENCE_COLUMNS:
            assert row[col] == pytest.approx(prof[col], abs=1e-9, nan_ok=True)
        vuln = u.assess_defensive_vulnerability(df, row['fighter_url'])
        for _, v in vuln.iterrows():
            assert row[f"{v['area']}_ratio"] == pytest.approx(v['vulnerability_ratio'], abs=1e-9, nan_ok=True)
            assert row[f"{v['area']}_z"] == pytest.approx(v['vulnerability_z'], abs=1e-9, nan_ok=True)
            got = row[f"{v['area']}_label"]
            assert (got if pd.notna(got) else None) == v['vulnerability_label']
