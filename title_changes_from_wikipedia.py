"""
Keep title_changes_outside_cage.csv current from Wikipedia's "List of UFC champions".

UFCStats records no strips, vacated belts, promotions or WEC/Strikeforce transfers,
so update_champion_status reads them from title_changes_outside_cage.csv. This
script runs in the weekly refresh, before the pipeline:

  1. fetch_wikitext()      -- the page's raw wikitext.
  2. parse_page()          -- every undisputed/interim reign (with its listed
                              defenses) and every note row, per division.
  3. events_from_page()    -- notes and "promoted" reigns turned into events:
                              stripped, vacated, interim_stripped, promoted.
  4. match_fighter()       -- Wikipedia name -> fighter_url, within the division.
  5. new_events()          -- events not already in the file. Existing rows are
                              never changed: a later Wikipedia edit can't rewrite
                              history, and hand corrections (Oliveira's weigh-in
                              date) stay.
  6. check_against_page()  -- agreement check: our title_defenses / is_champion
                              calls vs the page's defense lists and reign order.

Nothing here raises on a page problem. A fetch failure, a note it can't read or
a name it can't match becomes a warning in the report, which the refresh adds to
the PR body; the data refresh itself goes ahead.

Run:  python title_changes_from_wikipedia.py --dataset current_df.csv \\
          --events title_changes_outside_cage.csv --report title_changes_report.md
"""

import argparse
import re
import unicodedata
import urllib.request
from datetime import date

import pandas as pd

WIKI_RAW_URL = 'https://en.wikipedia.org/w/index.php?title=List_of_UFC_champions&action=raw'
SOURCE = 'Wikipedia: List of UFC champions'

# Section heading -> weight_class_cleaned
SECTIONS = {
    'Heavyweight Championship': 'HW', 'Light Heavyweight Championship': 'LHW',
    'Middleweight Championship': 'MW', 'Welterweight Championship': 'WW',
    'Lightweight Championship': 'LW', 'Featherweight Championship': 'FW',
    'Bantamweight Championship': 'BW', 'Flyweight Championship': 'FLW',
    "Women's Bantamweight Championship": 'WBW', "Women's Flyweight Championship": 'WFLW',
    "Women's Strawweight Championship": 'WSW', "Women's Featherweight Championship": 'WFW',
}
DATE = r'([A-Z][a-z]{2,8}\.? \d{1,2}, \d{4})'

# Wikipedia name -> UFCStats name, where the two differ beyond spelling.
ALIASES = {'Cris Cyborg': 'Cristiane Justino'}

# Two events of the same fighter, division and type within this many days are
# the same event (the file's hand-set date may differ from the page's).
SAME_EVENT_DAYS = 31


"""## 1-2. Fetch and parse the page"""

def fetch_wikitext(url=WIKI_RAW_URL):
    req = urllib.request.Request(url, headers={'User-Agent': 'ude-points-refresh/1.0 (title-changes check)'})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read().decode('utf-8')


def _link_name(s):
    """Display name of the first [[link]] in s (the part after | if there is one)."""
    m = re.search(r'\[\[([^\]|]+)(?:\|([^\]]+))?\]\]', s)
    return (m.group(2) or m.group(1)).strip() if m else None


def _to_date(s):
    return pd.to_datetime(re.sub(r'\bSept\b', 'Sep', s).replace('.', ''), format='mixed')


def parse_page(wikitext):
    """Reigns, their listed defenses, and note rows, in page order per division."""
    lines = wikitext.split('\n')
    reigns, defenses, notes = [], [], []
    wc, i = None, 0
    while i < len(lines):
        line = lines[i]
        heading = re.match(r'^===\s*(.+?)\s*===\s*$', line)
        if heading:
            wc = SECTIONS.get(heading.group(1).strip())
        elif line.startswith('==') and not line.startswith('===') and 'championship history' not in line \
                and 'Defunct' not in line:
            wc = None
        elif wc and re.match(r'^!\s*colspan', line):
            text = line.split("''", 1)[1] if "''" in line else line
            notes.append({'pos': i, 'wc': wc, 'text': re.sub(r'<ref.*', '', text).strip("' ")})
        elif wc and re.match(r'^!\s*(\d+|—)\s*$', line):
            block, j = [], i + 1
            while j < len(lines) and not lines[j].startswith(('|-', '|}', '==')):
                block.append(lines[j])
                j += 1
            text = '\n'.join(block)
            start = re.search(r'(?:align=center|text-align:center;")\|\s*' + DATE, text)
            name = _link_name(block[0]) if block else None
            interim = '—' in line
            reigns.append({'pos': i, 'wc': wc, 'fighter': name, 'interim': interim,
                           'promoted': bool(block) and 'promoted to undisputed' in block[0],
                           'start': _to_date(start.group(1)) if start else None})
            for m in re.finditer(r'def\. (.+?) at (?:\[\[.+?\]\]|[^<]+?) on ' + DATE, text):
                defenses.append({'wc': wc, 'champion': name, 'opponent': _link_name(m.group(1)) or m.group(1).strip(),
                                 'date': _to_date(m.group(2)), 'interim_reign': interim})
            i = j
            continue
        i += 1
    return pd.DataFrame(reigns), pd.DataFrame(defenses), pd.DataFrame(notes)


"""## 3. Notes and promotions -> events"""

def _note_event(text):
    """(fighter name, event, date) from one note row, or None with a reason."""
    if 'fought to a draw' in text:
        return None, 'draw note (no title change)'
    if text.startswith('Both '):
        return None, 'note names two fighters -- add these rows by hand'
    lead = re.match(r'^(.+?) (?:was|vacated|announced|won)\b', text)
    name = _link_name(lead.group(1)) or lead.group(1) if lead else None
    official = re.search(r'officially vacated on ' + DATE, text)
    if 'stripped of the interim title' in text:
        event = 'interim_stripped'
    elif 'stripped' in text:
        event = 'stripped'
    elif 'vacat' in text or 'retirement' in text:
        event = 'vacated'
    else:
        return None, 'note type not recognised'
    dated = official or re.search(r'stripped on ' + DATE, text) or re.search(r'on ' + DATE, text)
    if not dated:
        return None, 'note has no full date'
    return (name, event, _to_date(dated.group(1))), None


def events_from_page(reigns, notes):
    """Events from note rows and 'promoted' reigns, plus a list of notes it skipped."""
    events, skipped = [], []
    for n in notes.itertuples():
        found, reason = _note_event(n.text)
        if found:
            # Notes give a surname ("Jones was stripped"): take the full name from the
            # latest reign above the note, in the same division, with that surname.
            above = reigns[(reigns.wc == n.wc) & (reigns.pos < n.pos)]
            # (any word of the reign name, so "Zhang" finds "Zhang Weili").
            same = above[above.fighter.map(lambda f: _norm(found[0]).split()[-1] in _norm(f).split())]
            name = same.fighter.iloc[-1] if len(same) else found[0]
            events.append({'date': found[2], 'weight_class_cleaned': n.wc, 'fighter': name, 'event': found[1]})
        else:
            skipped.append((n.wc, reason, n.text[:120]))
    for r in reigns[reigns.promoted].itertuples():
        events.append({'date': r.start, 'weight_class_cleaned': r.wc, 'fighter': r.fighter, 'event': 'promoted'})
    return pd.DataFrame(events), skipped


"""## 4. Wikipedia name -> fighter_url"""

def _norm(name):
    name = str(name).replace('ł', 'l').replace('Ł', 'L')
    name = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode().lower()
    return ' '.join(name.replace('.', '').replace('-', ' ').split())


def _fighters_by_division(df):
    sides = [df[[f'fighter_{s}', f'fighter_url_fighter_{s}', 'weight_class_cleaned']]
             .set_axis(['name', 'url', 'wc'], axis=1) for s in ('1', '2')]
    return pd.concat(sides).drop_duplicates()


def match_fighter(name, wc, roster):
    """fighter_url for a Wikipedia name within one division: full name first, then a unique surname."""
    if not name:
        return None
    name = ALIASES.get(name, name)
    pool = roster[roster.wc == wc]
    full = pool[pool.name.map(_norm) == _norm(name)].url.unique()
    if len(full) == 1:
        return full[0]
    surname = _norm(name).split()[-1]
    last = pool[pool.name.map(lambda n: _norm(n).split()[-1]) == surname].url.unique()
    return last[0] if len(last) == 1 else None


"""## 5. New events only"""

def new_events(page_events, known, roster):
    """Page events not already in the file (same fighter, division, type within SAME_EVENT_DAYS)."""
    added, unmatched = [], []
    for e in page_events.itertuples():
        url = match_fighter(e.fighter, e.weight_class_cleaned, roster)
        if url is None:
            unmatched.append((e.date.date(), e.weight_class_cleaned, e.fighter, e.event))
            continue
        same = known[(known.fighter_url == url) & (known.weight_class_cleaned == e.weight_class_cleaned)
                     & (known.event == e.event) & ((known.date - e.date).abs() <= pd.Timedelta(days=SAME_EVENT_DAYS))]
        if len(same) == 0:
            name = roster.loc[roster.url == url, 'name'].iloc[0]
            added.append({'date': e.date, 'weight_class_cleaned': e.weight_class_cleaned, 'fighter': name,
                          'fighter_url': url, 'event': e.event,
                          'note': f'Added from Wikipedia on {date.today().isoformat()}; check before merging',
                          'source': SOURCE})
    return pd.DataFrame(added), unmatched


"""## 6. Agreement check"""

def check_against_page(df, reigns, defenses, notes, roster):
    """Counts of our defense calls and belt-change loser flags that disagree with the page."""
    df = df.assign(event_date=pd.to_datetime(df['event_date']))
    title = df[df['is_title_bout'] == 2]
    wins = []
    for s, o in (('1', '2'), ('2', '1')):
        w = title[title[f'fight_result_fighter_{s}'] == 'W']
        wins.append(pd.DataFrame({'fight_url': w['fight_url'], 'date': w['event_date'], 'wc': w['weight_class_cleaned'],
                                  'url': w[f'fighter_url_fighter_{s}'], 'loser_url': w[f'fighter_url_fighter_{o}'],
                                  'champ': w[f'is_champion_fighter_{s}'], 'loser_flag': w[f'is_champion_fighter_{o}']}))
    wins = pd.concat(wins)
    near = lambda d: (wins['date'] - d).abs() <= pd.Timedelta(days=1)

    # Defenses: a win counts for us when the winner enters flagged 2.
    listed = set()
    for d in defenses[~defenses.interim_reign].itertuples():
        url = match_fighter(d.champion, d.wc, roster)
        hit = wins[near(d.date) & (wins['url'] == url)]
        if len(hit):
            listed.add(hit.index[0])
    defense_disagree = wins[(wins['champ'] == 2) != wins.index.isin(listed)]

    # Belt changes: the loser held the belt iff the previous undisputed item on the page is a reign, not a note.
    items = pd.concat([reigns.assign(kind='reign'), notes.assign(kind='note')]).sort_values('pos')
    flag_disagree, n_changes = [], 0
    for wc, rows in items.groupby('wc', sort=False):
        prev = None
        for r in rows.itertuples():
            if r.kind == 'note':
                if 'interim title' not in r.text:   # an interim strip leaves the undisputed belt held
                    prev = r
                continue
            if r.interim:
                continue
            if not r.promoted and r.start is not None and not pd.isna(r.start):
                url = match_fighter(r.fighter, wc, roster)
                for x in wins[near(r.start) & (wins['url'] == url)].itertuples():
                    n_changes += 1
                    held = prev is not None and prev.kind == 'reign' and \
                        match_fighter(prev.fighter, wc, roster) == x.loser_url
                    if (x.loser_flag == 2) != held:
                        flag_disagree.append(x.fight_url)
            prev = r
    return {'title_wins': len(wins), 'defense_disagree': list(defense_disagree['fight_url']),
            'belt_changes': n_changes, 'flag_disagree': flag_disagree}


"""## Report"""

def run(dataset_csv, events_csv, report_path):
    lines = ['', '## Title changes outside the cage (Wikipedia check)', '']
    try:
        df = pd.read_csv(dataset_csv, low_memory=False)
        known = pd.read_csv(events_csv, parse_dates=['date'], keep_default_na=False)
        roster = _fighters_by_division(df)
        reigns, defenses, notes = parse_page(fetch_wikitext())
        if reigns.empty or notes.empty:
            raise ValueError('page parsed to no reigns or notes -- its layout may have changed')
        page_events, skipped = events_from_page(reigns, notes)
        added, unmatched = new_events(page_events, known, roster)

        if len(added):
            out = pd.concat([known, added], ignore_index=True).sort_values(['date', 'weight_class_cleaned', 'event'])
            out['date'] = pd.to_datetime(out['date']).dt.strftime('%Y-%m-%d')
            out.to_csv(events_csv, index=False)
            lines.append(f'**{len(added)} new event(s) added to `{events_csv}` -- check each before merging:**')
            lines += [f'- {r.date.date()} {r.weight_class_cleaned} {r.fighter} ({r.fighter_url.rsplit("/", 1)[-1]}): {r.event}'
                      for r in added.itertuples()]
        else:
            lines.append('No new title changes on the page.')
        if unmatched:
            lines.append('')
            lines.append('**Warning -- page events whose fighter could not be matched (add by hand if real):**')
            lines += [f'- {d} {wc} {name}: {ev}' for d, wc, name, ev in unmatched]
        # Skip notes already covered by a hand-kept row (a fighter of that division named in the note).
        covered = lambda wc, text: any(_norm(f).split()[-1] in _norm(text).split()
                                       for f in known.loc[known.weight_class_cleaned == wc, 'fighter'])
        notes_unread = [s for s in skipped if not s[1].startswith('draw') and not covered(s[0], s[2])]
        if notes_unread:
            lines.append('')
            lines.append(f'{len(notes_unread)} note(s) not read automatically (already covered by hand-kept rows unless new):')
            lines += [f'- {wc}: {reason} -- "{text}"' for wc, reason, text in notes_unread]

        check = check_against_page(df, reigns, defenses, notes, roster)
        lines += ['', f'Agreement check on the dataset before this refresh: '
                      f'{check["title_wins"] - len(check["defense_disagree"])} of {check["title_wins"]} undisputed '
                      f'title-bout wins counted as on the page; {len(check["flag_disagree"])} of {check["belt_changes"]} '
                      f'belt changes with a different loser flag.']
        if check['defense_disagree'] or check['flag_disagree']:
            lines.append('Fights that differ (known: Miletich df8d4014d8206ab1 and Shamrock b1134affa18449dd, '
                         'whose title wins UFCStats does not have): ' +
                         ', '.join(u.rsplit('/', 1)[-1] for u in check['defense_disagree'] + check['flag_disagree']))
    except Exception as exc:  # never block the data refresh on the page
        lines.append(f'**Warning -- Wikipedia check did not run:** {type(exc).__name__}: {exc}')
    text = '\n'.join(lines) + '\n'
    with open(report_path, 'w') as f:
        f.write(text)
    print(text)


def _parse_args():
    parser = argparse.ArgumentParser(description='Add new off-cage title changes from Wikipedia and report agreement.')
    parser.add_argument('--dataset', required=True, help='current_df.csv (the committed production file)')
    parser.add_argument('--events', required=True, help='title_changes_outside_cage.csv (appended in place)')
    parser.add_argument('--report', required=True, help='Markdown report for the PR body')
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    run(args.dataset, args.events, args.report)
