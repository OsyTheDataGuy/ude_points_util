"""
Keep ufc_roster.csv current from Wikipedia's "List of current UFC fighters".

UFCStats has no contract status, so "is this fighter still in the UFC" was a guess
from time since his last fight (FIGHTER_INACTIVE_AFTER_DAYS): it keeps a released
fighter (Farid Basharat, 7-0 in the UFC, contract not renewed 2026-10-14) and drops
an injured one who is still signed. This script runs in the weekly refresh:

  1. fetch_page()   -- the page's raw wikitext and revision id.
  2. parse_page()   -- one row per fighter: each division table ('current'), "Recent
                       signings" ('signed': signed or returning, no UFC fight since),
                       "Recent releases and retirements" ('released' / 'retired', with
                       date and reason). A name in the suspension table gets suspended.
  3. match_names()  -- page name -> fighter_url by full name (either word order) across
                       the dataset; a name two fighters share is split by division,
                       then by last opponent. Never by surname alone: a fighter who
                       hasn't debuted must not take someone else's id.
  4. run()          -- rewrites ufc_roster.csv only when a fighter's status changed,
                       and writes a report: who joined or left, names not matched.

Nothing here raises on a page problem: it becomes a warning in the report, and the
previous ufc_roster.csv is kept.

UFC.com shows the same status ("Active" / "Not Fighting") but refuses scripted
requests, so it is checked by hand for each fighter named in a post.

Run:  python roster_from_wikipedia.py --dataset current_df.csv \\
          --roster ufc_roster.csv --report roster_report.md
"""

import argparse
import json
import re
import urllib.request

import pandas as pd

from title_changes_from_wikipedia import _norm, _strip_refs

WIKI_API_URL = ('https://en.wikipedia.org/w/api.php?action=query&prop=revisions&titles=List_of_current_UFC_fighters'
                '&rvprop=ids|content&rvslots=main&format=json&formatversion=2')
REVISION_URL = 'https://en.wikipedia.org/w/index.php?title=List_of_current_UFC_fighters&oldid={}'

# Page division name (singular or plural, any case) -> weight_class_cleaned
DIVISIONS = {
    'heavyweight': 'HW', 'light heavyweight': 'LHW', 'middleweight': 'MW', 'welterweight': 'WW',
    'lightweight': 'LW', 'featherweight': 'FW', 'bantamweight': 'BW', 'flyweight': 'FLW',
    "women's bantamweight": 'WBW', "women's flyweight": 'WFLW', "women's strawweight": 'WSW',
    "women's featherweight": 'WFW',
}
STATUS_COLUMNS = ['fighter_url', 'name', 'division', 'status', 'status_date', 'reason', 'suspended']


"""## 1-2. Fetch and parse the page"""

def fetch_page(url=WIKI_API_URL):
    """(wikitext, revision id) of the page's current version."""
    req = urllib.request.Request(url, headers={'User-Agent': 'ude-points-refresh/1.0 (roster check)'})
    with urllib.request.urlopen(req, timeout=60) as resp:
        rev = json.loads(resp.read().decode('utf-8'))['query']['pages'][0]['revisions'][0]
    return rev['slots']['main']['content'], rev['revid']


def _division(text):
    """weight_class_cleaned for 'Bantamweights (135 lb, 61 kg)' or "Women's Flyweight", else None."""
    t = re.sub(r'\(.*', '', text).strip().lower()
    t = t[:-1] if t.endswith('s') else t
    return DIVISIONS.get(t)


def _display_name(cell):
    """Readable name from a name cell: {{sortname|First|Last|..}}, [[Target|Shown]], or plain text."""
    sort = re.search(r'\{\{\s*sortname\s*\|([^}]*)\}\}', cell, flags=re.I)
    if sort:
        parts = [p.strip() for p in sort.group(1).split('|') if '=' not in p]
        return ' '.join(parts[:2])
    link = re.search(r'\[\[([^\]|]+)(?:\|([^\]]+))?\]\]', cell)
    if link:
        return (link.group(2) or link.group(1)).strip()
    return re.sub(r"\{\{.*?\}\}|'''?|\*", '', cell).strip()


def _rows(table_text):
    """Cells of each row of a wikitable, refs and comments removed."""
    text = re.sub(r'<!--.*?-->', '', _strip_refs(table_text), flags=re.S)
    rows = []
    for chunk in re.split(r'\n\|-+', text)[1:]:
        cells = []
        for line in chunk.split('\n'):
            if line.startswith('|') and not line.startswith(('|}', '|-')):
                cells += [c.strip() for c in line[1:].split('||')]
        if cells:
            rows.append(cells)
    return rows


def _after_flag(cells):
    """Index of the cell after the flag cell (the name), or None."""
    for i, c in enumerate(cells):
        if 'flag' in c and '{{' in c:
            return i + 1 if i + 1 < len(cells) else None
    return None


def _record(cell):
    """'3-3' from a record cell like '{{ntsh|3.50}}3–3' or '9–11 (2 NC)', else ''."""
    m = re.search(r'\}\}\s*(\d+)\s*[–-]\s*(\d+)', cell)
    return f'{m.group(1)}-{m.group(2)}' if m else ''


def _sections(wikitext):
    """(heading, text) for each == / === section."""
    heads = list(re.finditer(r'^(==+)\s*(.+?)\s*\1\s*$', wikitext, flags=re.M))
    return [(h.group(2), wikitext[h.end(): heads[k + 1].start() if k + 1 < len(heads) else len(wikitext)])
            for k, h in enumerate(heads)]


def _last_opponent(cell):
    """Opponent named at the end of a 'Result – [[Event]] (City) – Opponent' cell, or ''."""
    if ' – ' not in cell:
        return ''
    return _display_name(cell.rsplit(' – ', 1)[1])


def parse_page(wikitext):
    """DataFrame: name, division, status, status_date, reason, last_opponent, suspended."""
    out, suspended = [], set()
    wikitext = re.sub(r'<!--.*?-->', '', wikitext, flags=re.S)     # comments sit after some headings
    for heading, text in _sections(wikitext):
        wc = _division(heading)
        if wc:                                                   # a division's roster table
            for cells in _rows(text):
                i = _after_flag(cells)
                if i is None:
                    continue
                result = next((c for c in cells[i + 1:] if ' – ' in c), '')
                out.append({'name': _display_name(cells[i]), 'division': wc, 'status': 'current',
                            'status_date': '', 'reason': '', 'last_opponent': _last_opponent(result),
                            'ufc_record': _record(next((c for c in cells if 'ntsh' in c), ''))})
        elif heading == 'Recent signings':
            for cells in _rows(text):
                i = _after_flag(cells)
                if i is not None and i + 2 < len(cells):
                    out.append({'name': _display_name(cells[i]), 'division': _division(cells[i + 2]) or '',
                                'status': 'signed', 'status_date': '', 'reason': '', 'last_opponent': '', 'ufc_record': ''})
        elif heading == 'Recent releases and retirements':
            for cells in _rows(text):
                i = _after_flag(cells)
                if i is None or i + 3 >= len(cells):
                    continue
                d = re.search(r'dts\|(\d{4})\|(\w+)\|(\d{1,2})', cells[0])
                reason = re.sub(r"'''?", '', cells[i + 2]).strip()
                out.append({'name': _display_name(cells[i]), 'division': _division(cells[i + 3]) or '',
                            'status': 'retired' if 'retire' in reason.lower() else 'released',
                            'status_date': pd.to_datetime(' '.join(d.groups())).date().isoformat() if d else '',
                            'reason': reason, 'last_opponent': '', 'ufc_record': ''})
        elif heading == 'Suspended fighters':
            for cells in _rows(text):
                i = _after_flag(cells)
                if i is not None:
                    suspended.add(_norm(_display_name(cells[i])))
    page = pd.DataFrame(out)
    page['suspended'] = page['name'].map(_norm).isin(suspended)
    # A fighter listed both as released and on a roster table (re-signed) is current.
    rank = page['status'].map({'current': 0, 'signed': 1, 'released': 2, 'retired': 2})
    page = page.assign(_r=rank, _n=page['name'].map(_norm)).sort_values('_r')
    return page.drop_duplicates(['_n', 'division']).drop(columns=['_r', '_n']).reset_index(drop=True)


"""## 3. Names -> fighter_url"""

# Page name -> UFCStats fighter id, where the two sites spell a name differently
# beyond what match_names handles. Each checked by its last fight and UFC record.
ROSTER_ALIASES = {
    'Bernardo Sopaj': 'b6c37948cb226e8c',       # UFCStats "Benardo Sopaj"
    'Mehemmedeli Osmanli': '2bbdbc925f4323ac',  # UFCStats "Mahammadali Osmanli"
    'Beatriz Mesquita': 'eff9074bc066f0f2',     # UFCStats "Bia Mesquita"
    'Gabriel Green': '5eb50529a8418b77',         # UFCStats "Gabe Green" (UFC 3-3 on both)
    'Darya Zheleznyakova': '07047eb7d17fb0a2',   # UFCStats "Daria Zhelezniakova" (UFC 2-3 on both)
    'Eduardo Henrique': '62f3039da2f92a8c',      # UFCStats "Eduardo Chapolin" (0-1, lost to Charles Johnson)
    'Valesca Machado': 'a1a91dbf41a029c4',       # UFCStats "Tina Black", her nickname (beat Melissa Amaya, 7bff1b7a45666e98)
}


def _words(name):
    """The name's words (hyphens split them), for the middle-name rule."""
    return set(_norm(str(name).replace("'", '')).split())


def _keys(name):
    """Name keys that ignore word order, hyphens, apostrophes and spaces:
    'Lee Jeong-yeong' fits 'Jeongyeong Lee', 'Alateng Heili' fits 'Alatengheili'."""
    name = str(name).replace("'", '')
    keys = {' '.join(sorted(_norm(n).split())) for n in (name, name.replace('-', ''))}
    return keys | {_norm(name).replace(' ', '')}


def _fighters(df):
    """One row per fighter: url, name, latest division, latest opponent, name keys, UFC W and L."""
    sides = [df[['event_date', f'fighter_{s}', f'fighter_url_fighter_{s}', f'fighter_{o}', 'weight_class_cleaned',
                 f'fight_result_fighter_{s}']]
             .set_axis(['date', 'name', 'url', 'opp', 'wc', 'res'], axis=1) for s, o in (('1', '2'), ('2', '1'))]
    all_fights = pd.concat(sides)
    wins = all_fights[all_fights['res'] == 'W'].groupby('url').size()
    losses = all_fights[all_fights['res'] == 'L'].groupby('url').size()
    f = all_fights.sort_values('date').groupby('url').tail(1).set_index('url', drop=False)
    return f.assign(keys=f['name'].map(_keys), words=f['name'].map(_words), opp_key=f['opp'].map(_norm),
                    ufc_w=wins.reindex(f.index).fillna(0).astype(int),
                    ufc_l=losses.reindex(f.index).fillna(0).astype(int))


def _same_person(page_words, data_words, data_name):
    """Within one division: one name's words all inside the other's (a middle name or a
    second surname: 'Jose Delano' / 'José Mauro Delano'), at least 2 words unless the
    UFCStats name is one word ('Mizuki' / 'Mizuki Inoue'); or the same surname with one
    first name starting the other ('Cam' / 'Cameron Rowston')."""
    short, long_ = sorted([page_words, data_words], key=len)
    if short <= long_ and (len(short) >= 2 or len(str(data_name).split()) == 1):
        return True
    return False


def _short_first_name(page_name, data_name):
    p, d = _norm(page_name.replace("'", '')).split(), _norm(str(data_name).replace("'", '')).split()
    return len(p) == 2 and len(d) == 2 and p[1] == d[1] and len(min(p[0], d[0], key=len)) >= 3 \
        and (p[0].startswith(d[0]) or d[0].startswith(p[0]))


def match_names(page, df):
    """page with fighter_url filled where exactly one fighter fits ('' otherwise) and a match note.

    In order: ROSTER_ALIASES; full name (any word order, hyphens, apostrophes, spaces), a name
    two fighters share split by division, then by the opponent the page lists; then, within
    the page's division only, a middle name or second surname, a short first name, or the
    opponent the page lists plus a shared word of the name. Never a surname alone."""
    fighters = _fighters(df)
    by_id = {u.rsplit('/', 1)[-1]: u for u in fighters['url']}
    urls, notes = [], []
    for r in page.itertuples():
        if r.name in ROSTER_ALIASES:
            urls.append(by_id.get(ROSTER_ALIASES[r.name], ''))
            notes.append('ROSTER_ALIASES')
            continue
        keys, words = _keys(r.name), _words(r.name)
        cand = fighters[fighters['keys'].map(lambda k: bool(k & keys))]
        note = ''
        if len(cand) > 1 and r.division and (cand['wc'] == r.division).any():
            cand = cand[cand['wc'] == r.division]
        if len(cand) > 1 and r.last_opponent and (cand['opp_key'] == _norm(r.last_opponent)).any():
            cand = cand[cand['opp_key'] == _norm(r.last_opponent)]
        if cand.empty and r.division:
            pool = fighters[fighters['wc'] == r.division]
            cand = pool[[_same_person(words, w, n) or _short_first_name(r.name, n)
                         for w, n in zip(pool['words'], pool['name'])]]
            note = 'matched by name within division'
            if cand.empty and r.last_opponent:
                cand = pool[(pool['opp_key'] == _norm(r.last_opponent)) & pool['words'].map(lambda w: bool(w & words))]
                note = 'matched by last opponent'
        if len(cand) == 1:
            urls.append(cand['url'].iloc[0])
            notes.append(note)
        else:
            urls.append('')
            notes.append('no UFC fight in the dataset' if cand.empty else f'{len(cand)} fighters share this name')
    return page.assign(fighter_url=urls, match_note=notes)


def record_mismatches(roster, df):
    """Fighters matched by anything looser than their full name whose UFC W-L on the page
    differs from the dataset's -- possibly the wrong fighter. (name, id, page record, ours.)
    Full-name matches aren't checked: Wikipedia's count differs from UFCStats' for ~4% of
    the roster (e.g. Belal Muhammad 15-5 vs 15-6), the right fighter either way."""
    f = _fighters(df)
    loose = roster[(roster['fighter_url'] != '') & (roster['ufc_record'] != '') & (roster['match_note'] != '')]
    out = []
    for r in loose.itertuples():
        w, l = (int(x) for x in r.ufc_record.split('-')[:2])
        if (w, l) != (f.at[r.fighter_url, 'ufc_w'], f.at[r.fighter_url, 'ufc_l']):
            out.append((r.name, r.fighter_url.rsplit('/', 1)[-1], r.ufc_record,
                        f'{f.at[r.fighter_url, "ufc_w"]}-{f.at[r.fighter_url, "ufc_l"]}'))
    return out


"""## 4. Write the file and the report"""

def _status_rows(roster):
    return set(roster[STATUS_COLUMNS].fillna('').astype(str).itertuples(index=False, name=None))


def run(dataset_csv, roster_csv, report_path):
    lines = ['', '## UFC roster (Wikipedia check)', '']
    try:
        df = pd.read_csv(dataset_csv, low_memory=False)
        df['event_date'] = pd.to_datetime(df['event_date'])
        wikitext, revid = fetch_page()
        page = parse_page(wikitext)
        if (page['status'] == 'current').sum() < 300:
            raise ValueError(f'only {(page["status"] == "current").sum()} current fighters parsed -- '
                             'the page layout may have changed')
        roster = match_names(page, df).assign(source_revision_url=REVISION_URL.format(revid))
        try:
            old = pd.read_csv(roster_csv, keep_default_na=False, dtype=str)
        except FileNotFoundError:
            old = pd.DataFrame(columns=STATUS_COLUMNS + ['source_revision_url'])
        counts = roster['status'].value_counts()
        lines.append(f'Page version checked: {REVISION_URL.format(revid)}')
        lines.append(f'{counts.get("current", 0)} on division rosters, {counts.get("signed", 0)} signed or '
                     f'returning, {counts.get("released", 0)} released and {counts.get("retired", 0)} retired '
                     f'in the last month.')
        if _status_rows(roster) == _status_rows(old):
            lines.append('No roster changes since the committed ufc_roster.csv.')
        else:
            (roster[STATUS_COLUMNS + ['source_revision_url']].sort_values(['status', 'division', 'name'])
             .to_csv(roster_csv, index=False))
            on = lambda r: set(r.loc[r['status'].isin(['current', 'signed']) & (r['fighter_url'] != ''), 'fighter_url'])
            left, joined = on(old) - on(roster), on(roster) - on(old)
            names = dict(zip(roster['fighter_url'], roster['name'])) | dict(zip(old['fighter_url'], old['name']))
            reasons = dict(zip(roster['fighter_url'], roster['status'] + ' ' + roster['status_date'] + ' '
                               + roster['reason']))
            lines.append(f'**`{roster_csv}` updated -- {len(left)} left the roster, {len(joined)} joined:**')
            lines += [f'- left: {names[u]} ({u.rsplit("/", 1)[-1]}) -- {reasons.get(u, "no longer listed").strip()}'
                      for u in sorted(left)]
            lines += [f'- joined: {names[u]} ({u.rsplit("/", 1)[-1]})' for u in sorted(joined)]
        unmatched = roster[(roster['fighter_url'] == '') & (roster['match_note'] != 'no UFC fight in the dataset')]
        if len(unmatched):
            lines.append('')
            lines.append('**Warning -- page names that fit more than one fighter (left unmatched):**')
            lines += [f'- {r.name} ({r.division}, {r.status}): {r.match_note}' for r in unmatched.itertuples()]
        wrong = record_mismatches(roster, df)
        if wrong:
            lines.append('')
            lines.append('**Warning -- fighters matched by a looser name rule whose UFC record differs from the page '
                         '(check it is the same fighter; fix with ROSTER_ALIASES):**')
            lines += [f'- {n} ({u}): page {p}, dataset {o}' for n, u, p, o in wrong]
        not_debuted = (roster['match_note'] == 'no UFC fight in the dataset').sum()
        lines.append(f'{not_debuted} listed names have no UFC fight in the dataset (mostly not yet debuted); '
                     'kept with a blank fighter_url.')
    except Exception as exc:  # never block the data refresh on the page
        lines.append(f'**Warning -- roster check did not run; ufc_roster.csv unchanged:** {type(exc).__name__}: {exc}')
    text = '\n'.join(lines) + '\n'
    with open(report_path, 'w') as f:
        f.write(text)
    print(text)


def _parse_args():
    parser = argparse.ArgumentParser(description='Write the current UFC roster from Wikipedia.')
    parser.add_argument('--dataset', required=True, help='current_df.csv')
    parser.add_argument('--roster', required=True, help='ufc_roster.csv (rewritten when a status changes)')
    parser.add_argument('--report', required=True, help='Markdown report for the PR body')
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    run(args.dataset, args.roster, args.report)
