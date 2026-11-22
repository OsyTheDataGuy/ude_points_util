"""Build raw, fighter-aligned round rows alongside the validated fight dataset."""

import argparse
import csv
import os
import re
from collections import defaultdict
from datetime import date
from pathlib import Path
from tempfile import NamedTemporaryFile


COUNT_PAIRS = {
    "SIG.STR.": "sig_strikes", "TOTAL STR.": "total_strikes", "TD": "td",
    "HEAD": "head_strikes", "BODY": "body_strikes", "LEG": "leg_strikes",
    "DISTANCE": "distance_strikes", "CLINCH": "clinch_strikes", "GROUND": "ground_strikes",
}
SINGLES = {"KD": "kd", "SUB.ATT": "sub_att", "REV.": "rev", "CTRL": "ctrl_in_secs"}
METRICS = [f"{base}_{kind}" for base in COUNT_PAIRS.values() for kind in ("landed", "attempted")]
METRICS += list(SINGLES.values())
STATS = [f"{metric}_fighter_{side}" for side in (1, 2) for metric in METRICS]
META = ["event_date", "event_name", "fight_url", "fighter_1", "fighter_url_fighter_1",
        "fighter_2", "fighter_url_fighter_2"]
COLUMNS = ["fight_url", "round_number", *[col for col in META if col != "fight_url"], *STATS]
RAW_COLUMNS = ["fight_url", "ROUND", "FIGHTER", "fighter_url", *COUNT_PAIRS, *SINGLES]
MISSING = {"", "--", "---"}


def source_rounds(soup, url):
    """Read only round-labelled tables. Join totals and strike splits by round and fighter URL."""
    result = {}
    for table in soup.select("table"):
        header = table.find("tr")
        if header is None:
            continue
        labels = [cell.get_text(strip=True).upper() for cell in header.select("th")]
        if "KD" in labels:
            fields = ("FIGHTER", "KD", "SIG.STR.", None, "TOTAL STR.", "TD", None, "SUB.ATT", "REV.", "CTRL")
        elif "GROUND" in labels:
            fields = ("FIGHTER", "SIG.STR.", None, "HEAD", "BODY", "LEG", "DISTANCE", "CLINCH", "GROUND")
        else:
            continue
        number = None
        for tr in table.select("tr"):
            heading = re.fullmatch(r"Round\s+(\d+)", tr.get_text(" ", strip=True), re.I)
            if heading:
                number = heading.group(1)
                continue
            cells = tr.find_all("td", recursive=False)
            if number is None or not cells:
                continue
            links = cells[0].select("a[href]")
            if len(links) != 2 or len(cells) != len(fields):
                raise ValueError(f"unexpected table layout in round {number}")
            for side, link in enumerate(links):
                key = number, link["href"]
                row = result.setdefault(key, {"fight_url": url, "ROUND": number,
                                              "FIGHTER": link.get_text(strip=True), "fighter_url": link["href"]})
                for field, cell in zip(fields[1:], cells[1:]):
                    if field is None:
                        continue
                    values = [p.get_text(strip=True) for p in cell.find_all("p")]
                    if len(values) != 2:
                        raise ValueError(f"missing fighter cell in round {number}")
                    if field in row and (field != "SIG.STR." or row[field] != values[side]):
                        raise ValueError(f"duplicate or inconsistent {field} in round {number}")
                    row[field] = values[side]
            number = None
    if not result or any(row.keys() != set(RAW_COLUMNS) for row in result.values()):
        raise ValueError("incomplete per-round tables")
    return list(result.values())


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        yield from csv.DictReader(stream)


def count(value):
    if not re.fullmatch(r"\d+(?:\.0+)?", value):
        raise ValueError(f"invalid count {value!r}")
    return int(value.split(".")[0])


def metrics(row):
    result = {}
    for field, base in COUNT_PAIRS.items():
        value = row[field].strip()
        landed = attempted = None
        if value not in MISSING:
            parts = value.split(" of ")
            if len(parts) != 2:
                raise ValueError(f"invalid {field} {value!r}")
            landed, attempted = map(count, parts)
            if landed > attempted:
                raise ValueError(f"{field} landed exceeds attempted")
        result[f"{base}_landed"] = landed
        result[f"{base}_attempted"] = attempted
    for field, name in SINGLES.items():
        value = row[field].strip()
        if value in MISSING:
            result[name] = None
        elif field == "CTRL":
            if not re.fullmatch(r"\d+:[0-5]\d", value):
                raise ValueError(f"invalid clock {value!r}")
            minutes, seconds = map(int, value.split(":"))
            result[name] = minutes * 60 + seconds
        else:
            result[name] = count(value)
    return result


def id_from_url(url):
    return url.rstrip("/").rsplit("/", 1)[-1]


def align(fight, rows):
    by_number = {}
    seen = set()
    ids = [id_from_url(fight[f"fighter_url_fighter_{side}"]) for side in (1, 2)]
    names = [fight[f"fighter_{side}"].strip().casefold() for side in (1, 2)]
    for raw in rows:
        number = count(raw["ROUND"].strip().removeprefix("Round "))
        if raw.get("fighter_url"):
            side = ids.index(id_from_url(raw["fighter_url"])) + 1
        else:
            name = raw["FIGHTER"].strip().casefold()
            if names.count(name) != 1:
                raise ValueError(f"unknown or ambiguous fighter {raw['FIGHTER']}")
            side = names.index(name) + 1
        if (number, side) in seen:
            raise ValueError(f"duplicate fighter in round {number}")
        seen.add((number, side))
        row = by_number.setdefault(number, {col: fight[col] for col in META})
        row["round_number"] = number
        row.update({f"{key}_fighter_{side}": value for key, value in metrics(raw).items()})
    return [by_number[n] for n in sorted(by_number)]


def validate(fight, rounds, *, archive=False):
    numbers = [int(row["round_number"]) for row in rounds]
    if numbers != list(range(1, count(fight["round_ended"]) + 1)):
        raise ValueError("missing or duplicate round")
    for row in rounds:
        if any(row.get(col) != fight[col] for col in META) or not set(STATS).issubset(row):
            raise ValueError("fighter metadata or round fields missing or inconsistent")
    clock_gaps = {}
    for col in STATS:
        values = [row[col] for row in rounds]
        if fight[col] and all(value not in (None, "") for value in values):
            difference = sum(map(int, values)) - count(fight[col])
            if difference and col.startswith("ctrl_in_secs_"):
                clock_gaps[col] = difference
                # Prefer the primary page if an archive clock differs substantially.
                if archive and abs(difference) > len(rounds):
                    raise ValueError(f"{col} needs direct round-table verification")
            elif difference:
                raise ValueError(f"{col} disagrees with fight total")
    return clock_gaps


def archive_groups(details, stats, pending):
    urls = defaultdict(set)
    for row in read_csv(details):
        urls[row["EVENT"].strip(), row["BOUT"].strip()].add(id_from_url(row["URL"]))
    grouped = defaultdict(list)
    for row in read_csv(stats):
        matches = urls.get((row["EVENT"].strip(), row["BOUT"].strip()), set())
        if len(matches) == 1:
            url = next(iter(matches))
            if url in pending:
                grouped[url].append(row)
    return grouped


def retry_pages(fights, pending, limit):
    """Retry a bounded slice of missing fight URLs, rotating on weekly runs."""
    import ude_scrape_new as scraper

    keys = sorted(pending)
    start = (date.today().toordinal() // 7 * limit) % len(keys)
    chosen = (keys[start:] + keys[:start])[:limit]
    result = {}
    try:
        try:
            scraper.start_browser()
        except Exception as exc:
            print(f"Round-page browser unavailable; missing fights remain pending: {exc}")
            return result
        for key in chosen:
            url = fights[key]["fight_url"]
            try:
                result[key] = source_rounds(scraper.fetch_page_content(url), url)
            except Exception as exc:
                # The shared browser fetcher can raise browser-specific navigation errors.
                print(f"Could not retrieve round data for {url}: {exc}")
    finally:
        scraper.stop_browser()
    return result


def update(dataset, output, live=None, details=None, archive=None, report=None, retry_limit=0):
    fights = {id_from_url(row["fight_url"]): row for row in read_csv(dataset)}
    grouped = defaultdict(list)
    if output.exists():
        for row in read_csv(output):
            grouped[id_from_url(row["fight_url"])].append(row)
    clock_gaps = {}
    for key, rounds in grouped.items():
        if key not in fights:
            raise ValueError(f"round data has no matching fight: {key}")
        clock_gaps[key] = validate(fights[key], rounds)
    pending = set(fights) - grouped.keys()
    sources = defaultdict(list)
    if live and live.exists():
        for row in read_csv(live):
            sources[id_from_url(row["fight_url"])].append(row)
    added = 0
    problems = {}

    def accept(candidates, *, from_archive=False):
        nonlocal added
        for key, raw in candidates.items():
            if key not in pending:
                continue
            try:
                rounds = align(fights[key], raw)
                gaps = validate(fights[key], rounds, archive=from_archive)
            except ValueError as exc:
                problems[key] = str(exc)
                continue
            grouped[key] = rounds
            clock_gaps[key] = gaps
            pending.remove(key)
            problems.pop(key, None)
            added += 1

    accept(sources)
    if archive and pending:
        accept(archive_groups(details, archive, pending), from_archive=True)
    retryable = pending - sources.keys()
    if retry_limit > 0 and retryable:
        accept(retry_pages(fights, retryable, retry_limit))

    if added or not output.exists():
        staged = None
        try:
            with NamedTemporaryFile(dir=output.parent, mode="w", encoding="utf-8", newline="",
                                    delete=False) as stream:
                staged = Path(stream.name)
                writer = csv.DictWriter(stream, COLUMNS, lineterminator="\n")
                writer.writeheader()
                for key in fights:
                    writer.writerows(grouped.get(key, ()))
            os.replace(staged, output)
        finally:
            if staged is not None:
                staged.unlink(missing_ok=True)
    differing = sum(bool(gaps) for gaps in clock_gaps.values())
    summary = (f"Round data: {sum(map(len, grouped.values()))} rounds across {len(grouped)}/{len(fights)} fights. "
               f"{added} fights added, {len(pending)} pending. Round clocks differ from fight totals in {differing} fights.")
    print(summary)
    if report:
        Path(report).write_text(summary + "\n", encoding="utf-8")
    for key in sorted(pending):
        print(f"Missing rounds for {fights[key]['fight_url']}: {problems.get(key, 'no round rows available')}")
    for key, gaps in clock_gaps.items():
        if any(abs(delta) > len(grouped[key]) for delta in gaps.values()):
            print(f"Round clock discrepancy for {fights[key]['fight_url']} (seconds, round sum minus fight total): {gaps}")
    return added


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("current_df.csv"))
    parser.add_argument("--output", type=Path, default=Path("fight_rounds.csv"))
    parser.add_argument("--live", type=Path, help="Raw round rows from this run's fight scraper")
    parser.add_argument("--details", type=Path, help="Greco fight details mapping archive rows to URLs")
    parser.add_argument("--archive", type=Path, help="Greco ufc_fight_stats.csv, for backfill or retry")
    parser.add_argument("--report", type=Path, help="Write the coverage summary for the refresh PR")
    parser.add_argument("--retry-limit", type=int, default=0, help="Maximum missing fight pages to revisit")
    args = parser.parse_args()
    if bool(args.details) != bool(args.archive):
        parser.error("--archive and --details must be supplied together")
    update(args.dataset, args.output, args.live, args.details, args.archive, args.report, args.retry_limit)
