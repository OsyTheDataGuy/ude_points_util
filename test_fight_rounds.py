"""Regression tests for the round-grain export; no live network is used."""

import csv
import tempfile
import unittest
from pathlib import Path

from bs4 import BeautifulSoup

import fight_rounds as fr


def table(headers, row_values):
    headings = "".join(f"<th>{label}</th>" for label in headers)
    cells = []
    for pair in row_values:
        cells.append("<td>" + "".join(f"<p>{value}</p>" for value in pair) + "</td>")
    return (f"<table><thead><tr>{headings}</tr></thead><tbody></tbody>"
            "<thead><tr><th>Round 1</th></tr></thead><tbody><tr>"
            + "".join(cells) + "</tr></tbody></table>")


def round_page():
    names = ('<a href="/b">Blue</a>', '<a href="/a">Red</a>')
    totals = table(("Fighter", "KD", "Sig. str.", "%", "Total", "TD", "%", "Sub", "Rev", "Ctrl"),
                   (names, ("0", "1"), ("1 of 1", "2 of 3"), ("100%", "66%"),
                    ("1 of 2", "2 of 4"), ("0 of 1", "0 of 0"), ("0%", "--"),
                    ("0", "0"), ("0", "0"), ("--", "0:00")))
    strikes = table(("Fighter", "Sig. str.", "%", "Head", "Body", "Leg", "Distance", "Clinch", "Ground"),
                    (names, ("1 of 1", "2 of 3"), ("100%", "66%"), ("1 of 1", "2 of 3"),
                     ("0 of 0", "0 of 0"), ("0 of 0", "0 of 0"), ("1 of 1", "2 of 3"),
                     ("0 of 0", "0 of 0"), ("0 of 0", "0 of 0")))
    return BeautifulSoup(totals + strikes, "html.parser")


def write_rows(path, header, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, header, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


class RoundTests(unittest.TestCase):
    def test_refresh_retries_missing_fight_and_preserves_bytes(self):
        fight = dict(event_name="Event", event_date="2020-01-01", fight_url="/match",
                     fighter_1="Red", fighter_url_fighter_1="/a",
                     fighter_2="Blue", fighter_url_fighter_2="/b", round_ended="1.0")
        raw = fr.source_rounds(round_page(), "/match")
        aligned, = fr.align(fight, raw)
        self.assertEqual(aligned["kd_fighter_1"], 1)
        self.assertEqual(aligned["ctrl_in_secs_fighter_1"], 0)
        self.assertIsNone(aligned["ctrl_in_secs_fighter_2"])
        fight.update({col: aligned[col] for col in fr.STATS})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, output, live = root/"fights.csv", root/"rounds.csv", root/"live.csv"
            write_rows(dataset, fight.keys(), [fight])
            self.assertEqual(fr.update(dataset, output), 0)
            write_rows(live, fr.RAW_COLUMNS, raw)
            self.assertEqual(fr.update(dataset, output, live), 1)
            before = output.read_bytes()
            self.assertEqual(fr.update(dataset, output, live), 0)
            self.assertEqual(before, output.read_bytes())
            self.assertNotIn(b"\r", before)
            self.assertEqual(next(fr.read_csv(output))["ctrl_in_secs_fighter_2"], "")
            # A later inconsistency must not replace the previously verified export.
            fight["kd_fighter_1"] = 9
            write_rows(dataset, fight.keys(), [fight])
            with self.assertRaisesRegex(ValueError, "kd_fighter_1"):
                fr.update(dataset, output)
            self.assertEqual(before, output.read_bytes())

    def test_missing_table_and_duplicate_side_rejected(self):
        soup = round_page()
        soup.select("table")[1].decompose()
        with self.assertRaisesRegex(ValueError, "incomplete"):
            fr.source_rounds(soup, "/match")
        raw = fr.source_rounds(round_page(), "/match")
        fight = {key: "" for key in fr.META}
        fight.update(fighter_1="Red", fighter_2="Blue", fighter_url_fighter_1="/a", fighter_url_fighter_2="/b")
        with self.assertRaisesRegex(ValueError, "duplicate fighter"):
            fr.align(fight, raw + raw)

    def test_archive_clock_disagreement_requires_primary_page(self):
        raw = fr.source_rounds(round_page(), "/match")
        fight = {key: "" for key in fr.META}
        fight.update(fighter_1="Red", fighter_2="Blue", fighter_url_fighter_1="/a",
                     fighter_url_fighter_2="/b", round_ended="1", fight_url="/match")
        rounds = fr.align(fight, raw)
        fight.update({col: str(rounds[0][col]) if rounds[0][col] is not None else "" for col in fr.STATS})
        fight["ctrl_in_secs_fighter_1"] = "5"
        with self.assertRaisesRegex(ValueError, "direct round-table verification"):
            fr.validate(fight, rounds, archive=True)
        self.assertEqual(fr.validate(fight, rounds), {"ctrl_in_secs_fighter_1": -5})

    def test_published_makhachev_volkanovski_fights(self):
        first, second = [], []
        for row in fr.read_csv(Path(__file__).with_name("fight_rounds.csv")):
            key = fr.id_from_url(row["fight_url"])
            if key == "256894b49303537b":
                first.append(row)
            elif key == "0ea435ddf02deb60":
                second.append(row)
        self.assertEqual([r["round_number"] for r in first], ["1", "2", "3", "4", "5"])
        self.assertEqual([r["round_number"] for r in second], ["1"])
        self.assertEqual(first[0]["fighter_1"], "Islam Makhachev")
        self.assertEqual(first[0]["fighter_2"], "Alexander Volkanovski")
        self.assertEqual([int(r["td_landed_fighter_1"]) for r in first], [1, 1, 1, 1, 0])
        self.assertEqual(sum(int(r["ctrl_in_secs_fighter_1"]) for r in first), 457)
        self.assertEqual(second[0]["kd_fighter_1"], "1")
