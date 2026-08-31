"""
Fighter bio scraper -- INCREMENTAL.

Scrapes a fighter's UFCStats.com bio page (Height, Weight, Reach, STANCE,
DOB) only when there's an actual reason to: they're new, they might have a
stale value, or their profile is incomplete. A full re-scrape of the whole
roster (~4,600+ pages) on every run isn't necessary and isn't polite to
UFCStats.com if this runs unattended on a schedule.

A fighter's URL is scraped if ANY of these hold:
  1. New       -- not present in --existing-fighters-csv at all.
  2. Active    -- appears as fighter_1/fighter_2 in --latest-fights-csv
                  (this run's freshly-scraped fights). Height/Reach/Stance/DOB
                  are permanent facts about a person and never need
                  re-fetching once captured, but Weight reflects UFCStats'
                  *current* contracted weight, which moves when a fighter
                  changes divisions -- and the most reliable signal that a
                  division change has become visible on their bio page is
                  that they just had a fight.
  3. Incomplete -- any of Height/Weight/Reach/STANCE/DOB is blank in
                  --existing-fighters-csv, e.g. because UFCStats hadn't
                  published it yet last time this ran.

Everyone else (known, not recently active, complete profile) is skipped.
Without --existing-fighters-csv, there's nothing to compare against, so
every fighter in --tott-csv is scraped -- this preserves the original
full-scrape behavior for a cold start.

Output is the FULL merged roster (existing rows carried through untouched,
re-scraped/new rows replacing or added), not just the delta -- so
--output can be used directly as next run's --existing-fighters-csv.
"""

import argparse
import time

from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm

BIO_FIELDS = ["Height", "Weight", "Reach", "STANCE", "DOB"]

PLAYWRIGHT = None
BROWSER = None
PAGE = None


def start_browser():
    global PLAYWRIGHT, BROWSER, PAGE

    # Imported lazily, not at module load, so determine_urls_to_scrape/
    # merge_scraped_results (pure logic, no browser needed) can be
    # imported and tested without playwright installed at all.
    from playwright.sync_api import sync_playwright

    PLAYWRIGHT = sync_playwright().start()
    BROWSER = PLAYWRIGHT.chromium.launch(headless=True)
    PAGE = BROWSER.new_page()

    # Block images/fonts/stylesheets
    PAGE.route(
        "**/*",
        lambda route: route.abort()
        if route.request.resource_type in ["image", "font", "stylesheet"]
        else route.continue_()
    )


def stop_browser():
    global PLAYWRIGHT, BROWSER

    if BROWSER:
        BROWSER.close()
    if PLAYWRIGHT:
        PLAYWRIGHT.stop()


def scrape_fighter(url, max_attempts=3, retry_delay_seconds=5):
    """
    Scrape one fighter's bio page. Retries on failure (network blip,
    Cloudflare hiccup) before giving up -- worth having now that this is
    meant to run unattended in CI rather than under a human who'd just
    re-run failed URLs by hand.
    """
    last_error = None

    for attempt in range(max_attempts):
        try:
            PAGE.goto(url, wait_until="domcontentloaded", timeout=30000)

            # Cloudflare protection
            if "Checking your browser" in PAGE.content():
                PAGE.wait_for_selector("ul.b-list__box-list", timeout=15000)

            html = PAGE.content()
            soup = BeautifulSoup(html, "html.parser")

            fighter_data = {"URL": url}

            labels = {
                "Height:": "Height",
                "Weight:": "Weight",
                "Reach:": "Reach",
                "STANCE:": "STANCE",
                "DOB:": "DOB",
            }

            for li in soup.select("ul.b-list__box-list li"):
                label_tag = li.find("i")
                if not label_tag:
                    continue

                label = label_tag.get_text(strip=True)
                if label in labels:
                    value = li.get_text(" ", strip=True).replace(label, "").strip()
                    fighter_data[labels[label]] = value

            for field in BIO_FIELDS:
                fighter_data.setdefault(field, "")

            return fighter_data

        except Exception as e:
            last_error = e
            if attempt < max_attempts - 1:
                tqdm.write(f"Retrying {url} (attempt {attempt + 2}/{max_attempts})")
                time.sleep(retry_delay_seconds)

    tqdm.write(f"❌ Failed: {url}")
    tqdm.write(str(last_error))
    return {"URL": url, "Error": str(last_error)}


def _is_missing(value):
    return pd.isna(value) or str(value).strip() in ("", "--")


def determine_urls_to_scrape(tott_df, existing_df, latest_fights_df):
    """
    Returns the set of fighter URLs to scrape, per the module docstring's
    three trigger conditions, plus a diagnostic dict of how many URLs each
    condition contributed (for logging -- useful to see at a glance in a
    CI log whether a run is doing the small incremental thing it should be).
    """
    all_known_urls = set(tott_df["URL"])

    if existing_df is None or existing_df.empty:
        # Cold start: nothing to compare against, scrape everyone.
        return all_known_urls, {"new": len(all_known_urls), "active": 0, "incomplete": 0}

    existing_urls = set(existing_df["URL"])
    new_urls = all_known_urls - existing_urls

    active_urls = set()
    if latest_fights_df is not None and not latest_fights_df.empty:
        for col in ["Fighter 1 URL", "Fighter 2 URL"]:
            if col in latest_fights_df.columns:
                active_urls |= set(latest_fights_df[col].dropna())
        active_urls &= existing_urls  # a brand-new fighter is already covered by new_urls

    incomplete_urls = set()
    present_bio_fields = [f for f in BIO_FIELDS if f in existing_df.columns]
    if present_bio_fields:
        incomplete_mask = existing_df[present_bio_fields].apply(
            lambda col: col.apply(_is_missing)
        ).any(axis=1)
        incomplete_urls = set(existing_df.loc[incomplete_mask, "URL"])

    urls_to_scrape = new_urls | active_urls | incomplete_urls
    return urls_to_scrape, {
        "new": len(new_urls),
        "active": len(active_urls),
        "incomplete": len(incomplete_urls),
    }


def merge_scraped_results(existing_df, scraped_records):
    """
    Merges freshly-scraped rows into the existing roster: replaces a row
    for any URL that was re-scraped, adds rows for brand-new URLs, and
    leaves every untouched URL's existing row exactly as it was. Rows
    where scraping failed (an 'Error' key, no real bio fields) are dropped
    from the merge rather than overwriting a previously-good row with
    blanks -- a transient failure this run shouldn't erase last run's
    successfully-scraped data.
    """
    scraped_df = pd.DataFrame(scraped_records)
    if "Error" in scraped_df.columns:
        failed_mask = scraped_df["Error"].notna()
        if failed_mask.any():
            tqdm.write(f"⚠️ {failed_mask.sum()} URL(s) failed to scrape -- keeping prior data for them, if any.")
        scraped_df = scraped_df.loc[~failed_mask].drop(columns=["Error"])

    if existing_df is None or existing_df.empty:
        return scraped_df

    untouched = existing_df[~existing_df["URL"].isin(scraped_df["URL"])]
    return pd.concat([untouched, scraped_df], ignore_index=True)


def run_fighter_scrape(tott_csv, existing_fighters_csv=None, latest_fights_csv=None,
                        output_csv="ufc_fighters_with_details_inc_dob.csv",
                        failed_urls_csv="fighter_failed_urls.csv", batch_size=500):
    tott_df = pd.read_csv(tott_csv)

    existing_df = None
    if existing_fighters_csv:
        try:
            existing_df = pd.read_csv(existing_fighters_csv)
        except FileNotFoundError:
            print(f"No existing fighters file at {existing_fighters_csv} -- treating as a cold start.")

    latest_fights_df = None
    if latest_fights_csv:
        try:
            latest_fights_df = pd.read_csv(latest_fights_csv)
        except FileNotFoundError:
            print(f"No latest-fights file at {latest_fights_csv} -- 'active fighter' trigger will find nothing.")

    urls_to_scrape, breakdown = determine_urls_to_scrape(tott_df, existing_df, latest_fights_df)
    urls_to_scrape = sorted(urls_to_scrape)

    print(
        f"Scraping {len(urls_to_scrape)} of {len(tott_df)} known fighters "
        f"(new: {breakdown['new']}, recently active: {breakdown['active']}, "
        f"incomplete profile: {breakdown['incomplete']})."
    )

    if not urls_to_scrape:
        print("Nothing to scrape. Writing existing roster through unchanged.")
        if existing_df is not None:
            existing_df.to_csv(output_csv, index=False)
        return

    start_browser()

    all_records = []
    failed_urls = []

    try:
        for url in tqdm(urls_to_scrape, desc="Scraping Fighters"):
            record = scrape_fighter(url)
            all_records.append(record)
            if "Error" in record:
                failed_urls.append(record)
    finally:
        stop_browser()

    if failed_urls:
        pd.DataFrame(failed_urls).to_csv(failed_urls_csv, index=False)
        print(f"⚠️ {len(failed_urls)} failure(s) saved to {failed_urls_csv}")

    merged = merge_scraped_results(existing_df, all_records)
    merged.to_csv(output_csv, index=False)
    print(f"✅ Finished. Roster now has {len(merged):,} fighters (written to {output_csv}).")


def _parse_args():
    parser = argparse.ArgumentParser(description="Incremental UFCStats.com fighter bio scraper.")
    parser.add_argument("--tott-csv", required=True,
                         help="Path to ufc_fighter_tott.csv (source of ALL known fighter URLs).")
    parser.add_argument("--existing-fighters-csv", default=None,
                         help="Path to the current fighters_df.csv. Omit for a cold-start full scrape.")
    parser.add_argument("--latest-fights-csv", default=None,
                         help="Path to this run's freshly-scraped fights (drives the 'active fighter' trigger).")
    parser.add_argument("--output", default="ufc_fighters_with_details_inc_dob.csv")
    parser.add_argument("--failed-urls-output", default="fighter_failed_urls.csv")
    parser.add_argument("--batch-size", type=int, default=500)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_fighter_scrape(
        tott_csv=args.tott_csv,
        existing_fighters_csv=args.existing_fighters_csv,
        latest_fights_csv=args.latest_fights_csv,
        output_csv=args.output,
        failed_urls_csv=args.failed_urls_output,
        batch_size=args.batch_size,
    )
