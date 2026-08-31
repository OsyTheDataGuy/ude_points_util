"""
Single entry point for a full UDE Points dataset refresh.

Chains: run_etl_pipeline -> Stance backfill -> engineer_all_features ->
calculate_ude_points_with_ablation -> add_ude_points_difference_columns ->
validate_dataset_regeneration -> write the new production file.

Note --current-dataset and --previous-production-file are NOT the same
file: --current-dataset (e.g. current_df.csv) is the raw/ETL-stage
historical snapshot run_etl_pipeline appends fresh scrapes onto; it has no
pdi_margin or UDE points at all. --previous-production-file is the prior
FULLY SCORED output (what --output was last time this ran) -- that's the
only thing validate_dataset_regeneration can meaningfully diff against,
since scoring doesn't exist until after this script's own pipeline runs.

Exits non-zero if validate_dataset_regeneration raises (a fight_url was
lost, or a column changed that wasn't explicitly allowed) -- a GitHub
Actions workflow should treat that exit code as "stop, don't open a PR."
"""

import argparse
import sys

import pandas as pd

from dataset_processing_pipeline import run_etl_pipeline, validate_dataset_regeneration
from ude_points_feature_engineering_pipeline import engineer_all_features
from ude_points_algorithm import calculate_ude_points_with_ablation, add_ude_points_difference_columns


def backfill_stance(raw_df: pd.DataFrame, fighters_df: pd.DataFrame) -> pd.DataFrame:
    """
    run_etl_pipeline's own bio merge (dataset_processing_pipeline.py's
    bio_cols step) only covers freshly-scraped rows -- historical rows
    carried in via --current-dataset don't go through it, so they'd have
    no Stance value without this. Stance is a largely time-invariant,
    roster-wide fighter property, so backfilling it directly from
    fighters_df by fighter_url is safe -- same approach used throughout
    this project's history for this exact file.
    """
    stance_lookup = fighters_df.set_index('URL')['STANCE'].replace('--', pd.NA)
    raw_df = raw_df.copy()
    for side in ('1', '2'):
        col = f'Stance_fighter_{side}'
        url_col = f'fighter_url_fighter_{side}'
        existing = raw_df[col] if col in raw_df.columns else pd.Series(pd.NA, index=raw_df.index)
        raw_df[col] = raw_df[url_col].map(stance_lookup).fillna(existing)
    return raw_df


def run_refresh(current_dataset_csv, fighters_csv, latest_fights_csv, latest_events_csv,
                 previous_production_csv, output_csv, columns_expected_to_change=None):
    print("Loading inputs...")
    current_dataset = pd.read_csv(current_dataset_csv, low_memory=False)
    fighters_df = pd.read_csv(fighters_csv, low_memory=False)
    latest_fights_df = pd.read_csv(latest_fights_csv, low_memory=False)
    latest_events_df = pd.read_csv(latest_events_csv, low_memory=False)
    previous_production_df = pd.read_csv(previous_production_csv, low_memory=False)

    print("Running ETL...")
    raw = run_etl_pipeline(
        scraped_fights_df=latest_fights_df,
        events_df=latest_events_df,
        fighters_df=fighters_df,
        current_dataset=current_dataset,
    )

    print("Backfilling Stance across the full historical set...")
    raw = backfill_stance(raw, fighters_df)

    print("Running feature engineering...")
    engineered = engineer_all_features(raw)

    print("Scoring UDE points...")
    scored = calculate_ude_points_with_ablation(engineered)
    final = add_ude_points_difference_columns(scored)

    print(f"Regenerated dataset: {final.shape}")

    print("Validating against the previous production file...")
    result = validate_dataset_regeneration(
        previous_production_df, final,
        columns_expected_to_change=columns_expected_to_change,
    )

    final.to_csv(output_csv, index=False)
    print(f"Wrote {output_csv}. New fights added this run: {len(result['new_keys'])}.")
    return result


def _parse_args():
    parser = argparse.ArgumentParser(description="Full UDE Points dataset refresh, with validation.")
    parser.add_argument("--current-dataset", required=True,
                         help="Prior raw/ETL-stage historical snapshot (e.g. current_df.csv) -- "
                              "NOT the fully-scored production file.")
    parser.add_argument("--fighters", required=True, help="fighters_df.csv")
    parser.add_argument("--latest-fights", required=True, help="This run's freshly-scraped fights.")
    parser.add_argument("--latest-events", required=True, help="Latest ufc_event_details.csv.")
    parser.add_argument("--previous-production-file", required=True,
                         help="The prior fully-scored production file, to diff the new one against.")
    parser.add_argument("--output", required=True, help="Where to write the newly regenerated production file.")
    parser.add_argument("--allow-changed-columns", default="",
                         help="Comma-separated columns allowed to differ from --previous-production-file "
                              "without failing validation. Normally empty -- a routine refresh (new fights "
                              "appended, no code changes) should never change any existing fight's own "
                              "columns. Only set this when a deliberate code fix is shipping alongside "
                              "this refresh, and only for the columns that fix is actually supposed to touch.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    allowed = [c.strip() for c in args.allow_changed_columns.split(",") if c.strip()]
    try:
        run_refresh(
            current_dataset_csv=args.current_dataset,
            fighters_csv=args.fighters,
            latest_fights_csv=args.latest_fights,
            latest_events_csv=args.latest_events,
            previous_production_csv=args.previous_production_file,
            output_csv=args.output,
            columns_expected_to_change=allowed,
        )
    except ValueError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)
