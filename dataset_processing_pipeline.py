# -*- coding: utf-8 -*-
"""
UFC Fight Data - ETL Processing Pipeline
Converts raw scraped fight, event, and fighter bio datasets into a clean 1-row-per-fight dataset.
"""

import pandas as pd
import numpy as np
from datetime import datetime


# ==============================================================================
# 1. Fighter Bio Preprocessing
# ==============================================================================

def process_fighter_bio(fighters_df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans HEIGHT, WEIGHT, REACH, and STANCE from raw fighter profile data.
    - HEIGHT: converted from feet/inches (e.g. 5' 11") to meters (float)
    - WEIGHT: converted from lbs string (e.g. '155 lbs.') to float
    - REACH: converted from inches string (e.g. '72"') to float
    - STANCE: passed through as-is (already a clean categorical string),
      exposed as 'Stance' to match the Title Case naming of the other
      derived bio fields (Height (m), Weight (lbs), Reach (in)) rather
      than the all-caps raw scrape column name.
    """
    # CHANGE: Maintained one copy here at the ingestion point, removed downstream copies.
    df = fighters_df.copy()

    # CHANGE: Pre-step standardizing for column names and null values.
    df.columns = df.columns.str.upper()
    resolve_cols = [c for c in ['HEIGHT', 'WEIGHT', 'REACH', 'STANCE', 'DOB'] if c in df.columns]
    if resolve_cols:
        df[resolve_cols] = df[resolve_cols].replace('--', np.nan)

    # CHANGE: Replaced Python-level .apply() functions with vectorized regex extraction.
    if 'HEIGHT' in df.columns:
        extracted = df['HEIGHT'].astype(str).str.extract(r"(\d+)'\s*(\d+)?")
        df['Height (m)'] = (extracted[0].astype(float) * 0.3048 + extracted[1].fillna(0).astype(float) * 0.0254).round(2)

    if 'WEIGHT' in df.columns:
        df['Weight (lbs)'] = df['WEIGHT'].astype(str).str.extract(r'(\d+)').astype(float)

    if 'REACH' in df.columns:
        df['Reach (in)'] = df['REACH'].astype(str).str.extract(r'(\d+)').astype(float)

    if 'STANCE' in df.columns:
        df['Stance'] = df['STANCE']

    return df


# ==============================================================================
# 2. Column Standardizing & Cleaning
# ==============================================================================

def rename_strike_columns(df: pd.DataFrame, strike_columns: list) -> pd.DataFrame:
    """Renames 'str' to 'strikes' and appends '_strikes' suffix to strike columns."""
    df.columns = df.columns.str.replace('str', 'strikes')
    rename_dict = {col: f"{col}_strikes" for col in strike_columns if col in df.columns}
    return df.rename(columns=rename_dict)


def standardize_columns(df: pd.DataFrame, strike: bool = True) -> pd.DataFrame:
    """
    Standardizes column names: lowercase, strip periods, replace spaces/periods with underscores.
    """
    # CHANGE: Removed df = df.copy() to eliminate memory allocation overhead.
    df.columns = (
        df.columns.str.lower()
        .str.strip('.')
        .str.replace(' ', '_')
        .str.replace('.', '_', regex=False)
        .str.replace('%', 'pct')
        .str.replace('__', '_')
    )

    if strike:
        strike_columns = ['head', 'body', 'leg', 'distance', 'clinch', 'ground']
        df = rename_strike_columns(df, strike_columns)

    if 'finish_details' in df.columns:
        df = df.rename(columns={'finish_details': 'details'})

    return df


# ==============================================================================
# 3. Split Fight Stats (1 Row -> 2 Rows: 1 per fighter)
# ==============================================================================

def split_fight_stats(df: pd.DataFrame, stat_cols: list) -> pd.DataFrame:
    """
    Splits multi-line newline-delimited statistics into two separate rows (one per fighter).
    """
    # CHANGE: Eliminated hardcoded positional index 'border=16' in favor of explicit schema declarations.
    actual_stat_cols = [c for c in stat_cols if c in df.columns]
    common_cols = [c for c in df.columns if c not in actual_stat_cols]

    df_f1 = df[common_cols].copy()
    df_f2 = df[common_cols].copy()

    # CHANGE: Replaced iterrows() loop with fully vectorized Pandas string splitting.
    for col in actual_stat_cols:
        if col == 'fighter':
            # Preserve spaces for fighter names
            split_df = df[col].astype(str).str.split('\n', expand=True)
        else:
            # Strip spaces from numeric stats before split
            split_df = df[col].astype(str).str.replace(' ', '').str.split('\n', expand=True)

        df_f1[col] = split_df[0].str.strip() if 0 in split_df.columns else ''
        df_f2[col] = split_df[1].str.strip() if 1 in split_df.columns else ''

    # Recombine split frames back into 2 rows per fight
    new_df = pd.concat([df_f1, df_f2], ignore_index=True)
    return new_df


def clean_modified_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assigns correct fighter_url and fight_result to each row and drops redundant columns.
    """
    # CHANGE: Removed df = df.copy().
    df['fighter_url'] = np.where(
        df['fighter_1_name'] == df['fighter'],
        df['fighter_1_url'],
        df['fighter_2_url']
    )
    df['fight_result'] = np.where(
        df['fighter_1_name'] == df['fighter'],
        df['fighter_1_result'],
        df['fighter_2_result']
    )

    drop_cols = [
        'fighter_1_name', 'fighter_1_url', 'fighter_1_result',
        'fighter_2_name', 'fighter_2_url', 'fighter_2_result'
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    return df


# ==============================================================================
# 4. Metric Parsing Helpers
# ==============================================================================

def split_strike_column(df: pd.DataFrame, col: str, new_col_landed: str, new_col_attempted: str) -> pd.DataFrame:
    """Splits 'landed of attempted' string into separate numeric columns."""
    split_df = df[col].astype(str).str.split('of', expand=True)
    df[new_col_landed] = pd.to_numeric(split_df[0].str.strip(), errors='coerce')
    df[new_col_attempted] = pd.to_numeric(split_df[1].str.strip(), errors='coerce') if split_df.shape[1] > 1 else np.nan
    return df


def apply_split(df: pd.DataFrame, columns_to_split: list, drop_original: bool = True) -> pd.DataFrame:
    """Splits a list of strike/grappling columns into '_landed' and '_attempted' numeric columns."""
    # CHANGE: Removed df = df.copy().
    col_mapping = {col: (f"{col}_landed", f"{col}_attempted") for col in columns_to_split}

    for col, (landed_col, attempted_col) in col_mapping.items():
        if col in df.columns:
            df = split_strike_column(df, col, landed_col, attempted_col)

    if drop_original:
        df = df.drop(columns=[col for col in col_mapping.keys() if col in df.columns])

    return df

def calculate_age(df: pd.DataFrame, dob_col: str = 'DOB', date_col: str = 'event_date') -> pd.DataFrame:
    """Calculates fighter age in years on fight day from DOB and event_date."""
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

    dob_cleaned = df[dob_col].astype(str).str.strip().replace('--', np.nan)
    # Flexible parsing to handle both raw scraped ('Oct 30, 1986') and timestamp strings ('1984-05-16 00:00:00')
    dob_parsed = pd.to_datetime(dob_cleaned, errors='coerce')

    # Calculate fight day age
    df['fight_day_age (yrs)'] = (df[date_col] - dob_parsed).dt.days / 365.25

    # Standardize DOB column to YYYY-MM-DD format
    df[dob_col] = dob_parsed.dt.strftime('%Y-%m-%d')

    return df

def process_control_time(df: pd.DataFrame, col: str = 'ctrl') -> pd.DataFrame:
    """Converts control time string 'MM:SS' into integer seconds."""
    # CHANGE: Refactored individual string logic into a vectorized structural operation to process all rows at once.
    if col in df.columns:
        time_parts = df[col].astype(str).str.split(':', expand=True)
        if time_parts.shape[1] == 2:
            df['ctrl_in_secs'] = (
                pd.to_numeric(time_parts[0], errors='coerce').fillna(0) * 60 +
                pd.to_numeric(time_parts[1], errors='coerce').fillna(0)
            )
        else:
            df['ctrl_in_secs'] = 0
        df = df.drop(columns=[col])
    return df


def convert_pct_to_float(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Converts percentage string (e.g. '48%') to float (0.48)."""
    if col in df.columns:
        cleaned = df[col].astype(str).replace('---', np.nan).str.rstrip('%')
        df[col] = pd.to_numeric(cleaned, errors='coerce') / 100.0
    return df


def convert_cols_from_pct_to_float(df: pd.DataFrame, cols_to_convert: list) -> pd.DataFrame:
    """Applies percentage-to-float conversion across multiple columns."""
    # CHANGE: Removed df = df.copy().
    for col in cols_to_convert:
        df = convert_pct_to_float(df, col)
    return df


# ==============================================================================
# 5. Reshape to 1-Fight-Per-Row
# ==============================================================================

def add_suffix_to_fighter_columns(df: pd.DataFrame, number: int, exclude_cols: list = None) -> pd.DataFrame:
    """Adds suffix '_fighter_1' or '_fighter_2' to fighter-specific metric columns."""
    # CHANGE: Removed df = df.copy().
    if exclude_cols is None:
        exclude_cols = [
            'event_name', 'event_url', 'event_date', 'weight_class',
            'fight_url', 'fighter', 'method', 'details', 'time', 'round', 'time_format'
        ]

    renames = {}
    for col in df.columns:
        if col not in exclude_cols:
            renames[col] = f"{col}_fighter_{number}"

    df = df.rename(columns=renames)
    df = df.rename(columns={'fighter': f'fighter_{number}'})
    return df


def convert_to_one_fight_one_row(df: pd.DataFrame) -> pd.DataFrame:
    """Pivots from 2 rows per fight to 1 row per fight with fighter_1 and fighter_2 statistics."""
    # drop=True: .nth(0)/.nth(1) preserve the original (pre-groupby) row
    # index, so a bare .reset_index() turns that meaningless row position
    # into a genuine 'index' column -- it was being silently caught by the
    # old ordered_columns whitelist at the end of run_etl_pipeline, which
    # is exactly the failure mode that whitelist was just changed to stop
    # relying on (see run_etl_pipeline's column-ordering step).
    f1_df = add_suffix_to_fighter_columns(df.groupby('fight_url').nth(0).reset_index(drop=True), 1)
    f2_df = add_suffix_to_fighter_columns(df.groupby('fight_url').nth(1).reset_index(drop=True), 2)

    fight_df = f1_df.merge(f2_df, on=['fight_url'])
    fight_df['bout'] = fight_df['fighter_1'] + ' vs. ' + fight_df['fighter_2']

    # Drop duplicated _y columns from merge and clean _x suffixes
    fight_df = fight_df.loc[:, ~fight_df.columns.str.endswith('_y')]
    fight_df = fight_df.rename(columns=lambda x: x[:-2] if x.endswith('_x') else x)

    # Standardize column naming
    fight_df = fight_df.rename(columns={
        'DOB_fighter_1': 'date_of_birth_fighter_1',
        'DOB_fighter_2': 'date_of_birth_fighter_2',
        'round': 'round_ended'
    })

    return fight_df

# ==============================================================================
# 5b. Event-Date Join Integrity
# ==============================================================================

def drop_rows_with_null_event_date(df: pd.DataFrame, date_col: str = 'event_date',
                                    id_cols: list = None, verbose: bool = True) -> pd.DataFrame:
    """
    Drops rows with a null event_date and reports exactly what was dropped,
    rather than silently discarding them.

    event_date isn't a minor missing-field gap like Height or DOB -- every
    downstream stage (feature engineering's per-fighter chronological state
    machines, UDE's temporal calibration) sorts and accumulates state by it,
    so a null row can't be safely placed in time or processed at all.
    validate_transformed_data's join-leakage check only monitors an aggregate
    null *percentage* against a threshold, so a handful of null rows in a
    large dataset can pass silently; this is a hard, unconditional drop
    instead, specifically because there's no safe way to process such a row.

    A null here almost always means the Event URL left-join in step 2 above
    found no match in events_df -- typically a newly added event not yet
    present in the events source, or a naming mismatch. Either way, it's
    worth a human looking at (the event might be missing real data upstream,
    not just this one row), not just discarding quietly.
    """
    if date_col not in df.columns:
        return df

    null_mask = df[date_col].isna()
    n_dropped = int(null_mask.sum())
    if n_dropped > 0:
        if id_cols is None:
            id_cols = [c for c in ['event_name', 'event_url', 'fight_url', 'bout',
                                    'fighter_1', 'fighter_2', 'fighter_1_name', 'fighter_2_name']
                       if c in df.columns]
        if verbose:
            print(f"WARNING: Dropping {n_dropped} row(s) with null '{date_col}' "
                  f"(event-date join found no match) -- investigate the events source:")
            print(df.loc[null_mask, id_cols].to_string(index=False))

    return df.loc[~null_mask].reset_index(drop=True)


# ==============================================================================
# 6. ETL Pipeline validator
# ==============================================================================
def validate_transformed_data(df: pd.DataFrame, max_null_pct: float = 0.05, verbose: bool = True,
                               current_dataset: pd.DataFrame = None) -> None:
    """
    Executes post-transformation sanity checks on the processed dataset.
    Prints status updates for each check and raises warnings/errors if invariants are violated.

    current_dataset (optional): the historical dataset this batch is about
    to be appended to (run_etl_pipeline's own current_dataset argument).
    When supplied, adds a check that none of `df`'s fight_url values already
    exist in current_dataset. This call runs on the freshly-scraped batch
    BEFORE current_dataset is merged in (see run_etl_pipeline step 13), so
    the Primary Key Uniqueness check below only ever sees this batch in
    isolation -- it cannot catch a fight that's genuinely new to this batch
    but was already scraped and merged in on a previous run (e.g. an
    incremental-fight-filter bug in the upstream scraper re-including
    something already processed). Only this explicit cross-check can.
    """
    import warnings

    if verbose:
        print("\n" + "=" * 50)
        print(" RUNNING ETL DATA VALIDATION SUITE")
        print("=" * 50)

    checks_executed = 0

    # 1. Primary Key Uniqueness
    if 'fight_url' in df.columns:
        if not df['fight_url'].is_unique:
            duplicates = df['fight_url'].duplicated().sum()
            raise ValueError(f"❌ [FAIL] Primary Key Check: 'fight_url' contains {duplicates} duplicate records.")
        checks_executed += 1
        if verbose:
            print("✓ [PASS] Primary Key Uniqueness (fight_url)")

    # 2. Logical Invariant: Landed <= Attempted
    strike_cols = [c.replace('_landed_', '') for c in df.columns if '_landed_' in c]
    for base in set(strike_cols):
        l_col, a_col = f"{base}_landed_fighter_1", f"{base}_attempted_fighter_1"
        if l_col in df.columns and a_col in df.columns:
            invalid_rows = (df[l_col] > df[a_col]).sum()
            if invalid_rows > 0:
                raise ValueError(f"❌ [FAIL] Logical Invariant Check: Found {invalid_rows} rows where {l_col} > {a_col}.")
    checks_executed += 1
    if verbose:
        print("✓ [PASS] Logical Invariants (Landed <= Attempted)")

    # 3. Physiological Bounds Checks
    age_cols = [c for c in df.columns if 'fight_day_age' in c]
    age_warnings = 0
    for col in age_cols:
        invalid_ages = df[(df[col] < 18) | (df[col] > 65)][col].count()
        if invalid_ages > 0:
            age_warnings += invalid_ages
            warnings.warn(f"Data Quality Warning: Found {invalid_ages} fighters with age outside [18, 65] in {col}.")
    checks_executed += 1
    if verbose:
        status = "✓ [PASS] Physiological Bounds (Ages 18–65)" if age_warnings == 0 else f"⚠️ [WARN] Physiological Bounds ({age_warnings} outliers flagged)"
        print(status)

    # 4. Control Time Upper Bound
    ctrl_cols = [c for c in df.columns if 'ctrl_in_secs' in c]
    ctrl_warnings = 0
    for col in ctrl_cols:
        invalid_ctrl = (df[col] > 1500).sum()
        if invalid_ctrl > 0:
            ctrl_warnings += invalid_ctrl
            warnings.warn(f"Data Quality Warning: Found {invalid_ctrl} rows with control time > 1500s in {col}.")
    checks_executed += 1
    if verbose:
        status = "✓ [PASS] Control Time Bounds (<= 1500s)" if ctrl_warnings == 0 else f"⚠️ [WARN] Control Time Bounds ({ctrl_warnings} outliers flagged)"
        print(status)

    # 5. Join Leakage Check (Excessive Nulls in Critical Columns)
    critical_cols = ['event_date', 'Height (m)_fighter_1', 'date_of_birth_fighter_1']
    null_warnings = 0
    for col in critical_cols:
        if col in df.columns:
            null_pct = df[col].isna().mean()
            if null_pct > max_null_pct:
                null_warnings += 1
                warnings.warn(f"Data Quality Warning: '{col}' has {null_pct:.1%} null values (exceeds threshold of {max_null_pct:.1%}).")
    checks_executed += 1
    if verbose:
        status = f"✓ [PASS] Join Coverage (<{max_null_pct:.0%} Nulls)" if null_warnings == 0 else f"⚠️ [WARN] Join Coverage ({null_warnings} columns exceeded null threshold)"
        print(status)

    # 6. No Duplicate vs. History (only runs if current_dataset supplied)
    if current_dataset is not None and 'fight_url' in df.columns and 'fight_url' in current_dataset.columns:
        already_seen = df['fight_url'].isin(current_dataset['fight_url'])
        n_already_seen = int(already_seen.sum())
        if n_already_seen > 0:
            raise ValueError(
                f"❌ [FAIL] No-Duplicate-vs-History Check: {n_already_seen} row(s) in this batch "
                f"have a fight_url already present in current_dataset -- likely a re-scrape of "
                f"already-processed fights. Affected fight_urls:\n"
                f"{df.loc[already_seen, 'fight_url'].to_string(index=False)}"
            )
        checks_executed += 1
        if verbose:
            print("✓ [PASS] No Duplicate vs. History (fight_url not already in current_dataset)")

    if verbose:
        print("-" * 50)
        print(f"Validation Complete: {checks_executed} quality suites executed.")
        print("=" * 50 + "\n")


def validate_dataset_regeneration(old_df: pd.DataFrame, new_df: pd.DataFrame, key_col: str = 'fight_url',
                                   columns_expected_to_change: list = None, verbose: bool = True) -> dict:
    """
    Compares a prior fully-processed dataset against a freshly regenerated
    one, joined on `key_col`, and reports exactly what changed. Formalizes
    the ad hoc diff checks this project has run by hand after every
    regeneration this session (STANCE casing fix, the ordered_columns/
    reset_index fixes, the decisive/close-wins classification fix, the
    UFC 330 stance addition) -- each of those was verified the same way:
    same row count and key set, then a column-by-column value diff against
    an explicit list of columns expected to change.

    Unlike validate_transformed_data (which runs on raw ETL output, before
    feature engineering or UDE scoring even happen), this is meant to run
    on the FINAL output of the full pipeline -- after engineer_all_features,
    calculate_ude_points_with_ablation, and add_ude_points_difference_columns
    -- since that's the only stage where a regression in pdi_margin, a UDE
    point, or an engineered feature would actually be visible. The two
    validators check different things at different pipeline stages and are
    both needed; neither can do the other's job.

    columns_expected_to_change: columns allowed to differ without raising
    (e.g. the classification-count columns for the phase-magnitude rounding
    fix). Any OTHER column that differs raises -- an unexpected column
    changing during what should be a routine data refresh is exactly the
    failure mode this project has hit multiple times (a column silently
    dropped, a stray column silently created, a fix's blast radius turning
    out wider than expected). Row-count and key-set mismatches always raise
    regardless of columns_expected_to_change, since a missing or duplicated
    fight is never an acceptable "expected change."

    Returns a dict: {'new_keys': set, 'missing_keys': set,
    'changed_columns': {col: n_rows_differing}, 'unexpected_changes': bool}.
    """
    if verbose:
        print("\n" + "=" * 50)
        print(" DATASET REGENERATION DIFF")
        print("=" * 50)

    columns_expected_to_change = set(columns_expected_to_change or [])

    old_keys = set(old_df[key_col])
    new_keys = set(new_df[key_col])
    added_keys = new_keys - old_keys
    missing_keys = old_keys - new_keys

    if missing_keys:
        raise ValueError(
            f"❌ [FAIL] {len(missing_keys)} {key_col} value(s) present in the old dataset are "
            f"missing from the new one -- a regeneration should never lose a previously-processed "
            f"fight. Sample: {list(missing_keys)[:5]}"
        )
    if verbose:
        print(f"✓ [PASS] No {key_col} values lost ({len(added_keys)} new added, "
              f"{len(old_keys & new_keys)} shared)")

    shared = old_df[old_df[key_col].isin(old_keys & new_keys)]
    new_shared = new_df[new_df[key_col].isin(old_keys & new_keys)]
    merged = shared.merge(new_shared, on=key_col, suffixes=('_old', '_new'))

    common_cols = [c for c in old_df.columns if c in new_df.columns and c != key_col]
    changed_columns = {}
    for c in common_cols:
        a, b = merged[f'{c}_old'], merged[f'{c}_new']
        both_null = a.isna() & b.isna()
        only_one_null = a.isna() ^ b.isna()
        if a.dtype.kind in 'biufc' and b.dtype.kind in 'biufc':
            # (a - b) is NaN whenever either side is NaN, and NaN > 1e-9 is
            # always False -- so a real regression where a numeric value
            # silently becomes NaN (or vice versa) would otherwise never be
            # caught. only_one_null closes that gap explicitly. both_null
            # needs no explicit handling here: NaN > 1e-9 already reads as
            # "no difference" for it, correctly.
            n_diff = int((((a - b).abs() > 1e-9) | only_one_null).sum())
        else:
            # both_null must be excluded explicitly here: pandas represents
            # a missing value as NaN after a CSV round-trip but as Python
            # None on a value fresh out of the pipeline (e.g. dominant_fighter),
            # and str(NaN) != str(None) -- without this, "no dominant fighter"
            # on both sides reads as a value CHANGE purely from which null
            # representation each side happened to use. Confirmed live: this
            # was the exact and only cause of an apparent 4,649-row
            # dominant_fighter "regression" that wasn't real.
            n_diff = int(((a.astype(str) != b.astype(str)) & ~both_null).sum())
        if n_diff > 0:
            changed_columns[c] = n_diff

    unexpected = {c: n for c, n in changed_columns.items() if c not in columns_expected_to_change}

    if verbose:
        if changed_columns:
            print(f"Columns that differ on shared {key_col}s:")
            for c, n in changed_columns.items():
                flag = "" if c in columns_expected_to_change else "  <-- UNEXPECTED"
                print(f"  {c}: {n} row(s) differ{flag}")
        else:
            print("No column-level differences on shared rows.")

    if unexpected:
        raise ValueError(
            f"❌ [FAIL] {len(unexpected)} column(s) changed unexpectedly (not in "
            f"columns_expected_to_change): {list(unexpected.keys())}. Either this regeneration "
            f"has a real regression, or columns_expected_to_change needs updating to reflect an "
            f"intentional change -- don't silence this by widening the allowlist without checking which."
        )

    if verbose:
        print("-" * 50)
        print("Validation Complete: regeneration diff is fully accounted for.")
        print("=" * 50 + "\n")

    return {
        'new_keys': added_keys,
        'missing_keys': missing_keys,
        'changed_columns': changed_columns,
        'unexpected_changes': bool(unexpected),
    }


# ==============================================================================
# 7. Master ETL Pipeline Function
# ==============================================================================

def run_etl_pipeline(
    scraped_fights_df: pd.DataFrame,
    events_df: pd.DataFrame,
    fighters_df: pd.DataFrame,
    current_dataset: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Master ETL pipeline that processes scraped fight data and merges with event and fighter details.
    """
    # 1. Process fighter biographical features (Height, Weight, Reach)
    fighters_clean = process_fighter_bio(fighters_df)

    # 2. Merge event date into scraped fight data
    events_subset = events_df[['URL', 'DATE']].drop_duplicates(subset=['URL'])
    df = pd.merge(scraped_fights_df, events_subset, left_on='Event URL', right_on='URL', how='left')
    df = df.drop(columns=['URL'], errors='ignore')
    df = df.rename(columns={'DATE': 'Event Date'})
    df = df.loc[:, ~df.columns.duplicated()]

    # 3. Standardize column names
    df = standardize_columns(df, strike=True)

    # 3b. Drop rows the event-date join above failed to match, before any
    # further processing wastes work on rows that can't be safely placed in
    # time anyway. See drop_rows_with_null_event_date's docstring.
    df = drop_rows_with_null_event_date(df)

    # 4. Rearrange columns for row splitting
    # CHANGE: Explicit definition of the exact stats being split, fed directly into the splitter.
    stat_cols = [
        'fighter', 'kd', 'sig_strikes', 'sig_strikes_pct', 'total_strikes',
        'td', 'td_pct', 'sub_att', 'rev', 'ctrl', 'head_strikes', 'body_strikes',
        'leg_strikes', 'distance_strikes', 'clinch_strikes', 'ground_strikes'
    ]
    meta_cols = [
        'fighter_1_name', 'fighter_1_url', 'fighter_1_result',
        'fighter_2_name', 'fighter_2_url', 'fighter_2_result',
        'event_name', 'event_url', 'event_date', 'weight_class', 'fight_url',
        'method', 'details', 'round', 'time', 'time_format'
    ]
    existing_split_cols = [c for c in stat_cols + meta_cols if c in df.columns]
    df_to_be_split = df.loc[:, existing_split_cols].copy()

    # 5. Split multi-line fight stats to two rows (1 per fighter)
    modified_df = split_fight_stats(df_to_be_split, stat_cols=stat_cols)
    modified_df_clean = clean_modified_df(modified_df)

    # 6. Merge fighter bio details (DOB, Height, Weight, Reach, Stance)
    bio_cols = ['URL', 'DOB', 'Height (m)', 'Weight (lbs)', 'Reach (in)', 'Stance']
    bio_subset = fighters_clean[[c for c in bio_cols if c in fighters_clean.columns]].drop_duplicates(subset=['URL'])
    df_with_bio = pd.merge(modified_df_clean, bio_subset, left_on='fighter_url', right_on='URL', how='left')
    df_with_bio = df_with_bio.drop(columns=['URL'], errors='ignore')

    # 7. Split 'X of Y' strikes and takedowns into numeric landed and attempted columns
    columns_to_split = [
        'sig_strikes', 'total_strikes', 'td', 'head_strikes', 'body_strikes',
        'leg_strikes', 'distance_strikes', 'clinch_strikes', 'ground_strikes'
    ]
    df_split = apply_split(df_with_bio, columns_to_split)

    # 8. Calculate fight day age
    df_split = calculate_age(df_split, dob_col='DOB', date_col='event_date')

    # 9. Process control time to seconds
    # CHANGE: Call the new vectorized processing function instead of iterating values via lambda.
    df_split = process_control_time(df_split, col='ctrl')

    # 10. Fix percentage columns to numeric floats
    pct_cols = ['td_pct', 'sig_strikes_pct']
    df_pct_fixed = convert_cols_from_pct_to_float(df_split, pct_cols)

    # 11. Reshape from 2 rows per fight to 1 row per fight
    final_df = convert_to_one_fight_one_row(df_pct_fixed)

    # Validate output schema & values before column ordering
    validate_transformed_data(final_df, current_dataset=current_dataset)

    # 12. Standard desired column ordering
    ordered_columns = [
        'event_name', 'event_url', 'event_date', 'bout', 'fight_url',
        'weight_class', 'time_format', 'method', 'details', 'time', 'round_ended',
        'fighter_1', 'fight_day_age (yrs)_fighter_1', 'fight_result_fighter_1',
        'kd_fighter_1', 'sig_strikes_landed_fighter_1',
        'sig_strikes_attempted_fighter_1', 'sig_strikes_pct_fighter_1',
        'total_strikes_landed_fighter_1', 'total_strikes_attempted_fighter_1',
        'td_landed_fighter_1', 'td_attempted_fighter_1', 'td_pct_fighter_1',
        'head_strikes_landed_fighter_1', 'head_strikes_attempted_fighter_1',
        'body_strikes_landed_fighter_1', 'body_strikes_attempted_fighter_1',
        'leg_strikes_landed_fighter_1', 'leg_strikes_attempted_fighter_1',
        'distance_strikes_landed_fighter_1', 'distance_strikes_attempted_fighter_1',
        'clinch_strikes_landed_fighter_1', 'clinch_strikes_attempted_fighter_1',
        'ground_strikes_landed_fighter_1', 'ground_strikes_attempted_fighter_1',
        'sub_att_fighter_1', 'rev_fighter_1', 'ctrl_in_secs_fighter_1',
        'fighter_url_fighter_1', 'date_of_birth_fighter_1',
        'Height (m)_fighter_1', 'Weight (lbs)_fighter_1', 'Reach (in)_fighter_1',
        'Stance_fighter_1',
        'fighter_2', 'fight_day_age (yrs)_fighter_2', 'fight_result_fighter_2',
        'kd_fighter_2', 'sig_strikes_landed_fighter_2',
        'sig_strikes_attempted_fighter_2', 'sig_strikes_pct_fighter_2',
        'total_strikes_landed_fighter_2', 'total_strikes_attempted_fighter_2',
        'td_landed_fighter_2', 'td_attempted_fighter_2', 'td_pct_fighter_2',
        'head_strikes_landed_fighter_2', 'head_strikes_attempted_fighter_2',
        'body_strikes_landed_fighter_2', 'body_strikes_attempted_fighter_2',
        'leg_strikes_landed_fighter_2', 'leg_strikes_attempted_fighter_2',
        'distance_strikes_landed_fighter_2', 'distance_strikes_attempted_fighter_2',
        'clinch_strikes_landed_fighter_2', 'clinch_strikes_attempted_fighter_2',
        'ground_strikes_landed_fighter_2', 'ground_strikes_attempted_fighter_2',
        'sub_att_fighter_2', 'rev_fighter_2', 'ctrl_in_secs_fighter_2',
        'fighter_url_fighter_2', 'date_of_birth_fighter_2',
        'Height (m)_fighter_2', 'Weight (lbs)_fighter_2', 'Reach (in)_fighter_2',
        'Stance_fighter_2'
    ]
    # Columns not named in ordered_columns are APPENDED, not dropped. By
    # this point in the pipeline every column present was deliberately
    # constructed by an earlier step -- there's no leftover junk left to
    # filter out here, so this list's real job is fixing a readable order
    # for the columns people care about seeing first, not deciding what
    # belongs in the output. A strict whitelist silently dropped a column
    # (STANCE) that a genuine upstream step had already added; appending
    # unlisted columns instead means a future addition shows up (at the
    # end) rather than vanishing if someone forgets to list it here too.
    present_cols = [c for c in ordered_columns if c in final_df.columns]
    remaining_cols = [c for c in final_df.columns if c not in ordered_columns]
    final_df = final_df.loc[:, present_cols + remaining_cols]

    # 13. Optional: merge/append with existing historical dataset
    if current_dataset is not None and not current_dataset.empty:
        curr_df = current_dataset.copy()
        cols_to_drop = set(curr_df.columns) - set(final_df.columns)
        if cols_to_drop:
            curr_df = curr_df.drop(columns=list(cols_to_drop))
        final_df = pd.concat([curr_df, final_df], ignore_index=True)

    # Ensure event_date is datetime and sort descending
    if 'event_date' in final_df.columns:
        final_df['event_date'] = pd.to_datetime(final_df['event_date'], errors='coerce')
        # Re-check for null event_date here too, not just on the freshly-scraped
        # df back in step 3b: current_dataset (just merged in above) may itself
        # carry rows with a null/unparseable event_date from before this check
        # existed, and errors='coerce' just above can turn an unparseable date
        # string into a fresh NaT. Without this, run_etl_pipeline's own output
        # could still contain null-date rows despite step 3b's check having
        # already run -- only appearing clean for freshly-scraped data.
        final_df = drop_rows_with_null_event_date(final_df)
        final_df = final_df.sort_values(by='event_date', ascending=False).reset_index(drop=True)

    return final_df


# if __name__ == "__main__":
#     import argparse

#     parser = argparse.ArgumentParser(description="Run UFC Fight Data ETL Pipeline")
#     parser.add_argument("--fights", required=True, help="Path to scraped fight data CSV")
#     parser.add_argument("--events", required=True, help="Path to events CSV (ufc_event_details.csv)")
#     parser.add_argument("--fighters", required=True, help="Path to fighters bio CSV")
#     parser.add_argument("--current", required=False, default=None, help="Path to existing dataset to append to")
#     parser.add_argument("--output", required=False, default="fights_ready_for_features.csv", help="Output path")

#     args = parser.parse_args()

#     scraped_df = pd.read_csv(args.fights)
#     events_df = pd.read_csv(args.events)
#     fighters_df = pd.read_csv(args.fighters)
#     curr_df = pd.read_csv(args.current) if args.current else None

#     processed_df = run_etl_pipeline(
#         scraped_fights_df=scraped_df,
#         events_df=events_df,
#         fighters_df=fighters_df,
#         current_dataset=curr_df
#     )

#     processed_df.to_csv(args.output, index=False)
#     print(f"ETL completed successfully! Output saved to: {args.output}")
