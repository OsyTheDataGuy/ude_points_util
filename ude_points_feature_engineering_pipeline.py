# -*- coding: utf-8 -*-
"""# 5. Feature Engineering for ude points - Dictionary Batched"""

import pandas as pd
import numpy as np

from ude_points_algorithm import quality_score as _quality_score_fn
from dataset_processing_pipeline import drop_rows_with_null_event_date

"""## 1. Create is_title_bout column"""

def create_is_title_bout_column(df, weight_class_col='weight_class'):
    """Create 'is_title_bout' column using batch dictionary concatenation."""
    s = df[weight_class_col].fillna('').astype(str)

    # Define boolean masks for explicit conditions
    is_title = s.str.contains('Title Bout')
    is_interim = s.str.contains('Interim')
    is_tournament = s.str.contains('Tournament')

    # Evaluate conditions: must be a Title/Interim bout AND NOT a Tournament
    is_tb = np.where(is_title & ~is_tournament, 2,
            np.where(is_interim & ~is_tournament, 1, 0))

    new_cols = {'is_title_bout': is_tb}
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

"""## 2. Fix the weight class column"""

def map_weight_class(weight_class):
    """Map weight classes to their corresponding codes."""
    if not isinstance(weight_class, str):
        return weight_class

    weight_classes = {
        'Light Heavyweight': 'LHW',
        'Heavyweight': 'HW',
        'Middleweight': 'MW',
        'Welterweight': 'WW',
        'Lightweight': 'LW',
        'Featherweight': 'FW',
        'Bantamweight': 'BW',
        'Flyweight': 'FLW',
        'Strawweight': 'SW'
    }

    # Longest name first: 'Heavyweight' is a literal substring of 'Light
    # Heavyweight', so a shorter name must never get a chance to match before
    # a longer one that contains it. Sorting removes the dependency on dict
    # insertion order (previously the only thing preventing a misclassification).
    for class_name, code in sorted(weight_classes.items(), key=lambda kv: -len(kv[0])):
        if class_name in weight_class:
            return 'W' + code if 'Women' in weight_class else code

    return weight_class

"""## 3. Create is_champion column"""

def update_champion_status(df):
    """
    Update champion status for fighters sequentially.
    Assumes chronological sort (ascending=True).
    """
    champ_1, champ_2 = [], []
    fighter_champions = {}

    for row in df.itertuples(index=False):
        is_tb = getattr(row, 'is_title_bout', 0)
        wc = getattr(row, 'weight_class_cleaned', '')
        f1, f2 = row.fighter_1, row.fighter_2
        r1, r2 = row.fight_result_fighter_1, row.fight_result_fighter_2

        if f1 not in fighter_champions: fighter_champions[f1] = {}
        if wc not in fighter_champions[f1]: fighter_champions[f1][wc] = {'status': 0}
        if f2 not in fighter_champions: fighter_champions[f2] = {}
        if wc not in fighter_champions[f2]: fighter_champions[f2][wc] = {'status': 0}

        c1_status = fighter_champions[f1][wc]['status']
        c2_status = fighter_champions[f2][wc]['status']

        champ_1.append(c1_status)
        champ_2.append(c2_status)

        # Update status after capturing current fight status
        if is_tb > 0:
            if r1 == 'W':
                fighter_champions[f1][wc]['status'] = 1 if is_tb == 1 else 2
            elif r1 == 'L':
                fighter_champions[f1][wc]['status'] = 0

            if r2 == 'W':
                fighter_champions[f2][wc]['status'] = 1 if is_tb == 1 else 2
            elif r2 == 'L':
                fighter_champions[f2][wc]['status'] = 0

    new_cols = {
        'is_champion_fighter_1': champ_1,
        'is_champion_fighter_2': champ_2
    }
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

"""## 4. Title defenses"""

def update_title_defenses(df):
    """
    Update title defenses for fighters based on fight outcomes.
    Assumes chronological sort (ascending=True).
    """
    def_1, def_2 = [], []
    fighter_defenses = {}

    for row in df.itertuples(index=False):
        is_tb = getattr(row, 'is_title_bout', 0)
        wc = getattr(row, 'weight_class_cleaned', '')
        f1, f2 = row.fighter_1, row.fighter_2
        r1, r2 = row.fight_result_fighter_1, row.fight_result_fighter_2
        champ1 = getattr(row, 'is_champion_fighter_1', 0)
        champ2 = getattr(row, 'is_champion_fighter_2', 0)

        if f1 not in fighter_defenses: fighter_defenses[f1] = {}
        if wc not in fighter_defenses[f1]: fighter_defenses[f1][wc] = 0
        if f2 not in fighter_defenses: fighter_defenses[f2] = {}
        if wc not in fighter_defenses[f2]: fighter_defenses[f2][wc] = 0

        cur_def1 = fighter_defenses[f1][wc]
        cur_def2 = fighter_defenses[f2][wc]

        def_1.append(cur_def1)
        def_2.append(cur_def2)

        if champ1 > 0:
            if r1 == 'W' and is_tb > 0:
                fighter_defenses[f1][wc] += 1
            elif r1 == 'L' and is_tb > 0:
                fighter_defenses[f1][wc] = 0

        if champ2 > 0:
            if r2 == 'W' and is_tb > 0:
                fighter_defenses[f2][wc] += 1
            elif r2 == 'L' and is_tb > 0:
                fighter_defenses[f2][wc] = 0

    new_cols = {
        'title_defenses_fighter_1': def_1,
        'title_defenses_fighter_2': def_2
    }
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

"""## 5. UFC record"""

def initialize_fighter_record():
    return {'W': 0, 'L': 0, 'D': 0, 'NC': 0}

def update_fighter_record(record, fight_result):
    if fight_result in record:
        record[fight_result] += 1
    return record

def format_record(record):
    return f"{record['W']}-{record['L']}-{record['D']} {record['NC']}"

def update_fight_records(df):
    pre_1, post_1, pre_2, post_2 = [], [], [], []
    fighter_records = {}

    for row in df.itertuples(index=False):
        f1, f2 = row.fighter_1, row.fighter_2
        r1, r2 = row.fight_result_fighter_1, row.fight_result_fighter_2

        if f1 not in fighter_records: fighter_records[f1] = initialize_fighter_record()
        if f2 not in fighter_records: fighter_records[f2] = initialize_fighter_record()

        p1_pre = format_record(fighter_records[f1])
        fighter_records[f1] = update_fighter_record(fighter_records[f1], r1)
        p1_post = format_record(fighter_records[f1])

        p2_pre = format_record(fighter_records[f2])
        fighter_records[f2] = update_fighter_record(fighter_records[f2], r2)
        p2_post = format_record(fighter_records[f2])

        pre_1.append(p1_pre)
        post_1.append(p1_post)
        pre_2.append(p2_pre)
        post_2.append(p2_post)

    new_cols = {
        'pre_fight_record_fighter_1_(W-L-D NC)': pre_1,
        'post_fight_record_fighter_1_(W-L-D NC)': post_1,
        'pre_fight_record_fighter_2_(W-L-D NC)': pre_2,
        'post_fight_record_fighter_2_(W-L-D NC)': post_2
    }
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

"""## 5b. Opponent quality score"""

def add_quality_score_columns(df):
    """
    Adds quality_score_fighter_1/quality_score_fighter_2: each fighter's
    pre-fight "opponent quality" score, as consumed both by the UDE scoring
    loop's opponent-quality adjustment and by the temporal method x PDI
    calibration in ude_points_algorithm.

    Why this needs to live here (in feature engineering) rather than only
    being written at UDE-scoring time:

    ude_points_algorithm's temporal method x PDI calibration
    (_build_temporal_calibration_cache -> calibrate_method_pdi_effects ->
    _build_future_performance_observations) builds its per-year calibration
    cache *before* any UDE points are scored. It reads
    `quality_score_fighter_{opponent_side}` directly off `df` while scanning
    fight history. But historically that column was only ever populated
    incrementally, mid-loop, by the scoring pass itself
    (`df.at[index, f'quality_score_{opponent_col}'] = ...`) -- so at
    calibration time the column didn't exist yet, `row.get(...)` fell back
    to NaN for every row, and every observation was subsequently dropped by
    `dropna(subset=['opponent_quality', ...])`. Calibration therefore never
    had enough (or any) usable observations, regardless of how much fight
    history was actually available.

    quality_score is a pure function of a fighter's own PRE-fight state
    (pre-fight record, current champion status, title-defense count) --
    fields already produced earlier in this pipeline -- so computing it
    here introduces no leakage: quality_score_fighter_1 for a given row
    depends only on information already known strictly before that fight.

    Must run after update_champion_status, update_title_defenses, and
    update_fight_records (needs is_champion_*, title_defenses_*, and
    pre_fight_record_*_(W-L-D NC) to already exist).
    """
    new_cols = {}
    for f in ['fighter_1', 'fighter_2']:
        new_cols[f'quality_score_{f}'] = df.apply(
            lambda row: _quality_score_fn(
                row[f'pre_fight_record_{f}_(W-L-D NC)'],
                row[f'is_champion_{f}'],
                row[f'title_defenses_{f}'],
            ),
            axis=1,
        )
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

"""## 6. Win streaks"""

def update_win_streaks(df):
    streak_1, streak_2 = [], []
    fighter_streaks = {}

    for row in df.itertuples(index=False):
        f1, f2 = row.fighter_1, row.fighter_2
        r1, r2 = row.fight_result_fighter_1, row.fight_result_fighter_2

        if f1 not in fighter_streaks: fighter_streaks[f1] = 0
        if f2 not in fighter_streaks: fighter_streaks[f2] = 0

        cur_s1 = fighter_streaks[f1]
        cur_s2 = fighter_streaks[f2]

        streak_1.append(cur_s1)
        streak_2.append(cur_s2)

        if r1 == 'W': fighter_streaks[f1] = cur_s1 + 1 if cur_s1 >= 0 else 1
        elif r1 == 'D': fighter_streaks[f1] = 0
        elif r1 == 'NC': pass
        elif r1 == 'L': fighter_streaks[f1] = cur_s1 - 1 if cur_s1 <= 0 else -1

        if r2 == 'W': fighter_streaks[f2] = cur_s2 + 1 if cur_s2 >= 0 else 1
        elif r2 == 'D': fighter_streaks[f2] = 0
        elif r2 == 'NC': pass
        elif r2 == 'L': fighter_streaks[f2] = cur_s2 - 1 if cur_s2 <= 0 else -1

    new_cols = {
        'W/L_streak_fighter_1': streak_1,
        'W/L_streak_fighter_2': streak_2
    }
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

"""## 7. Career striking and takedown accuracy"""

def update_career_means(df):
    c_sig_acc_1, c_sig_acc_2 = [], []
    c_sig_mean_1, c_sig_mean_2 = [], []
    c_td_mean_1, c_td_mean_2 = [], []
    c_td_acc_1, c_td_acc_2 = [], []

    fighter_stats = {}

    for row in df.itertuples(index=False):
        f1, f2 = row.fighter_1, row.fighter_2

        for f in [f1, f2]:
            if f not in fighter_stats:
                fighter_stats[f] = {
                    'sig_strikes_landed': 0,
                    'sig_strikes_attempted': 0,
                    'td_landed': 0,
                    'td_attempted': 0,
                    'fight_count': 0
                }

        curr1 = fighter_stats[f1]
        sig_acc1 = (curr1['sig_strikes_landed'] / curr1['sig_strikes_attempted'] * 100) if curr1['sig_strikes_attempted'] > 0 else 0.0
        td_acc1 = (curr1['td_landed'] / curr1['td_attempted'] * 100) if curr1['td_attempted'] > 0 else 0.0
        fc1 = curr1['fight_count']
        sig_mean1 = (curr1['sig_strikes_landed'] / fc1) if fc1 > 0 else 0.0
        td_mean1 = (curr1['td_landed'] / fc1) if fc1 > 0 else 0.0

        curr2 = fighter_stats[f2]
        sig_acc2 = (curr2['sig_strikes_landed'] / curr2['sig_strikes_attempted'] * 100) if curr2['sig_strikes_attempted'] > 0 else 0.0
        td_acc2 = (curr2['td_landed'] / curr2['td_attempted'] * 100) if curr2['td_attempted'] > 0 else 0.0
        fc2 = curr2['fight_count']
        sig_mean2 = (curr2['sig_strikes_landed'] / fc2) if fc2 > 0 else 0.0
        td_mean2 = (curr2['td_landed'] / fc2) if fc2 > 0 else 0.0

        c_sig_acc_1.append(sig_acc1)
        c_sig_mean_1.append(sig_mean1)
        c_td_mean_1.append(td_mean1)
        c_td_acc_1.append(td_acc1)

        c_sig_acc_2.append(sig_acc2)
        c_sig_mean_2.append(sig_mean2)
        c_td_mean_2.append(td_mean2)
        c_td_acc_2.append(td_acc2)

        curr1['sig_strikes_landed'] += getattr(row, 'sig_strikes_landed_fighter_1', 0)
        curr1['sig_strikes_attempted'] += getattr(row, 'sig_strikes_attempted_fighter_1', 0)
        curr1['td_landed'] += getattr(row, 'td_landed_fighter_1', 0)
        curr1['td_attempted'] += getattr(row, 'td_attempted_fighter_1', 0)
        curr1['fight_count'] += 1

        curr2['sig_strikes_landed'] += getattr(row, 'sig_strikes_landed_fighter_2', 0)
        curr2['sig_strikes_attempted'] += getattr(row, 'sig_strikes_attempted_fighter_2', 0)
        curr2['td_landed'] += getattr(row, 'td_landed_fighter_2', 0)
        curr2['td_attempted'] += getattr(row, 'td_attempted_fighter_2', 0)
        curr2['fight_count'] += 1

    new_cols = {
        'career_sig_strikes_landed_fighter_1 (mean)': c_sig_mean_1,
        'career_sig_strikes_landed_fighter_2 (mean)': c_sig_mean_2,
        'career_sig_striking_accuracy_fighter_1': c_sig_acc_1,
        'career_sig_striking_accuracy_fighter_2': c_sig_acc_2,
        'career_td_landed_fighter_1 (mean)': c_td_mean_1,
        'career_td_landed_fighter_2 (mean)': c_td_mean_2,
        'career_td_accuracy_fighter_1': c_td_acc_1,
        'career_td_accuracy_fighter_2': c_td_acc_2
    }
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

"""## 8. Fix the method column"""

def map_fight_method(result):
    mapping = {
        'Decision - Unanimous': 'UD',
        'Decision - Majority': 'MD',
        'Decision - Split': 'SD',
        'KO/TKO': 'Finish',
        'TKO - Doctor\'s Stoppage': 'Finish',
        'Submission': 'Finish'
    }
    return mapping.get(result, result)

"""## 9. Create rematch column"""

def add_rematch_features(df):
    pairs = [tuple(sorted([f1, f2])) for f1, f2 in zip(df['fighter_1'], df['fighter_2'])]
    pair_series = pd.Series(pairs, index=df.index)
    rematch_col = pair_series.groupby(pair_series).cumcount()
    is_rematch = (rematch_col > 0).astype(int)

    new_cols = {
        'rematch_column': rematch_col,
        'is_rematch': is_rematch
    }
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

"""## 10. Static and Dynamic Accuracy / Defence"""

def add_defense_columns(df):
    new_cols = {}
    for fighter_col, opponent_col in [('fighter_1', 'fighter_2'), ('fighter_2', 'fighter_1')]:
        opt_att_str = df[f'sig_strikes_attempted_{opponent_col}']
        opt_land_str = df[f'sig_strikes_landed_{opponent_col}']
        new_cols[f'sig_strikes_defense_{fighter_col}'] = np.where(
            opt_att_str == 0,
            np.nan,
            np.round((opt_att_str - opt_land_str) / opt_att_str * 100, 2)
        )

        opt_att_td = df[f'td_attempted_{opponent_col}']
        opt_land_td = df[f'td_landed_{opponent_col}']
        new_cols[f'td_defense_{fighter_col}'] = np.where(
            opt_att_td == 0,
            np.nan,
            np.round((opt_att_td - opt_land_td) / opt_att_td * 100, 2)
        )

    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

def add_dynamic_strike_accuracy(df):
    strike_types = ['sig', 'head', 'body', 'leg']
    new_cols = {f'dynamic_{st}_strikes_accuracy_{f}': [] for st in strike_types for f in ['fighter_1', 'fighter_2']}
    
    cumulative_stats = {st: {} for st in strike_types}

    for row in df.itertuples(index=False):
        for f_col in ['fighter_1', 'fighter_2']:
            f_url = getattr(row, f'fighter_url_{f_col}')
            for st in strike_types:
                landed = getattr(row, f'{st}_strikes_landed_{f_col}')
                attempted = getattr(row, f'{st}_strikes_attempted_{f_col}')

                if f_url not in cumulative_stats[st]:
                    cumulative_stats[st][f_url] = {'landed': 0, 'attempted': 0}

                c_landed = cumulative_stats[st][f_url]['landed']
                c_attempted = cumulative_stats[st][f_url]['attempted']
                
                curr_acc = np.nan if c_attempted == 0 else round(c_landed / c_attempted, 3)
                new_cols[f'dynamic_{st}_strikes_accuracy_{f_col}'].append(curr_acc)

                cumulative_stats[st][f_url]['landed'] += landed
                cumulative_stats[st][f_url]['attempted'] += attempted

    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

def add_dynamic_strike_defence(df):
    strike_types = ['sig', 'head', 'body', 'leg']
    new_cols = {f'dynamic_{st}_strikes_defence_{f}': [] for st in strike_types for f in ['fighter_1', 'fighter_2']}
    
    cumulative_defence_stats = {st: {} for st in strike_types}

    for row in df.itertuples(index=False):
        for f_col, opp_col in [('fighter_1', 'fighter_2'), ('fighter_2', 'fighter_1')]:
            f_url = getattr(row, f'fighter_url_{f_col}')
            for st in strike_types:
                opp_landed = getattr(row, f'{st}_strikes_landed_{opp_col}')
                opp_attempted = getattr(row, f'{st}_strikes_attempted_{opp_col}')

                if f_url not in cumulative_defence_stats[st]:
                    cumulative_defence_stats[st][f_url] = {'faced': 0, 'avoided': 0}

                c_faced = cumulative_defence_stats[st][f_url]['faced']
                c_avoided = cumulative_defence_stats[st][f_url]['avoided']

                curr_def = np.nan if c_faced == 0 else round(c_avoided / c_faced, 3)
                new_cols[f'dynamic_{st}_strikes_defence_{f_col}'].append(curr_def)

                avoided_in_fight = max(0, opp_attempted - opp_landed)
                cumulative_defence_stats[st][f_url]['faced'] += opp_attempted
                cumulative_defence_stats[st][f_url]['avoided'] += avoided_in_fight

    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

def add_dynamic_td_accuracy(df):
    td_acc_1, td_acc_2 = [], []
    cumulative_stats = {}

    for row in df.itertuples(index=False):
        for f_col in ['fighter_1', 'fighter_2']:
            f_url = getattr(row, f'fighter_url_{f_col}')
            td_landed = getattr(row, f'td_landed_{f_col}')
            td_attempted = getattr(row, f'td_attempted_{f_col}')

            if f_url not in cumulative_stats:
                cumulative_stats[f_url] = {'landed': 0, 'attempted': 0}

            c_landed = cumulative_stats[f_url]['landed']
            c_attempted = cumulative_stats[f_url]['attempted']

            acc = np.nan if c_attempted == 0 else round(c_landed / c_attempted, 3)
            if f_col == 'fighter_1':
                td_acc_1.append(acc)
            else:
                td_acc_2.append(acc)

            cumulative_stats[f_url]['landed'] += td_landed
            cumulative_stats[f_url]['attempted'] += td_attempted

    new_cols = {
        'dynamic_td_accuracy_fighter_1': td_acc_1,
        'dynamic_td_accuracy_fighter_2': td_acc_2
    }
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

def add_dynamic_td_defence(df):
    td_def_1, td_def_2 = [], []
    cumulative_defence_stats = {}

    for row in df.itertuples(index=False):
        for f_col, opp_col in [('fighter_1', 'fighter_2'), ('fighter_2', 'fighter_1')]:
            f_url = getattr(row, f'fighter_url_{f_col}')
            opp_td_landed = getattr(row, f'td_landed_{opp_col}')
            opp_td_attempted = getattr(row, f'td_attempted_{opp_col}')

            if f_url not in cumulative_defence_stats:
                cumulative_defence_stats[f_url] = {'faced': 0, 'avoided': 0}

            c_faced = cumulative_defence_stats[f_url]['faced']
            c_avoided = cumulative_defence_stats[f_url]['avoided']

            def_val = np.nan if c_faced == 0 else round(c_avoided / c_faced, 3)
            if f_col == 'fighter_1':
                td_def_1.append(def_val)
            else:
                td_def_2.append(def_val)

            avoided_in_fight = max(0, opp_td_attempted - opp_td_landed)
            cumulative_defence_stats[f_url]['faced'] += opp_td_attempted
            cumulative_defence_stats[f_url]['avoided'] += avoided_in_fight

    new_cols = {
        'dynamic_td_defence_fighter_1': td_def_1,
        'dynamic_td_defence_fighter_2': td_def_2
    }
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

"""## 11. Time calculations & per-minute rates"""

def time_to_minutes(time_str):
    try:
        minutes, seconds = map(int, str(time_str).split(':'))
        return minutes + (1 if seconds > 0 else 0)
    except Exception:
        return np.nan

def add_time_and_per_min_features(df):
    new_cols = {}
    
    if 'time' in df.columns:
        time_mins = df['time'].apply(time_to_minutes)
        new_cols['time_in_mins'] = time_mins
    else:
        time_mins = df['time_in_mins']

    if 'time_format' in df.columns:
        new_cols['match_format_rounds'] = df['time_format'].astype(str).str.extract(r'^(\d{1})')[0]

    round_ended = pd.to_numeric(df['round_ended'], errors='coerce')
    total_time = ((round_ended - 1) * 5) + time_mins
    total_time = total_time.replace(0, np.nan)
    new_cols['total_time_in_mins'] = total_time

    for col in list(df.columns):
        if ('strikes_landed' in col or 'strikes_attempted' in col) and ('career' not in col and 'diff' not in col and 'per_min' not in col):
            per_min_col = col.replace('_fighter', '_per_min_fighter')
            new_cols[per_min_col] = round(df[col] / total_time, 2)

        if 'strikes_landed' in col and ('career' not in col and 'diff' not in col and 'per_min' not in col):
            if 'fighter_1' in col:
                absorbed_col = col.replace('strikes_landed_fighter_1', 'strikes_absorbed_per_min_fighter_2')
                new_cols[absorbed_col] = round(df[col] / total_time, 2)
            elif 'fighter_2' in col:
                absorbed_col = col.replace('strikes_landed_fighter_2', 'strikes_absorbed_per_min_fighter_1')
                new_cols[absorbed_col] = round(df[col] / total_time, 2)

    for f in ['fighter_1', 'fighter_2']:
        new_cols[f'td_landed_per_15_minutes_{f}'] = round((df[f'td_landed_{f}'] / total_time) * 15, 2)

    new_cols['td_conceded_per_15_minutes_fighter_1'] = round((df['td_landed_fighter_2'] / total_time) * 15, 2)
    new_cols['td_conceded_per_15_minutes_fighter_2'] = round((df['td_landed_fighter_1'] / total_time) * 15, 2)

    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

"""## 12. Standing significant strikes"""

def add_standing_sig_strikes_columns(df):
    new_cols = {}
    for fighter in ['fighter_1', 'fighter_2']:
        new_cols[f'standing_sig_strikes_landed_{fighter}'] = df[f'distance_strikes_landed_{fighter}'] + df[f'clinch_strikes_landed_{fighter}']
        new_cols[f'standing_sig_strikes_attempted_{fighter}'] = df[f'distance_strikes_attempted_{fighter}'] + df[f'clinch_strikes_attempted_{fighter}']
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

"""## 13. Dominance Differentials & Phase Performance"""

def process_dominance_differentials(df, stat_prefixes=['sig_strikes_landed', 'head_strikes_landed', 'standing_sig_strikes_landed', 'kd', 'td_landed', 'ctrl_in_secs', 'ground_strikes_landed', 'sub_att']):
    new_cols = {}
    for prefix in stat_prefixes:
        s1 = pd.to_numeric(df[f'{prefix}_fighter_1'].astype(str).str.strip(), errors='coerce')
        s2 = pd.to_numeric(df[f'{prefix}_fighter_2'].astype(str).str.strip(), errors='coerce')
        new_cols[f'{prefix}_diff_fighter_1'] = s1 - s2
        new_cols[f'{prefix}_diff_fighter_2'] = s2 - s1
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

def add_who_won_col(df):
    new_cols = {}
    s1, s2 = df['standing_sig_strikes_landed_diff_fighter_1'], df['standing_sig_strikes_landed_diff_fighter_2']
    new_cols['who_won_striking'] = np.where(s1 > 0, df['fighter_1'], np.where(s2 > 0, df['fighter_2'], 'No Difference'))

    w1 = (df['td_landed_diff_fighter_1'] > 0) & (df['ground_strikes_landed_diff_fighter_1'] > 0)
    w2 = (df['td_landed_diff_fighter_2'] > 0) & (df['ground_strikes_landed_diff_fighter_2'] > 0)
    new_cols['who_won_wrestling'] = np.where(w1, df['fighter_1'], np.where(w2, df['fighter_2'], 'No Difference'))

    g1, g2 = df['sub_att_diff_fighter_1'], df['sub_att_diff_fighter_2']
    new_cols['who_won_grappling'] = np.where(g1 > 0, df['fighter_1'], np.where(g2 > 0, df['fighter_2'], 'No Difference'))

    c1, c2 = df['ctrl_in_secs_diff_fighter_1'], df['ctrl_in_secs_diff_fighter_2']
    new_cols['who_won_control'] = np.where(c1 > 0, df['fighter_1'], np.where(c2 > 0, df['fighter_2'], 'No Difference'))

    d1, d2 = df['kd_diff_fighter_1'], df['kd_diff_fighter_2']
    new_cols['who_won_standing_danger'] = np.where(d1 > 0, df['fighter_1'], np.where(d2 > 0, df['fighter_2'], 'No Difference'))

    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

def add_dominance_columns(df):
    phases = ['who_won_grappling', 'who_won_striking', 'who_won_wrestling', 'who_won_control', 'who_won_standing_danger']
    dom_fighters, phases_wons = [], []

    for row in df.itertuples(index=False):
        f1, f2 = row.fighter_1, row.fighter_2
        f1_wins = sum(getattr(row, phase) == f1 for phase in phases)
        f2_wins = sum(getattr(row, phase) == f2 for phase in phases)

        if f1_wins >= 3 and f1_wins > f2_wins:
            dom_f = f1
        elif f2_wins >= 3 and f2_wins > f1_wins:
            dom_f = f2
        else:
            dom_f = None

        dom_fighters.append(dom_f)
        phases_wons.append(max(f1_wins, f2_wins))

    new_cols = {
        'dominant_fighter': dom_fighters,
        'phases_won': phases_wons
    }
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

def _dominance_magnitude(landed_for, landed_against, count_threshold, prop_threshold,
                          count_width=0.10, prop_width=0.05):
    """
    Magnitude for a landed-count phase (takedowns, submission attempts):
    blends a "close" curve (capped at 0.35) and a "decisive" curve (starting
    at 0.36) by a continuous decisive_weight in [0, 1], instead of switching
    between them at a hard (count > count_threshold and proportion >=
    prop_threshold) gate. decisive_weight is a soft AND (product of two
    sigmoids, one per condition), so it only approaches 1 when BOTH the
    count and the proportion clear their threshold.

    count_width is deliberately tight (0.10): landed counts are integers, so
    there's nothing to interpolate between them -- a tight width preserves
    the original intent that 1-3 landed (whatever the proportion) stays
    capped near 0.35, rather than smoothing that protection away. The actual
    value this function adds over the old hard gate is in the proportion
    dimension (genuinely continuous) and in removing the 0.35/0.36 value gap
    that existed at the boundary even for infinitesimally close inputs.
    """
    diff = landed_for - landed_against
    total = landed_for + landed_against + 1e-5
    proportion = landed_for / total if total > 1e-5 else 0.5

    close = min(0.35, 0.10 + (diff / max(1, landed_for)) * 0.25)
    decisive = min(1.0, 0.36 + (diff / total) * 0.64)

    count_w = 1.0 / (1.0 + np.exp(-(landed_for - count_threshold) / count_width))
    prop_w = 1.0 / (1.0 + np.exp(-(proportion - prop_threshold) / prop_width))
    decisive_weight = count_w * prop_w

    return (1.0 - decisive_weight) * close + decisive_weight * decisive


def calculate_phase_magnitude_and_pdi(row, phases):
    STRIKING_VOLUME_FLOOR = 15

    # Helper function to enforce float conversion safely
    def _to_num(val):
        try:
            v = float(val)
            return 0.0 if np.isnan(v) else v
        except (ValueError, TypeError):
            return 0.0

    s_diff_1 = _to_num(row.get('standing_sig_strikes_landed_diff_fighter_1', 0))
    s_tot_1 = _to_num(row.get('standing_sig_strikes_landed_fighter_1', 0)) + _to_num(row.get('standing_sig_strikes_landed_fighter_2', 0))
    s_mag_raw = max(-1.0, min(1.0, s_diff_1 / (s_tot_1 + 1e-5)))
    s_mag = max(-0.35, min(0.35, s_mag_raw)) if s_tot_1 < STRIKING_VOLUME_FLOOR else s_mag_raw

    CONTROL_VOLUME_FLOOR = 12
    c_diff_1 = _to_num(row.get('ctrl_in_secs_diff_fighter_1', 0))
    c_tot_1 = _to_num(row.get('ctrl_in_secs_fighter_1', 0)) + _to_num(row.get('ctrl_in_secs_fighter_2', 0))
    c_mag_raw = max(-1.0, min(1.0, c_diff_1 / (c_tot_1 + 1e-5)))
    c_mag = max(-0.35, min(0.35, c_mag_raw)) if c_tot_1 < CONTROL_VOLUME_FLOOR else c_mag_raw

    td_landed_1 = _to_num(row.get('td_landed_fighter_1', 0))
    td_landed_2 = _to_num(row.get('td_landed_fighter_2', 0))

    if td_landed_1 > td_landed_2:
        td_mag = _dominance_magnitude(td_landed_1, td_landed_2, count_threshold=3.5, prop_threshold=0.80)
    elif td_landed_2 > td_landed_1:
        td_mag = -_dominance_magnitude(td_landed_2, td_landed_1, count_threshold=3.5, prop_threshold=0.80)
    else:
        td_mag = 0.0

    sub_att_1 = _to_num(row.get('sub_att_fighter_1', 0))
    sub_att_2 = _to_num(row.get('sub_att_fighter_2', 0))
    sub_diff_1 = sub_att_1 - sub_att_2

    if sub_att_1 > sub_att_2:
        sub_mag = _dominance_magnitude(sub_att_1, sub_att_2, count_threshold=2.5, prop_threshold=0.75)
    elif sub_att_2 > sub_att_1:
        sub_mag = -_dominance_magnitude(sub_att_2, sub_att_1, count_threshold=2.5, prop_threshold=0.75)
    else:
        sub_mag = 0.0

    kd_diff_1 = _to_num(row.get('kd_diff_fighter_1', 0))
    kd_mag = max(-1.0, min(1.0, kd_diff_1 / (abs(kd_diff_1) + 1.0)))

    magnitudes_1 = [s_mag, c_mag, td_mag, sub_mag, kd_mag]

    # Classification uses a ROUNDED copy, separate from the raw magnitudes
    # that feed pdi_1/pdi_2 below. _dominance_magnitude's soft blend
    # (td_mag/sub_mag) can never return EXACTLY its 0.35 "close" cap for a
    # shutout (0 landed against) -- decisive_weight is a product of two
    # sigmoids, which approach but never mathematically reach 0, so a
    # landed count safely below its count_threshold can still leak a
    # fractional decisive contribution (e.g. 1 landed vs 0, threshold 3.5,
    # evaluates to 0.35000000000886, not 0.35 -- 12 decimal places of
    # sigmoid noise, not a real signal, previously misclassifying a close
    # win as decisive). Rounding to the same 3-decimal precision pdi_margin
    # itself uses elsewhere in this function makes that noise vanish before
    # classification, while genuine near-threshold signal (2 landed vs 0
    # against a 2.5 threshold legitimately evaluates to 0.354) survives
    # rounding and still correctly reads as decisive.
    classification_magnitudes_1 = [round(m, 3) for m in magnitudes_1]

    decisive_wins_1 = sum(1 for m in classification_magnitudes_1 if m > 0.35)
    close_wins_1 = sum(1 for m in classification_magnitudes_1 if 0.0 < m <= 0.35)
    ties = sum(1 for m in classification_magnitudes_1 if m == 0.0)
    close_losses_1 = sum(1 for m in classification_magnitudes_1 if -0.35 <= m < 0.0)
    decisive_losses_1 = sum(1 for m in classification_magnitudes_1 if m < -0.35)

    decisive_wins_2 = sum(1 for m in classification_magnitudes_1 if m < -0.35)
    close_wins_2 = sum(1 for m in classification_magnitudes_1 if -0.35 <= m < 0.0)
    close_losses_2 = sum(1 for m in classification_magnitudes_1 if 0.0 < m <= 0.35)
    decisive_losses_2 = sum(1 for m in classification_magnitudes_1 if m > 0.35)

    pdi_1 = sum(max(0.0, m) for m in magnitudes_1)
    pdi_2 = sum(max(0.0, -m) for m in magnitudes_1)

    pdi_margin_1 = round(pdi_1 - pdi_2, 3)
    pdi_margin_2 = round(pdi_2 - pdi_1, 3)

    total_output_diff = s_diff_1 + (c_diff_1 / 60.0) + (kd_diff_1 * 10) + (sub_diff_1 * 5)
    magnitude_score = round(abs(total_output_diff) / (abs(total_output_diff) + 50.0), 3)

    return pd.Series({
        'pdi_fighter_1': round(pdi_1, 3),
        'pdi_fighter_2': round(pdi_2, 3),
        'pdi_margin_fighter_1': pdi_margin_1,
        'pdi_margin_fighter_2': pdi_margin_2,
        'fight_magnitude_score': magnitude_score,
        'decisive_wins_fighter_1': decisive_wins_1,
        'close_wins_fighter_1': close_wins_1,
        'close_losses_fighter_1': close_losses_1,
        'decisive_losses_fighter_1': decisive_losses_1,
        'decisive_wins_fighter_2': decisive_wins_2,
        'close_wins_fighter_2': close_wins_2,
        'close_losses_fighter_2': close_losses_2,
        'decisive_losses_fighter_2': decisive_losses_2,
        'ties': ties
    })

def add_pdi_columns(df):
    phases = ['who_won_grappling', 'who_won_striking', 'who_won_wrestling', 'who_won_control', 'who_won_standing_danger']
    pdi_df = df.apply(lambda row: calculate_phase_magnitude_and_pdi(row, phases), axis=1)
    return pd.concat([df, pdi_df], axis=1)

"""## 14. Phase-Level Fighter Strength/Weakness Profiles (added Aug 2026)"""

# Twelve phases spanning striking (by target), grappling, power, and finishing
# tendency. Deliberately NOT collapsed into one blended "skill" number (see
# discussion notes) -- a single scalar erases exactly the thing this feature
# exists to show, e.g. an elite wrestler's control dominance disappearing into
# an average of his middling striking numbers. Built entirely from columns
# already present in the raw scrape (head/body/leg/ground strikes, td, ctrl_in_secs,
# sub_att, kd, method) -- no new data collection required.
PHASE_METRICS = ['head_strikes_acc', 'body_strikes_acc', 'leg_strikes_acc', 'leg_strikes_volume',
                  'td_offense', 'td_defense', 'control', 'ground_strikes', 'submission_threat',
                  'knockdown_power', 'ko_finish_rate', 'sub_finish_rate']

def _init_phase_state():
    return dict(
        head_landed=0.0, head_att=0.0, body_landed=0.0, body_att=0.0,
        leg_landed=0.0, leg_att=0.0, leg_time=0.0,
        td_landed=0.0, td_att=0.0,
        td_faced=0.0, td_avoided=0.0,
        ctrl=0.0, ctrl_time=0.0,
        gs_landed=0.0, gs_time=0.0,
        sub_att=0.0, sub_time=0.0,
        kd=0.0, kd_time=0.0,
        wins=0, ko_wins=0, sub_wins=0,
    )

def _phase_snapshot(s):
    def ratio(n, d):
        return n / d if d > 0 else np.nan
    return {
        'head_strikes_acc': ratio(s['head_landed'], s['head_att']),
        'body_strikes_acc': ratio(s['body_landed'], s['body_att']),
        'leg_strikes_acc': ratio(s['leg_landed'], s['leg_att']),
        'leg_strikes_volume': ratio(s['leg_landed'], s['leg_time']),
        'td_offense': ratio(s['td_landed'], s['td_att']),
        'td_defense': ratio(s['td_avoided'], s['td_faced']),
        'control': ratio(s['ctrl'], s['ctrl_time']),
        'ground_strikes': ratio(s['gs_landed'], s['gs_time']),
        'submission_threat': ratio(s['sub_att'], s['sub_time']) * 15 if s['sub_time'] > 0 else np.nan,
        'knockdown_power': ratio(s['kd'], s['kd_time']) * 15 if s['kd_time'] > 0 else np.nan,
        'ko_finish_rate': ratio(s['ko_wins'], s['wins']),
        'sub_finish_rate': ratio(s['sub_wins'], s['wins']),
    }

def add_phase_profile_raw_columns(df):
    """
    Computes PRE-fight and POST-fight cumulative phase-skill snapshots for
    both fighters, across the 12 PHASE_METRICS.

    PRE_FIGHT  = career cumulative average strictly BEFORE this fight ("going
                 into this fight, they were X").
    POST_FIGHT = career cumulative average INCLUDING this fight ("coming out
                 of this fight -- or, on a fighter's last row, at the end of
                 their career -- they were X").

    Requires df to already be sorted chronologically ascending.

    Column naming: 'phase_{metric}_{pre_fight|post_fight}_fighter_{1|2}' --
    the temporal token is 'pre_fight'/'post_fight' (not bare 'pre'/'post')
    so it can't be visually confused with 'fighter', and fighter_{1|2} is
    always the LAST suffix, consistent with every other fighter-scoped
    column in this dataset (e.g. dynamic_td_accuracy_fighter_1).
    """
    state = {}
    def get_state(url):
        if url not in state:
            state[url] = _init_phase_state()
        return state[url]

    pre_cols = {f'phase_{p}_pre_fight_{fc}': [] for p in PHASE_METRICS for fc in ['fighter_1', 'fighter_2']}
    post_cols = {f'phase_{p}_post_fight_{fc}': [] for p in PHASE_METRICS for fc in ['fighter_1', 'fighter_2']}

    for row in df.itertuples(index=False):
        method_raw = getattr(row, 'method')
        tt = getattr(row, 'total_time_in_mins')
        for f_col, opp_col in [('fighter_1', 'fighter_2'), ('fighter_2', 'fighter_1')]:
            f_url = getattr(row, f'fighter_url_{f_col}')
            result = getattr(row, f'fight_result_{f_col}')
            s = get_state(f_url)
            pre = _phase_snapshot(s)
            for p in PHASE_METRICS:
                pre_cols[f'phase_{p}_pre_fight_{f_col}'].append(pre[p])

            if pd.notna(tt) and tt > 0:
                head_l, head_a = getattr(row, f'head_strikes_landed_{f_col}'), getattr(row, f'head_strikes_attempted_{f_col}')
                body_l, body_a = getattr(row, f'body_strikes_landed_{f_col}'), getattr(row, f'body_strikes_attempted_{f_col}')
                leg_l, leg_a = getattr(row, f'leg_strikes_landed_{f_col}'), getattr(row, f'leg_strikes_attempted_{f_col}')
                td_l, td_a = getattr(row, f'td_landed_{f_col}'), getattr(row, f'td_attempted_{f_col}')
                opp_td_l, opp_td_a = getattr(row, f'td_landed_{opp_col}'), getattr(row, f'td_attempted_{opp_col}')
                ctrl = getattr(row, f'ctrl_in_secs_{f_col}')
                gs = getattr(row, f'ground_strikes_landed_{f_col}')
                sub = getattr(row, f'sub_att_{f_col}')
                kd = getattr(row, f'kd_{f_col}')

                s['head_landed'] += head_l; s['head_att'] += head_a
                s['body_landed'] += body_l; s['body_att'] += body_a
                s['leg_landed'] += leg_l; s['leg_att'] += leg_a; s['leg_time'] += tt
                s['td_landed'] += td_l; s['td_att'] += td_a
                s['td_faced'] += opp_td_a; s['td_avoided'] += max(0, opp_td_a - opp_td_l)
                s['ctrl'] += ctrl; s['ctrl_time'] += tt
                s['gs_landed'] += gs; s['gs_time'] += tt
                s['sub_att'] += sub; s['sub_time'] += tt
                s['kd'] += kd; s['kd_time'] += tt
                if result == 'W':
                    s['wins'] += 1
                    if method_raw == 'KO/TKO': s['ko_wins'] += 1
                    elif method_raw == 'Submission': s['sub_wins'] += 1

            post = _phase_snapshot(s)
            for p in PHASE_METRICS:
                post_cols[f'phase_{p}_post_fight_{f_col}'].append(post[p])

    new_cols = {**pre_cols, **post_cols}
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


def _phase_windowed_percentile(d, col, weight_class_col='weight_class_cleaned',
                                date_col='event_date', window_years=3, min_cohort=5):
    """
    Percentile rank of each row's value against its CONTEMPORARIES only: rows
    in the same weight class whose event_date falls within +/- window_years.
    This is the 'era' percentile, kept alongside (not instead of) the
    full-history percentile because some phase metrics drift substantially
    over the sport's history (e.g. leg-kick volume has risen ~40% since
    2010) while others don't (e.g. control-time rate has actually *fallen*
    ~34% since the early 2000s) -- ranking a fighter against the sport's
    entire multi-decade history can meaningfully understate or overstate
    them depending on which direction their era drifted, and there's no way
    to know which, for a given fighter/metric, without computing both.
    Rows with fewer than `min_cohort` contemporaries are left NaN.
    """
    window_days = int(window_years * 365.25)
    dates_all = pd.to_datetime(d[date_col]).apply(lambda x: x.toordinal() if pd.notna(x) else np.nan)
    out = np.full(len(d), np.nan)
    for wc, g in d.groupby(weight_class_col):
        gg = g.dropna(subset=[col])
        gg_dates = dates_all.loc[gg.index]
        order = gg_dates.sort_values().index
        dates_sorted = gg_dates.loc[order].values
        vals_sorted = gg.loc[order, col].values
        for pos, idx in enumerate(order):
            lo = np.searchsorted(dates_sorted, dates_sorted[pos] - window_days, side='left')
            hi = np.searchsorted(dates_sorted, dates_sorted[pos] + window_days, side='right')
            cohort = vals_sorted[lo:hi]
            if len(cohort) < min_cohort:
                continue
            out[d.index.get_loc(idx)] = (cohort <= vals_sorted[pos]).mean()
    return out


def add_phase_profile_percentiles(df, window_years=3):
    """
    Adds full-history and era-windowed percentile ranks (within weight class)
    for every phase_{p}_{pre_fight|post_fight}_fighter_{1|2} raw column. Both
    flavors are kept side by side deliberately -- see
    _phase_windowed_percentile docstring.

    Output column pattern:
    'phase_{metric}_{pre_fight|post_fight}_pctile_{full|era}_fighter_{1|2}'
    -- fighter_{1|2} always LAST, consistent with every other fighter-scoped
    column in this pipeline (e.g. dynamic_sig_strikes_accuracy_fighter_1,
    opponent_quality_delta_fighter_1).
    """
    new_cols = {}
    for p in PHASE_METRICS:
        for suffix in ['pre_fight', 'post_fight']:
            for f_col in ['fighter_1', 'fighter_2']:
                raw_col = f'phase_{p}_{suffix}_{f_col}'
                new_cols[f'phase_{p}_{suffix}_pctile_full_{f_col}'] = df.groupby('weight_class_cleaned')[raw_col].rank(pct=True)
                new_cols[f'phase_{p}_{suffix}_pctile_era_{f_col}'] = _phase_windowed_percentile(df, raw_col, window_years=window_years)
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


"""## Master Orchestration Function"""

def engineer_all_features(
    df: pd.DataFrame,
    include_phase_profiles: bool = False,
    include_phase_profile_percentiles: bool = False,
    phase_profile_percentile_window_years: int = 3,
) -> pd.DataFrame:
    """
    Master pipeline executing feature engineering transformations via dictionary batching.

    Args:
        include_phase_profiles: If False (default), skips the 12-phase
            pre/post-fight raw snapshot columns entirely (48 columns) and
            the percentile columns that depend on them. Off by default
            because these aren't currently consumed downstream (not in
            ude_points_algorithm.py) and roughly double the dataset's
            column count.
        include_phase_profile_percentiles: Only relevant when
            include_phase_profiles=True. Lets you get the 48 raw columns
            without paying for the 96 percentile columns (full + era, x2
            fighters, x12 metrics), which are the more expensive half —
            the era-windowed rank does a per-row search over each
            weight-class cohort.
        phase_profile_percentile_window_years: Passed through to the era
            percentile calculation.
    """
    df = df.copy()

    if 'event_date' in df.columns:
        df['event_date'] = pd.to_datetime(df['event_date'])
        # Safety net: dataset_processing_pipeline.run_etl_pipeline already
        # drops+reports null event_date rows at the source (the event-date
        # join). Re-checking here too in case this df didn't come through
        # that path -- every chronological state machine below assumes a
        # valid, sortable date on every row.
        df = drop_rows_with_null_event_date(df)

    df = df.sort_values(by='event_date', ascending=True).reset_index(drop=True)

    # Defensive NaN guard for the raw landed/attempted stat columns consumed
    # by update_career_means, add_dynamic_strike_accuracy/defence, and
    # add_dynamic_td_accuracy/defence: each accumulates a running per-fighter
    # total via `+=` with no NaN check, so a single NaN input would silently
    # poison that fighter's cumulative total -- and every derived column for
    # them -- to NaN for the remainder of their career. Not currently
    # triggered (verified zero NaN across these columns in the live dataset)
    # but unguarded, and the failure mode is exactly the shape of this
    # project's one confirmed bug class (a silent, cascading NaN). Filling
    # to 0 here matches _to_num's existing NaN-as-0 convention used in
    # calculate_phase_magnitude_and_pdi below.
    _cumulative_stat_cols = [
        f'{stat}_fighter_{side}'
        for stat in ('sig_strikes_landed', 'sig_strikes_attempted', 'td_landed', 'td_attempted',
                     'head_strikes_landed', 'head_strikes_attempted',
                     'body_strikes_landed', 'body_strikes_attempted',
                     'leg_strikes_landed', 'leg_strikes_attempted')
        for side in (1, 2)
    ]
    present = [c for c in _cumulative_stat_cols if c in df.columns]
    df[present] = df[present].fillna(0)

    # 1. Title bout column
    df = create_is_title_bout_column(df)

    # 2. Weight class cleaning
    if 'weight_class' in df.columns:
        df['weight_class_cleaned'] = df['weight_class'].apply(map_weight_class)

    # 3. Champion status & Title defenses
    df = update_champion_status(df)
    df = update_title_defenses(df)

    # 4. Fight records & Streaks
    df = update_fight_records(df)
    df = update_win_streaks(df)

    # 4b. Opponent quality scores. Must run after champion status, title
    # defenses, and fight records above; must run before UDE scoring, since
    # ude_points_algorithm's temporal method x PDI calibration reads these
    # columns before any UDE points are computed (see docstring).
    df = add_quality_score_columns(df)

    # 5. Career means
    df = update_career_means(df)

    # 6. Method cleaning
    if 'method' in df.columns:
        df['method_mapped'] = df['method'].apply(map_fight_method)

    # 7. Rematches
    df = add_rematch_features(df)

    # 8. Static & Dynamic Defense/Accuracy
    df = add_defense_columns(df)
    df = add_dynamic_strike_accuracy(df)
    df = add_dynamic_strike_defence(df)
    df = add_dynamic_td_accuracy(df)
    df = add_dynamic_td_defence(df)

    # 9. Standing significant strikes (must run before per-minute rates below,
    # which scan df.columns for every *_landed/*_attempted column still
    # present -- standing_sig_strikes_landed_per_min would otherwise silently
    # never get generated since its source column wouldn't exist yet)
    df = add_standing_sig_strikes_columns(df)

    # 10. Time calculations & per-minute rates
    df = add_time_and_per_min_features(df)

    # 11. Differentials & Dominance Features
    df = process_dominance_differentials(df)
    df = add_who_won_col(df)
    df = add_dominance_columns(df)
    df = add_pdi_columns(df)

    # 12. Phase-level fighter strength/weakness profiles (pre/post-fight,
    # full-history + era-windowed percentiles). Must run while still sorted
    # ascending, like everything else above.
    if include_phase_profiles:
        df = add_phase_profile_raw_columns(df)
        if include_phase_profile_percentiles:
            df = add_phase_profile_percentiles(df, window_years=phase_profile_percentile_window_years)

    df = df.sort_values(by='event_date', ascending=False).reset_index(drop=True)

    return df
