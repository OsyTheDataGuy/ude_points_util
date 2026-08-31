# -*- coding: utf-8 -*-
# import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import re
from datetime import date, datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ude_points_algorithm import is_no_score_fight

'''1. Functions to Create Fighter Career Dataset'''
def create_fighter_career_dataset(df, fighter_name):
    """
    Creates a dataset of the specified fighter's career.

    Args:
    - df (pd.DataFrame): The full dataset of all fights.
    - fighter_name (str): Name of the fighter to generate the career dataset for.

    Returns:
    - pd.DataFrame: A new dataset containing the career details of the fighter.
    """
    fighter_fights = filter_fighter_fights(df, fighter_name)
    fighter_details = extract_fighter_details_programmatically(fighter_fights, fighter_name)
    opponent_details = extract_opponent_details_programmatically(fighter_fights, fighter_name)
    final_dataset = reorganize_fight_data_programmatically(fighter_fights, fighter_details, opponent_details)
    final_dataset = create_diff_columns(final_dataset)

    return final_dataset

def filter_fighter_fights(df, fighter_name):
    """
    Filters the dataset to include only fights involving the specified fighter.

    Args:
    - df (pd.DataFrame): The full dataset of all fights.
    - fighter_name (str): Name of the fighter.

    Returns:
    - pd.DataFrame: Filtered dataset with only the fights involving the fighter.
    """
    return df[(df['fighter_1'] == fighter_name) | (df['fighter_2'] == fighter_name)].copy()

def extract_fighter_details_programmatically(df, fighter_name):
    """
    Extracts dynamic details of the specified fighter from each fight.

    Args:
    - df (pd.DataFrame): Filtered dataset of the fighter's fights.
    - fighter_name (str): Name of the fighter.

    Returns:
    - pd.DataFrame: DataFrame containing fighter details for each fight.
    """
    # Create a mask to identify if the fighter is in fighter_1 or fighter_2 columns
    is_fighter_1 = df['fighter_1'] == fighter_name

    # Identify all columns that contain '_fighter_1' or '_fighter_2', except 'fighter_1' and 'fighter_2'
    fighter_columns = [col for col in df.columns if ('fighter_1' in col or 'fighter_2' in col) and col not in ['fighter_1', 'fighter_2']]

    # Dynamically create a dictionary of opponent-specific stats
    fighter_stats = df.apply(
        lambda row: {
            'fighter': row['fighter_1'] if is_fighter_1[row.name] else row['fighter_2'],
            'age': row['fight_day_age (yrs)_fighter_1'] if is_fighter_1[row.name] else row['fight_day_age (yrs)_fighter_2'],
            **{
                col.replace('_fighter_1', '').replace('_fighter_2', ''): 
                # Find the column from fighter_columns that matches the cleaned column and contains '_fighter_2' or '_fighter_1'
                row[col if 'fighter_1' in col else col.replace('_fighter_2', '_fighter_1')] if is_fighter_1[row.name] 
                else row[col if 'fighter_2' in col else col.replace('_fighter_1', '_fighter_2')]
                for col in fighter_columns
            }
        }, axis=1, result_type='expand'
    )

    return fighter_stats


def extract_opponent_details_programmatically(df, fighter_name):
    """
    Extracts dynamic details of the opponent from each fight.

    Args:
    - df (pd.DataFrame): Filtered dataset of the fighter's fights.
    - fighter_name (str): Name of the fighter.

    Returns:
    - pd.DataFrame: DataFrame containing opponent details for each fight.
    """
    # Create a mask to identify if the fighter is in fighter_1 or fighter_2 columns
    is_fighter_1 = df['fighter_1'] == fighter_name

    # Identify all columns that contain '_fighter_1' or '_fighter_2', except 'fighter_1' and 'fighter_2'
    fighter_columns = [col for col in df.columns if ('fighter_1' in col or 'fighter_2' in col) and col not in ['fighter_1', 'fighter_2']]

    # Dynamically create a dictionary of opponent-specific stats
    opponent_stats = df.apply(
        lambda row: {
            'opponent': row['fighter_2'] if is_fighter_1[row.name] else row['fighter_1'],
            'opponent_age': row['fight_day_age (yrs)_fighter_2'] if is_fighter_1[row.name] else row['fight_day_age (yrs)_fighter_1'],
            **{
                'opponent_' + col.replace('_fighter_1', '').replace('_fighter_2', ''): 
                # Find the column from fighter_columns that matches the cleaned column and contains '_fighter_2' or '_fighter_1'
                row[col if 'fighter_2' in col else col.replace('_fighter_1', '_fighter_2')] if is_fighter_1[row.name] 
                else row[col if 'fighter_1' in col else col.replace('_fighter_2', '_fighter_1')]
                for col in fighter_columns
            }
        }, axis=1, result_type='expand'
    )

    return opponent_stats

def reorganize_fight_data_programmatically(df, fighter_details, opponent_details):
    """
    Reorganizes and combines fight data into the final dataset structure.

    Args:
    - df (pd.DataFrame): Filtered dataset of the fighter's fights.
    - fighter_details (pd.DataFrame): Fighter stats and details.
    - opponent_details (pd.DataFrame): Opponent stats and details.

    Returns:
    - pd.DataFrame: Final dataset with organized fight data.
    """
    shared_cols = [col for col in df.columns if 'fighter_1' not in col and 'fighter_2' not in col]
    fight_data = pd.concat([df[shared_cols], fighter_details, opponent_details], axis=1)
    return fight_data.reset_index(drop=True)


def reorganize_fight_data(df, fighter_details, opponent_details):
    """
    Reorganizes and combines fight data into the final dataset structure.

    Args:
    - df (pd.DataFrame): Filtered dataset of the fighter's fights.
    - fighter_details (pd.DataFrame): Fighter stats and details.
    - opponent_details (pd.DataFrame): Opponent stats and details.

    Returns:
    - pd.DataFrame: Final dataset with organized fight data.
    """
    fight_data = pd.concat([df[['event_date', 'event_name', 'event_url', 'bout', 'fight_url', 'weight_class', 'weight_class_cleaned','is_title_bout',
                                'time_format', 'match_format_rounds', 'is_rematch', 'method', 'method_mapped', 'time', 'time_in_mins',
                                'round_ended', 'total_time_in_mins', 'who_won_striking', 'who_won_wrestling', 'who_won_grappling', 'who_won_control',
                                'who_won_standing_danger', 'dominant_fighter', 'phases_won']], fighter_details, opponent_details], axis=1)
    return fight_data.reset_index(drop=True)

def create_diff_columns(df):
    """
    Create height_diff, reach_diff, and age_diff columns in the final dataset structure

    Args:
    - df (pd.DataFrame): Final dataset with organized fight data.

    Returns:
    - pd.DataFrame: Final dataset with height_diff, reach_diff, and age_diff columns.
    """
    df['height_diff'] = df['Height (m)'] - df['opponent_Height (m)']
    df['reach_diff'] = df['Reach (in)'] - df['opponent_Reach (in)']
    df['age_diff'] = df['age'] - df['opponent_age']

    return df.sort_values(by='event_date', ascending=False).reset_index(drop=True)

# Create function that takes a fighter's name and returns their dataset, their total opponent_sig_strikes_landed and opponent_total_strikes_landed, their mean and median opponent_sig_strikes_landed
def fighter_stats(df, fighter_name):
    fighter_stats = create_fighter_career_dataset(df, fighter_name)
    total_sig_strikes_absorbed = fighter_stats['opponent_sig_strikes_landed'].sum()
    total_fights = fighter_stats.shape[0]
    mean_opponent_sig_strikes_landed = fighter_stats['opponent_sig_strikes_landed'].mean()
    median_opponent_sig_strikes_landed = fighter_stats['opponent_sig_strikes_landed'].median()

    return fighter_stats, total_sig_strikes_absorbed, total_fights, mean_opponent_sig_strikes_landed, median_opponent_sig_strikes_landed


'''2. Functions to Create Championship Reigns and Contendership Datasets'''
# Function to filter title bouts
def filter_title_bouts(df):
    """Filter rows where the bout is a title fight."""
    return df[df['is_title_bout'] == 2].copy()

# Function to assign champion and contender columns
def assign_champion_contender(title_bouts):
    """Assign champion and contender based on boolean columns, handling vacant belts."""
    def determine_champion(row):
        if row['is_champion_fighter_1'] == 2:
            return row['fighter_1']
        elif row['is_champion_fighter_2'] == 2:
            return row['fighter_2']
        return None  # Vacant belt case

    def determine_contender(row):
        if row['is_champion_fighter_1'] == 2:
            return row['fighter_2']
        elif row['is_champion_fighter_2'] == 2:
            return row['fighter_1']
        return None  # Vacant belt case

    title_bouts['champion'] = title_bouts.apply(determine_champion, axis=1)
    title_bouts['contender'] = title_bouts.apply(determine_contender, axis=1)

    # Handle vacant belts: Both fighters are contenders
    title_bouts.loc[title_bouts['champion'].isnull(), 'champion'] = 'Vacant'
    return title_bouts

# Function to dynamically assign champion and contender stats
def assign_champion_contender_stats(title_bouts):
    """Assign stats dynamically to champion and contender, handling vacant belts."""
    # Champion stats
    title_bouts['champion_age'] = title_bouts.apply(
        lambda row: row['fight_day_age (yrs)_fighter_1'] if row['champion'] == row['fighter_1'] else (
            row['fight_day_age (yrs)_fighter_2'] if row['champion'] == row['fighter_2'] else None), axis=1
    )
    title_bouts['champion_W/L_streak'] = title_bouts.apply(
        lambda row: row['W/L_streak_fighter_1'] if row['champion'] == row['fighter_1'] else (
            row['W/L_streak_fighter_2'] if row['champion'] == row['fighter_2'] else None), axis=1
    )
    title_bouts['champion_result'] = title_bouts.apply(
        lambda row: row['fight_result_fighter_1'] if row['champion'] == row['fighter_1'] else (
            row['fight_result_fighter_2'] if row['champion'] == row['fighter_2'] else None), axis=1
    )

    # Contender stats
    title_bouts['contender_age'] = title_bouts.apply(
        lambda row: row['fight_day_age (yrs)_fighter_2'] if row['contender'] == row['fighter_2'] else (
            row['fight_day_age (yrs)_fighter_1'] if row['contender'] == row['fighter_1'] else None), axis=1
    )
    title_bouts['contender_W/L_streak'] = title_bouts.apply(
        lambda row: row['W/L_streak_fighter_2'] if row['contender'] == row['fighter_2'] else (
            row['W/L_streak_fighter_1'] if row['contender'] == row['fighter_1'] else None), axis=1
    )
    title_bouts['contender_result'] = title_bouts.apply(
        lambda row: row['fight_result_fighter_2'] if row['contender'] == row['fighter_2'] else (
            row['fight_result_fighter_1'] if row['contender'] == row['fighter_1'] else None), axis=1
    )

    # Handle vacant belts: Assign stats for both as contenders
    vacant_mask = title_bouts['champion'] == 'Vacant'
    title_bouts.loc[vacant_mask, 'champion_age'] = None
    title_bouts.loc[vacant_mask, 'champion_W/L_streak'] = None
    title_bouts.loc[vacant_mask, 'champion_result'] = None

    return title_bouts

# Function to select relevant columns
def select_title_bout_columns(title_bouts_dataset):
    """Select columns of interest for title fight analysis."""
    columns_to_keep = [
        'event_name', 'event_date', 'champion', 'contender',
        'champion_age', 'contender_age',
        'champion_W/L_streak', 'contender_W/L_streak',
        'champion_result', 'contender_result'
    ]
    return title_bouts_dataset[columns_to_keep].copy()

# Function to filter vacant title bouts
def filter_vacant_title_bouts(df):
    """Filter rows where the title bout is for a vacant belt."""
    vacant_belts = (df['is_champion_fighter_1'] != 2) & (df['is_champion_fighter_2'] != 2)
    return df[(df['is_title_bout'] == 2) & vacant_belts].copy()

# Function to assign contender_a and contender_b for vacant bouts
def assign_vacant_contenders(vacant_bouts_dataset):
    """Assign fighter_1 and fighter_2 as contender_a and contender_b."""
    vacant_bouts_dataset.rename(columns={
        'fighter_1': 'contender_a',
        'fighter_2': 'contender_b',
        'fight_day_age (yrs)_fighter_1': 'contender_a_age',
        'fight_day_age (yrs)_fighter_2': 'contender_b_age',
        'W/L_streak_fighter_1': 'contender_a_W/L_streak',
        'W/L_streak_fighter_2': 'contender_b_W/L_streak',
        'fight_result_fighter_1': 'contender_a_result',
        'fight_result_fighter_2': 'contender_b_result',
    }, inplace=True)
    return vacant_bouts_dataset

# Function to select relevant columns for vacant title bouts
def select_vacant_columns(vacant_bouts_dataset):
    """Select columns of interest for vacant title bouts."""
    columns_to_keep = [
        'event_name', 'event_date', 'contender_a', 'contender_b',
        'contender_a_age', 'contender_b_age',
        'contender_a_W/L_streak', 'contender_b_W/L_streak',
        'contender_a_result', 'contender_b_result'
    ]
    return vacant_bouts_dataset[columns_to_keep].copy()

# Main function to create both datasets
def create_title_bouts_datasets(df):
    """Create datasets for title bouts and vacant title bouts."""
    # Champion/Contender Dataset
    title_bouts = filter_title_bouts(df)
    title_bouts = assign_champion_contender(title_bouts)
    title_bouts = assign_champion_contender_stats(title_bouts)
    champion_contender_dataset = select_title_bout_columns(title_bouts)

    # Vacant Title Bouts Dataset
    vacant_bouts = filter_vacant_title_bouts(df)
    vacant_bouts = assign_vacant_contenders(vacant_bouts)
    vacant_bouts_dataset = select_vacant_columns(vacant_bouts)

    return champion_contender_dataset, vacant_bouts_dataset


'''3. Functions to Plot Graphs from Fighter Career dataset'''
# Plot differentials columns
def plot_diff(fighter_stats, fighter_name, diff_column='age_diff', title_bouts=True, sort_ascending=True, subtitle=None, **kwargs):
    # Filter for title bouts if title_bouts is True
    if title_bouts:
        data = fighter_stats[fighter_stats['is_title_bout'] > 0]
    else:
        data = fighter_stats

    # Sort strictly by event_date
    data = data.sort_values(by='event_date', ascending=sort_ascending)

    # Create color column based on diff_column
    data['color'] = data[diff_column].apply(lambda x: 'red' if x < 0 else 'blue')

    # Convert event_date to string format to remove time
    data['event_date_str'] = data['event_date'].dt.strftime('%Y-%m-%d')

    # Explicitly set the order for the x-axis to maintain chronological order
    category_order = data['event_date_str'].tolist()

    # Obtain first word from diff_column for labels
    first_word = diff_column.split('_')[0].capitalize()

    # Create the Plotly bar chart
    fig = px.bar(
        data_frame=data,
        x='event_date_str',
        y=diff_column,
        color='color',
        text='opponent',  # Add opponent names as text annotations
        color_discrete_map={'red': 'red', 'blue': 'blue'},
        labels={diff_column: f'{first_word} Difference (Fighter {first_word} - Opponent {first_word})', 'event_date_str': 'Fight Date'},
        title=f"{first_word} Difference in {'Title Bouts' if title_bouts else 'Career'} for {fighter_name}",
        category_orders={'event_date_str': category_order},  # Fix order of the x-axis
        # on hover show opponent_age but shut off 'color'
        hover_data={'opponent_age':True, 'color':False}
    )

    # Customize the layout
    fig.update_traces(textposition='outside', textfont_size=9)
    fig.update_xaxes(
        type='category',  # Enforce categorical axis
        tickvals=category_order,
        ticktext=category_order,  # Use the formatted date as the tick text
        tickangle=45
    )

    # Add subtitle if provided
    if subtitle:
        fig.add_annotation(
            x=0.5,  # Position at the center of the plot
            y=1.05,  # Below the title (adjust y as necessary)
            text=subtitle,
            showarrow=False,
            font=dict(size=16, color="gray"),
            align="center",
            xref="paper",
            yref="paper"
        )

    # Update layout with any additional arguments passed via kwargs (e.g., height, width)
    fig.update_layout(**kwargs)

    # Show the plot
    fig.show()

# Plot cumulative_metric columns
def plot_cumulative_metric_solo(fighter_stats, column='dynamic_sig_strikes_def', title_bouts=False, subtitle=None, avg_med='mean', **kwargs):
    # Filter for title bouts if title_bouts is True
    if title_bouts:
        data = fighter_stats[fighter_stats['is_title_bout'] > 0]
    else:
        data = fighter_stats

    # Sort strictly by event_date
    data = data.sort_values(by='event_date')
    fighter_name = data.loc[0,'fighter']

    # Convert event_date to string format to remove time
    data['event_date_str'] = data['event_date'].dt.strftime('%Y-%m-%d')

    # Generate the graph title
    graph_title = column.replace('_', ' ').title()

    # Create the Plotly line chart
    fig = px.line(
        data_frame=data,
        x='event_date_str',
        y=column,
        text='opponent',  # Add opponent names as text annotations
        title=f"{graph_title} in {'Title Bouts' if title_bouts else 'Career'} for {fighter_name}",
        labels={
            column: f'{column}',
            'event_date_str': 'Fight Date'
        },
        hover_data={
            'age': True,
            'opponent': True,
            'opponent_age': True,
            'event_name': True,
            'event_date_str': True,
            column: True
        }
    )

    # Customize the layout
    fig.update_traces(textposition='top center', textfont_size=9, mode='lines+markers+text')
    fig.update_xaxes(
        type='category',  # Enforce categorical axis
        tickangle=45
    )

    # Add subtitle if provided
    if subtitle:
        fig.add_annotation(
            x=0.5,  # Position at the center of the plot
            y=1.05,  # Below the title (adjust y as necessary)
            text=subtitle,
            showarrow=False,
            font=dict(size=16, color="gray"),
            align="center",
            xref="paper",
            yref="paper"
        )

    # Add dashed line for mean or median if specified
    if avg_med in ['mean', 'median']:
        line_value = data[column].mean() if avg_med == 'mean' else data[column].median()
        fig.add_hline(
            y=line_value,
            line_dash="dash",
            line_color="red",
            annotation_text=f"{avg_med.capitalize()}: {line_value:.2f}",
            annotation_position="top left",
            annotation_font_size=10
        )

    # Update layout with any additional arguments passed via kwargs (e.g., height, width)
    fig.update_layout(**kwargs)

    # Show the plot
    fig.show()


# Plot two cumulative_metrics
def plot_cumulative_metric_combo(fighter_stats, column='dynamic_sig_strikes_def', opponent_column='opponent_dynamic_sig_strikes_acc', title_bouts=False, subtitle=None, **kwargs):
    # Filter for title bouts if title_bouts is True
    if title_bouts:
        data = fighter_stats[fighter_stats['is_title_bout'] > 0]
    else:
        data = fighter_stats

    # Sort strictly by event_date
    data = data.sort_values(by='event_date')
    fighter_name = data.loc[0, 'fighter']

    # Convert event_date to string format to remove time
    data['event_date_str'] = data['event_date'].dt.strftime('%Y-%m-%d')

    last_word = column.split('_')[-1].capitalize()

    # Initialize a Plotly figure
    fig = go.Figure()

    # Add the fighter's metric line
    fig.add_trace(
        go.Scatter(
            x=data['event_date_str'],
            y=data[column],
            mode='lines+markers',
            name=f"{fighter_name}'s {last_word}",
            text=data['opponent'],  # Opponent names as annotations
            hovertemplate=(
                f"<b>Fight Date:</b> {{%{{x}}}}<br>"
                f"<b>{fighter_name}'s {last_word}:</b> {{%{{y}}}}<br>"
                f"<b>Opponent:</b> {{%{{text}}}}<extra></extra>"
            )
        )
    )

    # Add the opponents' metric line
    fig.add_trace(
        go.Scatter(
            x=data['event_date_str'],
            y=data[opponent_column],
            mode='lines+markers',
            name="Opponents' Strike Accuracy",
            line=dict(dash='dash'),  # Dashed line for distinction
            hovertemplate=(
                f"<b>Fight Date:</b> {{%{{x}}}}<br>"
                f"<b>Opponents' Strike Accuracy:</b> {{%{{y:.2f}}}}<extra></extra>"
            )
        )
    )

    # Add title and axis labels
    fig.update_layout(
        title=f"Cumulative Significant Strike {last_word} and Opponents' Accuracy in {'Title Bouts' if title_bouts else 'Career'} for {fighter_name}",
        xaxis_title="Fight Date",
        yaxis_title="Value",
        hovermode='x unified',  # Unified hover mode for clearer comparisons
        yaxis=dict(title="Metric Value"),  # Shared y-axis for both metrics
        **kwargs  # Additional layout customizations
    )

    # Add subtitle if provided
    if subtitle:
        fig.add_annotation(
            x=0.5,  # Position at the center of the plot
            y=1.05,  # Below the title (adjust y as necessary)
            text=subtitle,
            showarrow=False,
            font=dict(size=16, color="gray"),
            align="center",
            xref="paper",
            yref="paper"
        )

    # Show the plot
    fig.show()

# Plot dynamic stats for two fighters
def plot_dynamic_stat_comparison(fighter_stats_1, fighter_stats_2, column='dynamic_sig_strikes_def', subtitle=None, **kwargs):
    """
    Plot the dynamic stats of two fighters over time for comparison.

    Args:
        fighter_stats (DataFrame): The dataset containing fighters' stats.
        fighter_1 (str): Name of the first fighter.
        fighter_2 (str): Name of the second fighter.
        column (str): The dynamic stat column to plot.
        subtitle (str): Optional subtitle for the chart.
        **kwargs: Additional arguments for Plotly layout (e.g., width, height).
    """
    # Filter data for the two fighters
    data_1 = fighter_stats_1.sort_values(by='event_date')
    data_2 = fighter_stats_2.sort_values(by='event_date')

    # Ensure event_date is string for better formatting
    data_1['event_date_str'] = data_1['event_date'].dt.strftime('%Y-%m-%d')
    data_2['event_date_str'] = data_2['event_date'].dt.strftime('%Y-%m-%d')

    # Combine data into a single DataFrame with a column indicating the fighter
    # data_1['warrior'] = fighter_1
    # data_2['warrior'] = fighter_2
    combined_data = pd.concat([data_1, data_2], ignore_index=True)

    # Create the Plotly line chart
    fig = px.line(
        data_frame=combined_data,
        x='event_date_str',
        y=column,
        color='fighter',
        text='opponent',  # Add opponent names as text annotations
        title=f"Comparison of {column.replace('_', ' ').title()} for {data_1.loc[0,'fighter']} and {data_2.loc[0,'fighter']}",
        labels={
            column: column.replace('_', ' ').title(),
            'event_date_str': 'Fight Date',
            'fighter': 'Fighter'
        },
        hover_data={
            'opponent': True,
            'event_name': True,
            'event_date_str': True,
            column: True
        }
    )

    # Customize the layout
    fig.update_traces(textposition='top center', textfont_size=9, mode='lines+markers+text')
    fig.update_xaxes(
        type='category',  # Enforce categorical axis
        tickangle=45
    )

    # Add subtitle if provided
    if subtitle:
        fig.add_annotation(
            x=0.5,  # Position at the center of the plot
            y=1.05,  # Below the title (adjust y as necessary)
            text=subtitle,
            showarrow=False,
            font=dict(size=16, color="gray"),
            align="center",
            xref="paper",
            yref="paper"
        )

     # Update layout with additional arguments
    fig.update_layout(
        xaxis=dict(type='date'),  # Ensure a shared time-based x-axis
        **kwargs
    )

    # Show the plot
    fig.show()

'''4. Functions to ranking by Ude points (career-end rank)'''
def get_latest_ude_points_with_details(df):
    """
    Get each fighter's latest Ude points after their last fight,
    including age and record at that time.
    """
    # Sort the dataframe by event date to ensure the most recent fight is last
    df = df.sort_values(by='event_date')

    # Initialize a dictionary to store the latest Ude points, age, and record for each fighter
    latest_ude_points = {}

    # Loop through each fight and update the latest Ude points, age, and record for each fighter
    for index, row in df.iterrows():
        # For fighter 1
        fighter_1 = row['fighter_1']
        fighter_1_url = row['fighter_url_fighter_1']
        post_fight_ude_1 = row['ude_points_post_fight_fighter_1']
        age_1 = row['fight_day_age (yrs)_fighter_1']
        record_1 = row['post_fight_record_fighter_1_(W-L-D NC)']

        latest_ude_points[fighter_1_url] = {
            'fighter': fighter_1,
            'latest_ude_points': post_fight_ude_1,
            'age': age_1,
            'record': record_1
        }

        # For fighter 2
        fighter_2 = row['fighter_2']
        fighter_2_url = row['fighter_url_fighter_2']
        post_fight_ude_2 = row['ude_points_post_fight_fighter_2']
        age_2 = row['fight_day_age (yrs)_fighter_2']
        record_2 = row['post_fight_record_fighter_2_(W-L-D NC)']

        latest_ude_points[fighter_2_url] = {
            'fighter': fighter_2,
            'latest_ude_points': post_fight_ude_2,
            'age': age_2,
            'record': record_2
        }

    return latest_ude_points

def rank_fighters_by_latest_ude_points(df):
    """
    Rank fighters by their latest Ude points and return as a dataframe.
    """
    # Get the latest Ude points, age, and record for all fighters
    latest_ude_points = get_latest_ude_points_with_details(df)

    # Convert the latest Ude points dictionary into a dataframe
    ude_points_df = pd.DataFrame.from_dict(latest_ude_points, orient='index')

    # Rename the columns for clarity
    ude_points_df.reset_index(inplace=True)
    ude_points_df.rename(columns={'index': 'fighter_url'}, inplace=True)

    # Sort the dataframe by Ude points in descending order
    ude_points_df = ude_points_df.sort_values(by='latest_ude_points', ascending=False).reset_index(drop=True)

    return ude_points_df[['fighter', 'fighter_url', 'age', 'record', 'latest_ude_points']]

'''5. Functions to ranking by Ude points (rank by career peak Ude rating)'''
def rank_fighters_by_peak_ude_points(df):
    # Melt the dataframe to combine fighter_1 and fighter_2 stats into a single 'fighter' column
    fighter_1_data = df[['fighter_1', 'fighter_url_fighter_1','ude_points_post_fight_fighter_1',
                          'fight_day_age (yrs)_fighter_1',
                          'post_fight_record_fighter_1_(W-L-D NC)']].rename(columns={
        'fighter_1': 'fighter', 'fighter_url_fighter_1': 'fighter_url',
        'ude_points_post_fight_fighter_1': 'ude_points_post_fight',
        'fight_day_age (yrs)_fighter_1': 'age_at_peak_ude_points',
        'post_fight_record_fighter_1_(W-L-D NC)': 'post_fight_record'
    })

    fighter_2_data = df[['fighter_2', 'fighter_url_fighter_2','ude_points_post_fight_fighter_2',
                          'fight_day_age (yrs)_fighter_2',
                          'post_fight_record_fighter_2_(W-L-D NC)']].rename(columns={
        'fighter_2': 'fighter', 'fighter_url_fighter_2': 'fighter_url',
        'ude_points_post_fight_fighter_2': 'ude_points_post_fight',
        'fight_day_age (yrs)_fighter_2': 'age_at_peak_ude_points',
        'post_fight_record_fighter_2_(W-L-D NC)': 'post_fight_record'
    })

    # Combine both sets of data
    combined_fighter_data = pd.concat([fighter_1_data, fighter_2_data], axis=0)

    # Find the index of the maximum ude_points_post_fight for each fighter
    max_indices = combined_fighter_data.groupby('fighter')['ude_points_post_fight'].idxmax()

    # Use the max indices to filter the combined data
    fighter_max_ude_points = combined_fighter_data.loc[max_indices]

    # Sort by ude_points_post_fight in descending order to rank fighters
    fighter_max_ude_points_sorted = fighter_max_ude_points.sort_values(by='ude_points_post_fight', ascending=False).drop_duplicates('fighter_url', keep='first').reset_index(drop=True)

    # Add a ranking column
    fighter_max_ude_points_sorted['rank'] = fighter_max_ude_points_sorted['ude_points_post_fight'].rank(method='dense', ascending=False).astype(int)

    # Return relevant columns
    return fighter_max_ude_points_sorted[['fighter', 'fighter_url', 'age_at_peak_ude_points', 'post_fight_record', 'ude_points_post_fight', 'rank']]
 
'''6. Functions to ranking by Ude points (rank by shrunk career points-per-fight rate)'''
def rank_fighters_by_shrunk_ude_rate(df, prior_strength=10.0, min_fights=None):
    """
    Rank fighters by career UDE points earned PER FIGHT, shrunk toward the
    population mean rate via Bayesian shrinkage -- the same pattern used by
    shrunk_win_rate() in ude_points_algorithm.py (there with a strength-5
    prior toward a .500 win rate; here with a strength-10 prior toward the
    population's mean per-fight rate).

    Why this exists: rank_fighters_by_latest_ude_points() and
    rank_fighters_by_peak_ude_points() both rank on a cumulative career sum
    that starts at 500 and never decays or normalizes by fight count. That
    makes total career fight VOLUME alone able to compound a mediocre
    per-fight rate into a high cumulative rank -- e.g. a fighter who fought
    20 times and won 12 (60%) can out-accumulate a fighter who fought 8 times
    and won 7 (87.5%), purely because they had more fights to accumulate
    points in, independent of whether each individual fight was scored
    correctly. This function answers a different, and for a "greatest ever"
    ranking arguably more appropriate, question: how much value did this
    fighter generate ON AVERAGE per fight, with small career sample sizes
    (e.g. a fighter with 3-4 UFC fights) pulled toward the mean rather than
    let a tiny denominator produce an inflated or deflated rate.

    Bayesian shrinkage alone is not sufficient stabilization at very small n:
    a prior_strength strong enough to tame a 3-4 fight outlier also distorts
    everyone else's rate, and even a heavy shrink can leave an extreme-enough
    raw rate (e.g. a career built on stacked title-bout bonuses) ranked far
    higher than a 3-4 fight sample supports -- rank became highly unstable
    under small perturbations of prior_strength in exactly these cases.
    `min_fights` is a separate, direct floor for exactly that failure mode:
    fighters below it are excluded before ranking, rather than relying on
    the prior to argue them out of contention.

    Args:
    - df (pd.DataFrame): The scored fight dataset (post ude_points_algorithm).
      Does not need to be the full dataset -- see career_point_gain note
      below for what passing a filtered subset (e.g. one division) means.
    - prior_strength (float): Bayesian shrinkage prior strength, in
      equivalent number of fights. Default 10.0, consistent with the
      project's existing shrinkage convention.
    - min_fights (int or None): If set, fighters with fewer than this many
      scored fights are excluded before ranking. `rank` is then dense-ranked
      within the qualifying subset only (not the full population), so a
      returned rank of 1 is the best fighter who clears the floor.
      `population_mean_rate` (and therefore `shrunk_rate` for everyone who
      does qualify) is still computed from the FULL, unfiltered population --
      excluding sparse careers from the prior itself would bias the shrinkage
      target for everyone else.

    career_point_gain is the sum of this fighter's own ude_points_diff
    values, summed ONLY over the fights present in `df` -- not a snapshot
    of their career-wide cumulative ledger. For the full, unfiltered
    dataset these are mathematically identical (the ledger IS the running
    sum of every diff from a 500-point start), so this is a no-op for
    rank_fighters_by_shrunk_ude_rate's own direct use (§2a). They diverge
    for a filtered subset, though -- which is exactly what
    rank_fighters_by_shrunk_ude_rate_by_weight_class and
    rank_fighters_by_shrunk_ude_rate_by_gender pass in. A ledger snapshot
    at a fighter's last fight in the filtered subset would still carry
    whatever they earned or lost in fights OUTSIDE it (a different division,
    a fight excluded by the year filter): confirmed this materially
    distorted real results before the fix -- e.g. Frankie Edgar's old LW
    title run was inflating his FW-scoped ranking, and Jose Aldo's mixed BW
    stint was deflating his. Summing directly within `df` fixes this by
    construction: it can only reflect fights actually passed in. See
    project_history.md #51.

    Returns:
    - pd.DataFrame with columns: fighter, fighter_url, age, record, n_fights,
      career_point_gain, raw_rate, shrunk_rate, rank. Sorted descending by
      shrunk_rate.
    """
    # age/record still come from the fighter's last fight WITHIN df -- correct
    # as-is for both the full population and a filtered subset.
    latest_ude_points = get_latest_ude_points_with_details(df)
    ude_points_df = pd.DataFrame.from_dict(latest_ude_points, orient='index')
    ude_points_df.reset_index(inplace=True)
    ude_points_df.rename(columns={'index': 'fighter_url'}, inplace=True)

    # career_point_gain: summed directly from df's own ude_points_diff
    # columns (both sides), not read off the career-wide ledger -- see the
    # docstring above for why this matters for a filtered subset.
    fighter_1_diffs = df[['fighter_url_fighter_1', 'ude_points_diff_fighter_1']].rename(
        columns={'fighter_url_fighter_1': 'fighter_url', 'ude_points_diff_fighter_1': 'diff'})
    fighter_2_diffs = df[['fighter_url_fighter_2', 'ude_points_diff_fighter_2']].rename(
        columns={'fighter_url_fighter_2': 'fighter_url', 'ude_points_diff_fighter_2': 'diff'})
    point_gains = pd.concat([fighter_1_diffs, fighter_2_diffs]).groupby('fighter_url')['diff'].sum()
    point_gains.name = 'career_point_gain'

    ude_points_df = ude_points_df.merge(point_gains, left_on='fighter_url', right_index=True, how='left')

    # Count total scored fights per fighter_url from both sides of the dataset.
    fighter_1_urls = df[['fighter_url_fighter_1']].rename(columns={'fighter_url_fighter_1': 'fighter_url'})
    fighter_2_urls = df[['fighter_url_fighter_2']].rename(columns={'fighter_url_fighter_2': 'fighter_url'})
    fight_counts = pd.concat([fighter_1_urls, fighter_2_urls]).groupby('fighter_url').size()
    fight_counts.name = 'n_fights'

    ude_points_df = ude_points_df.merge(fight_counts, left_on='fighter_url', right_index=True, how='left')

    # Raw (unshrunk) per-fight rate.
    ude_points_df['raw_rate'] = ude_points_df['career_point_gain'] / ude_points_df['n_fights']

    # Bayesian shrinkage toward the population mean rate, weighted by prior_strength
    # equivalent fights -- identical in form to shrunk_win_rate()'s treatment of
    # win rate, just applied to points-per-fight instead of wins-per-fight.
    # Computed over the FULL population, before any min_fights filtering, so
    # the shrinkage target itself isn't biased by which careers get excluded.
    population_mean_rate = ude_points_df['raw_rate'].mean()
    ude_points_df['shrunk_rate'] = (
        (ude_points_df['career_point_gain'] + prior_strength * population_mean_rate)
        / (ude_points_df['n_fights'] + prior_strength)
    )

    if min_fights is not None:
        ude_points_df = ude_points_df[ude_points_df['n_fights'] >= min_fights]

    ude_points_df = ude_points_df.sort_values(by='shrunk_rate', ascending=False).reset_index(drop=True)
    ude_points_df['rank'] = ude_points_df['shrunk_rate'].rank(method='dense', ascending=False).astype(int)

    return ude_points_df[['fighter', 'fighter_url', 'age', 'record', 'n_fights',
                           'career_point_gain', 'raw_rate', 'shrunk_rate', 'rank']]

'''7. Weight-class / era-scoped shrunk-rate ranking'''

def filter_by_weight_class(df, weight_class=None):
    """
    Filter fights to one weight class (a weight_class_cleaned code, e.g.
    'FW', 'LHW'). None returns the full dataframe unchanged.

    Deliberately does NOT also require both fighters to individually clear
    a minimum fight count in this weight class -- an earlier version of
    this did, which drops fights against one-off/passing-through opponents
    entirely, silently undercounting the fighter actually being ranked
    (e.g. a champion's fight against a debuting challenger would vanish
    from the champion's own count). Excluding sparse careers from the
    final ranking is rank_fighters_by_shrunk_ude_rate's own min_fights
    floor's job, applied after scoring on the fighter being ranked, not on
    both sides of every fight up front.
    """
    if weight_class is None:
        return df.copy()
    return df[df['weight_class_cleaned'] == weight_class].copy()


def filter_by_year(df, start_year=None, end_year=None):
    """Filter fights to an event_date year range. Either bound may be omitted."""
    if start_year is None and end_year is None:
        return df.copy()
    event_years = pd.to_datetime(df['event_date']).dt.year
    mask = pd.Series(True, index=df.index)
    if start_year is not None:
        mask &= event_years >= start_year
    if end_year is not None:
        mask &= event_years <= end_year
    return df[mask].copy()


def rank_fighters_by_shrunk_ude_rate_by_weight_class(df, weight_class=None, start_year=None,
                                                      end_year=None, prior_strength=10.0, min_fights=None):
    """
    rank_fighters_by_shrunk_ude_rate(), scoped to one weight class and/or
    event-date year range -- e.g. "who scores best fighting at FW between
    2015 and 2020."

    population_mean_rate (the shrinkage target) is computed from THIS
    filtered population, not the full roster -- deliberately, so a
    featherweight is shrunk toward the featherweight mean rate rather than
    the promotion-wide mean across every division, which would be a much
    less meaningful prior for a division-scoped question.

    Note: UDE ratings are one running per-fighter ledger, not reset per
    weight class -- but rank_fighters_by_shrunk_ude_rate's career_point_gain
    is summed directly from the fights actually passed to it (see that
    function's own docstring), so a fighter's ledger value from OTHER
    divisions does not bleed into their score here. Confirmed this mattered
    in practice, not just in theory: before this was fixed, Frankie Edgar's
    old LW title run was inflating his FW-scoped ranking, and Jose Aldo's
    mixed BW stint was deflating his -- see project_history.md #51. What's
    still NOT accounted for is competition strength -- a dominant run in a
    shallow division-era scores the same as one in a stacked one -- see
    canonical_project_state.md section 6 for that separate, much larger,
    open problem.

    Args: see rank_fighters_by_shrunk_ude_rate. weight_class/start_year/end_year
    are applied as pre-filters before that function runs.
    """
    filtered_df = filter_by_weight_class(df, weight_class)
    filtered_df = filter_by_year(filtered_df, start_year, end_year)
    return rank_fighters_by_shrunk_ude_rate(filtered_df, prior_strength=prior_strength, min_fights=min_fights)


MENS_WEIGHT_CLASSES = ['LW', 'WW', 'MW', 'FW', 'BW', 'LHW', 'HW', 'FLW']
WOMENS_WEIGHT_CLASSES = ['WSW', 'WFLW', 'WBW', 'WFW']


def filter_by_gender(df, gender=None):
    """
    Filter fights to one gender's divisions, by weight_class_cleaned code --
    every women's division code is 'W' + the men's code (e.g. 'BW' vs
    'WBW'), a naming convention already baked in by map_weight_class. None
    returns the full dataframe unchanged.

    'Catch Weight Bout' and 'Open Weight Bout' rows (85 fights in the
    current dataset) aren't attributable to either gender and are excluded
    from BOTH 'M' and 'F', not just the one not requested. A fighter who
    had one of these bouts will show one fewer n_fights here than in an
    unfiltered ranking that includes it.
    """
    if gender is None:
        return df.copy()
    if gender == 'M':
        return df[df['weight_class_cleaned'].isin(MENS_WEIGHT_CLASSES)].copy()
    if gender == 'F':
        return df[df['weight_class_cleaned'].isin(WOMENS_WEIGHT_CLASSES)].copy()
    raise ValueError(f"gender must be 'M', 'F', or None, got {gender!r}")


def rank_fighters_by_shrunk_ude_rate_by_gender(df, gender=None, start_year=None,
                                                end_year=None, prior_strength=10.0, min_fights=None):
    """
    rank_fighters_by_shrunk_ude_rate(), scoped to one gender's divisions
    and/or an event-date year range -- e.g. "who scores best among female
    fighters between 2015 and 2020."

    population_mean_rate (the shrinkage target) is computed from THIS
    filtered population, not the full roster -- same reasoning as
    rank_fighters_by_shrunk_ude_rate_by_weight_class: a female fighter
    should be shrunk toward the female population's own mean rate, not a
    promotion-wide mean dominated by the much larger male fight count.

    Args: see rank_fighters_by_shrunk_ude_rate. gender is 'M', 'F', or None
    (no gender filter); start_year/end_year are applied the same way as in
    the weight-class version.
    """
    filtered_df = filter_by_gender(df, gender)
    filtered_df = filter_by_year(filtered_df, start_year, end_year)
    return rank_fighters_by_shrunk_ude_rate(filtered_df, prior_strength=prior_strength, min_fights=min_fights)


'''8. Fighter active/inactive status'''

FIGHTER_INACTIVE_AFTER_DAYS = 730  # ~2 years since last fight

def get_latest_fight_info(df):
    """Each fighter's most recent fight: bout, event date, fight URL, weight class."""
    df_sorted = df.sort_values(by='event_date', ascending=False)

    latest_fighter_1 = df_sorted.drop_duplicates(subset=['fighter_url_fighter_1'], keep='first')[
        ['fighter_1', 'fighter_url_fighter_1', 'bout', 'event_date', 'fight_url', 'weight_class_cleaned']
    ].rename(columns={'fighter_1': 'fighter', 'fighter_url_fighter_1': 'fighter_url',
                       'weight_class_cleaned': 'weight_class'})

    latest_fighter_2 = df_sorted.drop_duplicates(subset=['fighter_url_fighter_2'], keep='first')[
        ['fighter_2', 'fighter_url_fighter_2', 'bout', 'event_date', 'fight_url', 'weight_class_cleaned']
    ].rename(columns={'fighter_2': 'fighter', 'fighter_url_fighter_2': 'fighter_url',
                       'weight_class_cleaned': 'weight_class'})

    return pd.concat([latest_fighter_1, latest_fighter_2], ignore_index=True)


def determine_fighter_status(event_date, as_of=None):
    """
    1 (active) if `event_date` is within FIGHTER_INACTIVE_AFTER_DAYS of
    `as_of`, else 0 (inactive).

    `as_of` defaults to datetime.now() -- unlike the temporal discipline
    enforced elsewhere in this project (calibration, scoring), that makes
    the result non-deterministic across runs: the same dataset can return
    a different 'active' flag depending on what day you run it. That's
    intentional here, since "is this fighter still active" is genuinely an
    as-of-today question rather than a historical reconstruction -- but it
    does mean this function's output shouldn't be cached or diffed across
    runs taken on different days. Pass `as_of` explicitly for a
    reproducible/backtested cutoff instead.
    """
    if as_of is None:
        as_of = datetime.now()
    event_date = pd.to_datetime(event_date)
    return 1 if (as_of - event_date).days <= FIGHTER_INACTIVE_AFTER_DAYS else 0


def add_fighter_status(fighter_df, as_of=None):
    """Add an 'active' column based on each row's event_date. See determine_fighter_status."""
    fighter_df = fighter_df.copy()
    fighter_df['active'] = fighter_df['event_date'].apply(lambda d: determine_fighter_status(d, as_of=as_of))
    return fighter_df


def create_fighter_status_dataset(df, as_of=None):
    """
    One row per fighter: their most recent fight and whether they're still
    active (fought within FIGHTER_INACTIVE_AFTER_DAYS of `as_of`, default
    today). Pass `as_of` explicitly for a reproducible cutoff -- see
    determine_fighter_status.
    """
    latest_fight_df = get_latest_fight_info(df)
    final_df = add_fighter_status(latest_fight_df, as_of=as_of)

    final_df = final_df.sort_values(by='event_date', ascending=False).drop_duplicates(
        subset=['fighter_url'], keep='first'
    )
    final_df = final_df.rename(columns={'bout': 'most_recent_bout', 'event_date': 'most_recent_event_date'})

    return final_df[['fighter', 'fighter_url', 'weight_class', 'most_recent_bout',
                      'most_recent_event_date', 'fight_url', 'active']]


'''9. Fighter finishing potency by weight class'''

RATE_SHRINKAGE_PRIOR_STRENGTH = 5.0

def _shrink_rate(count, total, prior_strength, prior_rate):
    """
    Bayesian-shrinkage rate estimate -- same shape as
    ude_points_algorithm.shrunk_win_rate (there: wins/fights shrunk toward
    a .500 prior), generalized here to an arbitrary count/total pair
    (accepts scalars or pandas Series) so a finishing rate can be shrunk
    toward a population baseline instead of a fixed constant.
    """
    return (count + prior_strength * prior_rate) / (total + prior_strength)


def calculate_striking_potency(df, prior_strength=RATE_SHRINKAGE_PRIOR_STRENGTH):
    """
    Per-fighter, per-weight-class KO/TKO finishing rate among their wins
    (ko_tko_wins / wins), shrunk toward that weight class's own population
    finishing rate.

    Deliberately NOT ko_tko_wins / sig_strikes_landed (the original
    formulation) -- that divided a fight-level count (wins) by a
    strike-level count (total significant strikes landed across those
    wins), two incommensurate units. The result rewarded low-volume
    fighters disproportionately: landing only a handful of strikes across
    a few winning fights produces a tiny denominator and an inflated
    ratio, independent of whether the fighter is actually more finish-prone
    than someone with a genuinely larger sample. min-max normalizing that
    ratio afterward made it worse, not better -- see project discussion
    (grappling's equivalent bug produced a 25.0 "potency" from a single
    submission win over just 4 career grappling actions). Shrinking a
    same-units rate toward a population baseline replaces both the ratio
    construction and the normalization step at once.
    """
    wins_1 = df['fight_result_fighter_1'] == 'W'
    wins_2 = df['fight_result_fighter_2'] == 'W'
    # 'details' is the raw scrape's finish-method text, after
    # dataset_processing_pipeline.standardize_columns renames
    # 'finish_details' -> 'details'. An earlier version of this referenced
    # the pre-rename column name ('method_details'), which the current
    # pipeline no longer produces.
    not_injury = ~df['details'].astype(str).str.contains('injury', case=False, na=False)

    striking_1 = (
        df[wins_1 & not_injury]
        .groupby(['fighter_1', 'weight_class_cleaned'])
        .agg(
            sig_strikes_landed=('sig_strikes_landed_fighter_1', 'sum'),
            ko_tko_wins=('method', lambda x: (x == 'KO/TKO').sum()),
            fighter_url=('fighter_url_fighter_1', 'first'),
            wins=('fighter_1', 'count'),
        )
        .reset_index()
        .rename(columns={'fighter_1': 'fighter'})
    )
    striking_2 = (
        df[wins_2 & not_injury]
        .groupby(['fighter_2', 'weight_class_cleaned'])
        .agg(
            sig_strikes_landed=('sig_strikes_landed_fighter_2', 'sum'),
            ko_tko_wins=('method', lambda x: (x == 'KO/TKO').sum()),
            fighter_url=('fighter_url_fighter_2', 'first'),
            wins=('fighter_2', 'count'),
        )
        .reset_index()
        .rename(columns={'fighter_2': 'fighter'})
    )

    striking_data = pd.concat([striking_1, striking_2], ignore_index=True)
    striking_data = striking_data.groupby(['fighter', 'weight_class_cleaned']).agg(
        sig_strikes_landed=('sig_strikes_landed', 'sum'),
        ko_tko_wins=('ko_tko_wins', 'sum'),
        fighter_url=('fighter_url', 'first'),
        wins=('wins', 'sum'),
    ).reset_index()

    # Division-specific baseline: KO rates differ hugely by weight class
    # (heavyweight vs. flyweight), so shrinking everyone toward one global
    # mean would systematically overrate finishers in low-finish divisions
    # and underrate them in high-finish ones.
    division_totals = striking_data.groupby('weight_class_cleaned').agg(
        division_ko=('ko_tko_wins', 'sum'), division_wins=('wins', 'sum')
    ).reset_index()
    division_totals['division_ko_rate'] = (
        division_totals['division_ko'] / division_totals['division_wins'].replace(0, np.nan)
    ).fillna(0.0)
    striking_data = striking_data.merge(
        division_totals[['weight_class_cleaned', 'division_ko_rate']], on='weight_class_cleaned', how='left'
    )

    striking_data['striking_potency'] = _shrink_rate(
        striking_data['ko_tko_wins'], striking_data['wins'], prior_strength, striking_data['division_ko_rate']
    )

    return striking_data[['fighter', 'fighter_url', 'wins', 'weight_class_cleaned',
                           'sig_strikes_landed', 'ko_tko_wins', 'striking_potency']]


def calculate_grappling_potency(df, prior_strength=RATE_SHRINKAGE_PRIOR_STRENGTH):
    """
    Per-fighter, per-weight-class submission finishing rate among their
    wins (submission_wins / wins), shrunk toward that weight class's own
    population submission rate. Same fix as calculate_striking_potency --
    see its docstring for why this replaces the original
    submission_wins / (td_landed + sub_att) formulation.
    """
    wins_1 = df['fight_result_fighter_1'] == 'W'
    wins_2 = df['fight_result_fighter_2'] == 'W'
    not_injury = ~df['details'].astype(str).str.contains('injury', case=False, na=False)

    grappling_1 = (
        df[wins_1 & not_injury]
        .groupby(['fighter_1', 'weight_class_cleaned'])
        .agg(
            sub_att=('sub_att_fighter_1', 'sum'),
            submission_wins=('method', lambda x: (x == 'Submission').sum()),
            td_landed=('td_landed_fighter_1', 'sum'),
            fighter_url=('fighter_url_fighter_1', 'first'),
            wins=('fighter_1', 'count'),
        )
        .reset_index()
        .rename(columns={'fighter_1': 'fighter'})
    )
    grappling_2 = (
        df[wins_2 & not_injury]
        .groupby(['fighter_2', 'weight_class_cleaned'])
        .agg(
            sub_att=('sub_att_fighter_2', 'sum'),
            submission_wins=('method', lambda x: (x == 'Submission').sum()),
            td_landed=('td_landed_fighter_2', 'sum'),
            fighter_url=('fighter_url_fighter_2', 'first'),
            wins=('fighter_2', 'count'),
        )
        .reset_index()
        .rename(columns={'fighter_2': 'fighter'})
    )

    grappling_data = pd.concat([grappling_1, grappling_2], ignore_index=True)
    grappling_data = grappling_data.groupby(['fighter', 'weight_class_cleaned']).agg(
        sub_att=('sub_att', 'sum'),
        submission_wins=('submission_wins', 'sum'),
        td_landed=('td_landed', 'sum'),
        fighter_url=('fighter_url', 'first'),
        wins=('wins', 'sum'),
    ).reset_index()

    division_totals = grappling_data.groupby('weight_class_cleaned').agg(
        division_sub=('submission_wins', 'sum'), division_wins=('wins', 'sum')
    ).reset_index()
    division_totals['division_sub_rate'] = (
        division_totals['division_sub'] / division_totals['division_wins'].replace(0, np.nan)
    ).fillna(0.0)
    grappling_data = grappling_data.merge(
        division_totals[['weight_class_cleaned', 'division_sub_rate']], on='weight_class_cleaned', how='left'
    )

    grappling_data['grappling_potency'] = _shrink_rate(
        grappling_data['submission_wins'], grappling_data['wins'], prior_strength, grappling_data['division_sub_rate']
    )

    return grappling_data[['fighter', 'fighter_url', 'wins', 'weight_class_cleaned',
                            'sub_att', 'submission_wins', 'td_landed', 'grappling_potency']]


def calculate_overall_potency(df, prior_strength=RATE_SHRINKAGE_PRIOR_STRENGTH):
    """
    Combine striking and grappling finishing potency into one
    per-fighter/weight-class score: the geometric mean of both shrunk
    rates.

    Because each input is already a shrunk rate in [0, 1] rather than a raw
    count-ratio, this drops the original version's ad hoc
    volume_weighted_potency() action-count penalty, the min-max
    normalization step, and adjust_potency()'s special-cased 0.7 fallback
    multiplier for a zero-valued side -- all three existed to patch over
    small-sample noise that the shrinkage now absorbs directly, in a way
    consistent with how this project already shrinks small samples
    elsewhere (shrunk_win_rate, rank_fighters_by_shrunk_ude_rate).
    """
    striking_data = calculate_striking_potency(df, prior_strength=prior_strength)
    grappling_data = calculate_grappling_potency(df, prior_strength=prior_strength)

    potency_df = pd.merge(
        striking_data, grappling_data,
        on=['fighter', 'fighter_url', 'weight_class_cleaned'],
        how='outer', suffixes=('_striking', '_grappling'),
    )
    numeric_cols = ['wins_striking', 'wins_grappling', 'sig_strikes_landed', 'ko_tko_wins',
                     'sub_att', 'submission_wins', 'td_landed', 'striking_potency', 'grappling_potency']
    potency_df[numeric_cols] = potency_df[numeric_cols].fillna(0.0)

    # wins_striking and wins_grappling come from the same win-filtered rows
    # (calculate_striking_potency/calculate_grappling_potency both group
    # the exact same (fighter, weight_class) win set), so they're always
    # equal when both are present -- max() here only matters for the
    # edge case where a fighter has wins recorded on one side but the
    # outer merge left the other NaN-turned-0.
    potency_df['wins'] = potency_df[['wins_striking', 'wins_grappling']].max(axis=1).astype(int)
    potency_df['overall_potency'] = np.sqrt(potency_df['striking_potency'] * potency_df['grappling_potency'])

    return potency_df[['fighter', 'fighter_url', 'wins', 'weight_class_cleaned',
                        'sig_strikes_landed', 'ko_tko_wins', 'td_landed', 'sub_att',
                        'submission_wins', 'striking_potency', 'grappling_potency',
                        'overall_potency']]


def get_potency_by_weight_class(potency_ratings, weight_class, min_wins=5):
    """Filter a calculate_overall_potency() result to one weight class, sorted best-first.

    min_wins is a floor on top of the shrinkage in calculate_overall_potency,
    not a replacement for it -- shrinkage pulls a 1-win fighter's rate
    toward the division baseline instead of leaving it at a raw 100%/0%,
    but a 1-win career still isn't a meaningful "most potent" claim on its
    own. Same lesson as rank_fighters_by_shrunk_ude_rate's min_fights floor:
    shrinkage alone isn't sufficient stabilization at very small n.
    """
    weight_class_potency = potency_ratings[potency_ratings['weight_class_cleaned'] == weight_class]
    weight_class_potency = weight_class_potency[weight_class_potency['wins'] >= min_wins]
    return weight_class_potency.sort_values(by='overall_potency', ascending=False).reset_index(drop=True)


'''10. Striking power ("who hits hardest") by weight class'''

def calculate_striking_power(df, prior_strength=RATE_SHRINKAGE_PRIOR_STRENGTH):
    """
    Per-fighter, per-weight-class knockdown rate per head strike landed
    (kd / head_strikes_landed), shrunk toward that weight class's own
    population rate. Answers "who hits hardest," a different question from
    calculate_striking_potency's "who finishes fights":

    - Not win-conditioned. A knockdown the opponent survives -- fight
      continues, maybe even ends in a loss on the scorecards -- still
      demonstrates power exactly as much as one that ends the fight.
      Potency is win-conditioned by design (it's measuring how fights got
      won); power is a property of the strikes themselves and shouldn't be
      restricted to wins only.
    - head_strikes_landed, not sig_strikes_landed or total volume, as the
      denominator. Knockdowns come overwhelmingly from head strikes, not
      leg kicks or body shots -- using total significant strikes would
      dilute a genuinely heavy-handed fighter's rate with strikes that
      were never going to produce a knockdown regardless of power, and
      would inflate a leg-kick-heavy fighter's apparent "power" on an
      unrelated axis (denominator shrinks, ratio rises, nothing about
      hand power changed).

    Excludes NC results (no-contest -- typically DQ'd/overturned
    circumstances unrelated to the striking itself) and injury stoppages
    (details containing 'injury' -- e.g. a leg/knee/eye injury TKO is
    method == 'KO/TKO' in the raw data despite having nothing to do with
    striking power; same exclusion calculate_striking_potency already
    applies, for the same reason). Draws are kept, since a draw still
    reflects real landed strikes and real knockdowns.
    """
    not_nc = (df['fight_result_fighter_1'] != 'NC') & (df['fight_result_fighter_2'] != 'NC')
    not_injury = ~df['details'].astype(str).str.contains('injury', case=False, na=False)
    valid = not_nc & not_injury

    power_1 = (
        df[valid]
        .groupby(['fighter_1', 'weight_class_cleaned'])
        .agg(
            head_strikes_landed=('head_strikes_landed_fighter_1', 'sum'),
            kd=('kd_fighter_1', 'sum'),
            fighter_url=('fighter_url_fighter_1', 'first'),
            fights=('fighter_1', 'count'),
        )
        .reset_index()
        .rename(columns={'fighter_1': 'fighter'})
    )
    power_2 = (
        df[valid]
        .groupby(['fighter_2', 'weight_class_cleaned'])
        .agg(
            head_strikes_landed=('head_strikes_landed_fighter_2', 'sum'),
            kd=('kd_fighter_2', 'sum'),
            fighter_url=('fighter_url_fighter_2', 'first'),
            fights=('fighter_2', 'count'),
        )
        .reset_index()
        .rename(columns={'fighter_2': 'fighter'})
    )

    power_data = pd.concat([power_1, power_2], ignore_index=True)
    power_data = power_data.groupby(['fighter', 'weight_class_cleaned']).agg(
        head_strikes_landed=('head_strikes_landed', 'sum'),
        kd=('kd', 'sum'),
        fighter_url=('fighter_url', 'first'),
        fights=('fights', 'sum'),
    ).reset_index()

    # Division-specific baseline, same reasoning as calculate_striking_potency:
    # heavyweight and flyweight do not knock people down at the same rate
    # per landed head strike, so one global mean would misjudge both.
    division_totals = power_data.groupby('weight_class_cleaned').agg(
        division_kd=('kd', 'sum'), division_head_strikes=('head_strikes_landed', 'sum')
    ).reset_index()
    division_totals['division_kd_rate'] = (
        division_totals['division_kd'] / division_totals['division_head_strikes'].replace(0, np.nan)
    ).fillna(0.0)
    power_data = power_data.merge(
        division_totals[['weight_class_cleaned', 'division_kd_rate']], on='weight_class_cleaned', how='left'
    )

    power_data['striking_power'] = _shrink_rate(
        power_data['kd'], power_data['head_strikes_landed'], prior_strength, power_data['division_kd_rate']
    )

    return power_data[['fighter', 'fighter_url', 'fights', 'weight_class_cleaned',
                        'head_strikes_landed', 'kd', 'striking_power']]


def get_power_by_weight_class(power_ratings, weight_class, min_head_strikes_landed=100):
    """
    Filter a calculate_striking_power() result to one weight class, sorted
    hardest-hitting first.

    min_head_strikes_landed, not min_fights/min_wins, is the floor here --
    the ratio's own denominator is head strikes landed, so that's the
    quantity that needs a minimum sample before the shrunk rate means
    anything, the same logic as get_potency_by_weight_class's min_wins but
    keyed to this metric's actual unit of exposure.
    """
    weight_class_power = power_ratings[power_ratings['weight_class_cleaned'] == weight_class]
    weight_class_power = weight_class_power[weight_class_power['head_strikes_landed'] >= min_head_strikes_landed]
    return weight_class_power.sort_values(by='striking_power', ascending=False).reset_index(drop=True)


'''11. Rematch history and outcomes'''

def find_rematch_pairs(df):
    """All fighter-url pairs (as frozensets) that met at least twice."""
    rematch_fights = df[df['is_rematch'] == 1]
    pairs = set()
    for _, row in rematch_fights.iterrows():
        pairs.add(frozenset([row['fighter_url_fighter_1'], row['fighter_url_fighter_2']]))
    return pairs


def get_all_fights_between_pairs(df, fighter_pairs):
    """Every meeting (not just the rematch itself) for the given fighter-url pairs."""
    pair_series = df[['fighter_url_fighter_1', 'fighter_url_fighter_2']].apply(lambda x: frozenset(x), axis=1)
    fights = df[pair_series.isin(fighter_pairs)]
    return fights.sort_values(by=['fighter_url_fighter_1', 'fighter_url_fighter_2', 'event_date'])


def is_immediate_rematch(fight, df):
    """True if neither fighter had any other fight between their last meeting and this one."""
    fighter1 = fight['fighter_url_fighter_1']
    fighter2 = fight['fighter_url_fighter_2']
    fight_date = fight['event_date']

    pair_series = df[['fighter_url_fighter_1', 'fighter_url_fighter_2']].apply(lambda x: frozenset(x), axis=1)
    previous_fights = df[(pair_series == frozenset([fighter1, fighter2])) & (df['event_date'] < fight_date)]
    if previous_fights.empty:
        return 0

    last_fight_date = previous_fights['event_date'].max()
    f1_intervening = df[(df['event_date'] > last_fight_date) & (df['event_date'] < fight_date) &
                         ((df['fighter_url_fighter_1'] == fighter1) | (df['fighter_url_fighter_2'] == fighter1))]
    f2_intervening = df[(df['event_date'] > last_fight_date) & (df['event_date'] < fight_date) &
                         ((df['fighter_url_fighter_1'] == fighter2) | (df['fighter_url_fighter_2'] == fighter2))]
    return 1 if f1_intervening.empty and f2_intervening.empty else 0


WINNER_DRAW = 'draw'
WINNER_NO_CONTEST = 'no_contest'

def assign_winner(fight):
    """
    (winner_url, winner_name) for one fight, or one of two DISTINCT
    sentinels for a non-decisive result.

    Draw and no-contest are deliberately different sentinels. An earlier
    version of this collapsed both to a single value (effectively None),
    which meant two draws -- or two no-contests -- between the same pair
    would spuriously compare equal in a "did the same fighter win both
    meetings" check downstream (find_same_winner_rematches). Checked
    against the live dataset: no fighter pair currently has 2+ draws
    against each other (61 draws total, zero repeated pairs), so this was
    dormant, not live -- but nothing prevented it from becoming live.
    """
    if fight['fight_result_fighter_1'] == 'W':
        return fight['fighter_url_fighter_1'], fight['fighter_1']
    elif fight['fight_result_fighter_1'] == 'L':
        return fight['fighter_url_fighter_2'], fight['fighter_2']
    elif fight['fight_result_fighter_1'] == 'D':
        return WINNER_DRAW, WINNER_DRAW
    return WINNER_NO_CONTEST, WINNER_NO_CONTEST


def add_fighter_pair_column(df):
    """Creates a fighter_pair column combining fighter_1 and fighter_2 (order-independent)."""
    df = df.copy()
    df['fighter_pair'] = df.apply(lambda x: frozenset([x['fighter_1'], x['fighter_2']]), axis=1)
    return df


def filter_invalid_rematches(df):
    """
    Drops no-score fights from each pair's rematch sequence, and drops
    everything chronologically after the first such fight for that pair --
    a broken sequence can't be trusted to represent a clean rematch history
    from that point on.

    Uses ude_points_algorithm.is_no_score_fight (checks both fight_result
    == 'NC' and method in {'DQ', 'Overturned'}) rather than re-deriving the
    check. An earlier version of this only checked
    method_mapped in ['Overturned', 'Could Not Continue'] -- checked
    against the live dataset, that happens to catch every actual
    no-contest currently present (89/89 NC-result fights have one of
    those two method values), but that's what the raw data happens to
    contain, not something the check itself guaranteed, and it never
    accounted for a 'DQ' method at all.
    """
    filtered_fights = []
    for fighter_pair, group in df.groupby('fighter_pair', group_keys=False):
        group = group.sort_values(by='event_date')
        invalid_indices = set()
        prev_valid = True

        for idx, row in group.iterrows():
            method = row.get('method', row.get('method_mapped'))
            is_invalid = (is_no_score_fight(row['fight_result_fighter_1'], method) or
                          is_no_score_fight(row['fight_result_fighter_2'], method))
            if is_invalid:
                invalid_indices.add(idx)
                prev_valid = False
            elif not prev_valid:
                invalid_indices.add(idx)
            else:
                prev_valid = True

        valid_fights = group.drop(index=invalid_indices)
        if not valid_fights.empty:
            filtered_fights.append(valid_fights)

    if not filtered_fights:
        return df.iloc[0:0]
    return pd.concat(filtered_fights).reset_index(drop=True)


def process_rematch_data(df, exclude_no_contests=False):
    """
    Finds every fighter pair with a rematch and returns all their
    meetings, whether each was an immediate rematch (no intervening fight
    for either fighter), and who won.
    """
    fighter_pairs = find_rematch_pairs(df)
    rematch_fights = get_all_fights_between_pairs(df, fighter_pairs).copy()
    rematch_fights['immediate'] = rematch_fights.apply(lambda x: is_immediate_rematch(x, df), axis=1)
    rematch_fights[['winner', 'winner_name']] = rematch_fights.apply(
        lambda x: pd.Series(assign_winner(x)), axis=1
    )
    rematch_fights = add_fighter_pair_column(rematch_fights)

    if exclude_no_contests:
        rematch_fights = filter_invalid_rematches(rematch_fights)

    return rematch_fights.reset_index(drop=True)


def find_same_winner_rematches(all_rematches):
    """
    Fighter pairs whose first two meetings were won by the same fighter.
    Takes process_rematch_data()'s output as input.

    Deliberately built on top of process_rematch_data's already-correct,
    fully general is_immediate_rematch/winner logic rather than
    re-implementing rematch detection -- an earlier standalone version of
    this question (filter_immediate_fighter_rematches) hardcoded a check
    that a pair's prior fight had is_rematch == 0, which is only correct
    for a first rematch specifically and would silently misfire if reused
    for a second rematch (trilogy) onward. It was never actually called
    with anything but the default here, so that bug was never live, but
    routing through process_rematch_data avoids reintroducing it.

    Draws/no-contests never count as a "same winner" match -- they're
    distinct sentinels (see assign_winner), not a value that can
    spuriously equal itself across two different fights.
    """
    result = []
    for _, group in all_rematches.groupby('fighter_pair'):
        fights = group.sort_values(by='event_date').reset_index(drop=True)
        if len(fights) < 2:
            continue
        first_winner = fights.loc[0, 'winner']
        second_winner = fights.loc[1, 'winner']
        if first_winner in (WINNER_DRAW, WINNER_NO_CONTEST):
            continue
        if first_winner == second_winner:
            result.append(fights.loc[[0, 1]])

    if not result:
        return all_rematches.iloc[0:0]
    return pd.concat(result).reset_index(drop=True)


'''12. Opponent-durability-adjusted striking power'''

STANDING_KO_TKO_EXCLUDE_KEYWORDS = ['injury', 'ground', 'mount', 'guard', 'control', 'bottom', 'fatigue', 'crucifix']

def is_standing_ko_tko(method, details):
    """
    True for a KO/TKO finish landed with both fighters standing --
    excludes ground-based finishes (mount/guard/control/bottom/crucifix)
    and injury stoppages. Isolates standing striking power specifically:
    a ground-and-pound stoppage or a doctor's stoppage says nothing about
    standing punching/kicking power.
    """
    if method != 'KO/TKO':
        return False
    details_lower = str(details).lower()
    return not any(keyword in details_lower for keyword in STANDING_KO_TKO_EXCLUDE_KEYWORDS)


DURABILITY_SHRINKAGE_PRIOR_STRENGTH = 15.0  # in equivalent standing strikes absorbed
DURABILITY_MIN_MULTIPLIER = 0.5
DURABILITY_MAX_MULTIPLIER = 2.0
DURABILITY_MIN_DIVISION_OBSERVATIONS = 3000  # in standing strikes absorbed, before falling back to the dataset-wide rate

def add_opponent_durability_multiplier(df, prior_strength=DURABILITY_SHRINKAGE_PRIOR_STRENGTH):
    """
    Adds opponent_durability_multiplier_fighter_1/_fighter_2: a bounded
    multiplier reflecting how durable each side's OPPONENT had been
    (standing KO/TKO losses per standing significant strike absorbed),
    computed from the opponent's fight history strictly BEFORE this fight.
    >1.0 rewards a finish over a historically durable opponent; <1.0
    discounts one over a historically fragile opponent; 1.0 is neutral.

    Fixes two failures found in the notebook version this was adapted
    from, both confirmed against the live dataset before fixing:
    - 56.6% of fighter-fight observations had a raw opponent vulnerability
      of exactly 0 (opponent had never been finished standing before) --
      the original divided by a hardcoded, uncalibrated 0.0005 there, an
      unbounded up-to-2000x multiplier hitting the MAJORITY of the
      dataset, not a rare edge case.
    - 15.4% had a raw opponent vulnerability of NaN (opponent had zero
      cumulative standing-strikes-absorbed history yet) -- silently
      propagated to NaN through the division instead of a neutral
      fallback. Same bug class as the _age_multiplier NaN issue already
      found and fixed in ude_points_algorithm.py.

    Both are replaced with Bayesian shrinkage toward the population's own
    standing KO/TKO-loss rate (the same _shrink_rate machinery used by
    calculate_striking_power/calculate_striking_potency): zero or absent
    history shrinks cleanly to the population baseline (multiplier 1.0),
    not to an arbitrary constant or NaN. The resulting ratio is bounded to
    [DURABILITY_MIN_MULTIPLIER, DURABILITY_MAX_MULTIPLIER], matching how
    every other multiplier in this project (age_adjustment,
    opponent_quality_multiplier) is bounded rather than left unbounded.

    Baseline is scoped to the CURRENT FIGHT's weight class, not one
    dataset-wide number -- checked directly against the live dataset: the
    standing KO/TKO-loss rate per strike absorbed spans an 8.4x range
    across divisions (HW 0.64% vs. WSW 0.076%), against which a single
    global baseline (0.33%) is badly miscalibrated in a predictable
    direction for almost every division -- systematically too low for
    heavier weight classes (pushing multipliers toward the ceiling) and
    too high for lighter ones (pushing them toward the floor). Confirmed
    this was the actual cause, not a hypothetical: with the single global
    baseline, 64% of all fighter-fight observations sat pinned at one of
    the two bounds (51.4% at the ceiling, 12.5% at the floor), leaving
    only 36% in the discriminating middle range.

    Each fighter's own cumulative absorbed-strikes/losses STATE is still
    tracked globally per fighter_url, not reset per weight class --
    consistent with how every other dynamic_* running-state column in
    this project works (dynamic_td_accuracy, career_sig_strikes_landed,
    etc.), and because a fighter's underlying durability doesn't plausibly
    reset when they change divisions; only the yardstick it's judged
    against should be division-specific, not the history itself. A
    division with too little absorbed-strike volume to trust its own rate
    (< DURABILITY_MIN_DIVISION_OBSERVATIONS) falls back to the dataset-wide
    rate instead -- the same "insufficient prior history -> broader
    fallback" pattern _build_temporal_calibration_cache already uses for
    age/method calibration.
    """
    d = df.sort_values(by='event_date').copy()
    d['event_date'] = pd.to_datetime(d['event_date'])

    # str(details) turns a NaN details value into the literal string 'nan',
    # which matches none of the exclude keywords -- a KO/TKO row with
    # missing details defaults to counting as standing rather than raising
    # or silently becoming NaN. KO/TKO rows essentially always have
    # non-null details in practice, so this default rarely if ever fires.
    d['_standing_ko_tko'] = d.apply(lambda r: is_standing_ko_tko(r['method'], r['details']), axis=1)

    long = pd.DataFrame({
        'weight_class': pd.concat([d['weight_class_cleaned'], d['weight_class_cleaned']], ignore_index=True),
        'absorbed': pd.concat([d['standing_sig_strikes_landed_fighter_1'].fillna(0),
                                d['standing_sig_strikes_landed_fighter_2'].fillna(0)], ignore_index=True),
        'loss': pd.concat([(d['_standing_ko_tko'] & (d['fight_result_fighter_1'] == 'L')).astype(float),
                            (d['_standing_ko_tko'] & (d['fight_result_fighter_2'] == 'L')).astype(float)],
                           ignore_index=True),
    })
    global_absorbed = long['absorbed'].sum()
    global_baseline = long['loss'].sum() / global_absorbed if global_absorbed > 0 else 0.0

    by_wc = long.groupby('weight_class').agg(absorbed=('absorbed', 'sum'), loss=('loss', 'sum'))
    by_wc['rate'] = np.where(
        by_wc['absorbed'] >= DURABILITY_MIN_DIVISION_OBSERVATIONS,
        by_wc['loss'] / by_wc['absorbed'].replace(0, np.nan),
        global_baseline,
    )
    baseline_by_weight_class = by_wc['rate'].fillna(global_baseline).to_dict()

    cumulative = {}  # fighter_url -> {'absorbed': float, 'losses': float}

    def _get_state(url):
        return cumulative.setdefault(url, {'absorbed': 0.0, 'losses': 0.0})

    def _multiplier_for(url, baseline):
        state = _get_state(url)
        shrunk = _shrink_rate(state['losses'], state['absorbed'], prior_strength, baseline)
        raw = baseline / shrunk if shrunk > 0 else 1.0
        return max(DURABILITY_MIN_MULTIPLIER, min(DURABILITY_MAX_MULTIPLIER, raw))

    mult_1, mult_2 = [], []
    for _, row in d.iterrows():
        fighter_1_url = row['fighter_url_fighter_1']
        fighter_2_url = row['fighter_url_fighter_2']
        baseline = baseline_by_weight_class.get(row['weight_class_cleaned'], global_baseline)

        # Snapshot BOTH sides' pre-fight multipliers before updating either
        # side's cumulative state -- fighter_1 and fighter_2 are opponents
        # within the same fight, so neither side's performance in THIS
        # fight may leak into the other side's opponent-durability reading.
        mult_1.append(_multiplier_for(fighter_2_url, baseline))
        mult_2.append(_multiplier_for(fighter_1_url, baseline))

        is_standing_finish = row['_standing_ko_tko']
        absorbed_1 = row.get('standing_sig_strikes_landed_fighter_2', 0) or 0
        absorbed_2 = row.get('standing_sig_strikes_landed_fighter_1', 0) or 0
        loss_1 = 1.0 if (is_standing_finish and row['fight_result_fighter_1'] == 'L') else 0.0
        loss_2 = 1.0 if (is_standing_finish and row['fight_result_fighter_2'] == 'L') else 0.0

        s1 = _get_state(fighter_1_url)
        s1['absorbed'] += absorbed_1
        s1['losses'] += loss_1
        s2 = _get_state(fighter_2_url)
        s2['absorbed'] += absorbed_2
        s2['losses'] += loss_2

    d['opponent_durability_multiplier_fighter_1'] = mult_1
    d['opponent_durability_multiplier_fighter_2'] = mult_2
    d = d.drop(columns=['_standing_ko_tko'])
    return d.sort_values(by='event_date', ascending=False).reset_index(drop=True)


def calculate_durability_adjusted_power(df, prior_strength_kd=RATE_SHRINKAGE_PRIOR_STRENGTH,
                                         prior_strength_durability=DURABILITY_SHRINKAGE_PRIOR_STRENGTH):
    """
    calculate_striking_power(), reweighted so a knockdown scored against a
    historically durable opponent counts for more than one scored against
    a historically fragile opponent. See add_opponent_durability_multiplier.
    """
    d = add_opponent_durability_multiplier(df, prior_strength=prior_strength_durability)
    not_nc = (d['fight_result_fighter_1'] != 'NC') & (d['fight_result_fighter_2'] != 'NC')
    not_injury = ~d['details'].astype(str).str.contains('injury', case=False, na=False)
    valid = not_nc & not_injury

    def side(fighter_col, opponent_col, multiplier_col):
        weighted_kd_col = f'_weighted_kd_{fighter_col}'
        subset = d[valid].copy()
        subset[weighted_kd_col] = subset[f'kd_{fighter_col}'] * subset[multiplier_col]
        return (
            subset
            .groupby([fighter_col, 'weight_class_cleaned'])
            .agg(
                head_strikes_landed=(f'head_strikes_landed_{fighter_col}', 'sum'),
                weighted_kd=(weighted_kd_col, 'sum'),
                fighter_url=(f'fighter_url_{fighter_col}', 'first'),
                fights=(fighter_col, 'count'),
            )
            .reset_index()
            .rename(columns={fighter_col: 'fighter'})
        )

    side_1 = side('fighter_1', 'fighter_2', 'opponent_durability_multiplier_fighter_1')
    side_2 = side('fighter_2', 'fighter_1', 'opponent_durability_multiplier_fighter_2')

    data = pd.concat([side_1, side_2], ignore_index=True)
    data = data.groupby(['fighter', 'weight_class_cleaned']).agg(
        head_strikes_landed=('head_strikes_landed', 'sum'),
        weighted_kd=('weighted_kd', 'sum'),
        fighter_url=('fighter_url', 'first'),
        fights=('fights', 'sum'),
    ).reset_index()

    division_totals = data.groupby('weight_class_cleaned').agg(
        division_weighted_kd=('weighted_kd', 'sum'), division_hs=('head_strikes_landed', 'sum')
    ).reset_index()
    division_totals['division_rate'] = (
        division_totals['division_weighted_kd'] / division_totals['division_hs'].replace(0, np.nan)
    ).fillna(0.0)
    data = data.merge(division_totals[['weight_class_cleaned', 'division_rate']], on='weight_class_cleaned', how='left')

    data['durability_adjusted_power'] = _shrink_rate(
        data['weighted_kd'], data['head_strikes_landed'], prior_strength_kd, data['division_rate']
    )
    return data[['fighter', 'fighter_url', 'fights', 'weight_class_cleaned',
                 'head_strikes_landed', 'weighted_kd', 'durability_adjusted_power']]


def get_durability_adjusted_power_by_weight_class(power_ratings, weight_class, min_head_strikes_landed=100):
    """Filter a calculate_durability_adjusted_power() result to one weight class, sorted best-first."""
    weight_class_power = power_ratings[power_ratings['weight_class_cleaned'] == weight_class]
    weight_class_power = weight_class_power[weight_class_power['head_strikes_landed'] >= min_head_strikes_landed]
    return weight_class_power.sort_values(by='durability_adjusted_power', ascending=False).reset_index(drop=True)


'''13. Fighter profile & opponent-similarity matching'''

# Physical and style similarity are kept as SEPARATE comparisons, not
# blended into one score -- a fighter can be a close physical match for a
# future opponent while fighting a completely different style, or vice
# versa, and blending both into one number hides which kind of similarity
# is actually driving the result. Consistent with this project's existing
# principle of not collapsing distinct signals into one opaque composite
# (see canonical_project_state.md's "non-redundant signals" design goal).
PHYSICAL_SIMILARITY_COLUMNS = ['age', 'Height (m)', 'Reach (in)']
STYLE_SIMILARITY_COLUMNS = [
    'dynamic_sig_strikes_accuracy', 'dynamic_sig_strikes_defence',
    'dynamic_td_accuracy', 'dynamic_td_defence',
]

def generate_fighter_profile(df, fighter_name):
    """
    One-row physical + style profile for a fighter: their current age,
    height, reach, stance, and the cumulative dynamic_* skill snapshot
    from their most recent fight.

    A notebook this was adapted from stated an intent to compare fighters
    on "age, height, reach, stance" but never actually captured age
    anywhere in the profile, and stance wasn't reachable at all: the raw
    fighter-bio scrape has a STANCE field, but
    dataset_processing_pipeline.run_etl_pipeline's bio_cols list didn't
    carry it into the merged fight dataset. Both gaps are closed now --
    process_fighter_bio exposes the raw STANCE field as 'Stance' (Title
    Case, matching 'Height (m)'/'Weight (lbs)'/'Reach (in)''s naming
    rather than the all-caps raw scrape column), it was added to bio_cols,
    and the dataset regenerated (see
    latest_fights_up_to_islam_garry_with_ude_points_calculated_v2_6_with_stance.csv,
    validated bit-identical to v2_6.csv on every other column).

    Stance is included here as a plain field, not folded into
    PHYSICAL_SIMILARITY_COLUMNS -- it's categorical (Orthodox/Southpaw/
    Switch/...), not a subtractable numeric quantity, so
    calculate_similarity_differences surfaces it as a separate
    match/mismatch flag rather than blending it into total_difference.

    The dynamic_* values reflect the fighter's cumulative state walking
    INTO their most recent fight, not including how that fight itself
    went -- dynamic_* columns are pre-fight snapshots by convention
    throughout this project (see mma_content_strategy.md's data-sourcing
    rules). df without a Stance column (e.g. plain v2_6.csv) still works;
    the profile's 'Stance' value is just None in that case.
    """
    career = create_fighter_career_dataset(df, fighter_name)
    if career.empty:
        raise ValueError(f"No fights found for fighter '{fighter_name}'")

    latest = career.sort_values(by='event_date', ascending=False).iloc[0]
    profile = {'fighter': fighter_name, 'fighter_url': latest['fighter_url'],
               'Stance': latest.get('Stance')}
    for col in PHYSICAL_SIMILARITY_COLUMNS + STYLE_SIMILARITY_COLUMNS:
        profile[col] = latest.get(col)

    return pd.DataFrame([profile])


def _stance_match_label(future_stance, past_stance):
    """'same'/'different'/'unknown' -- 'unknown' (not a false 'different')
    when either side's stance is missing, since Stance coverage in the
    current dataset is ~98%, not 100%, and a missing value is not evidence
    of a mismatch."""
    if pd.isna(future_stance) or pd.isna(past_stance):
        return 'unknown'
    return 'same' if future_stance == past_stance else 'different'


def calculate_similarity_differences(future_profile, career_dataset, columns_to_compare):
    """
    Signed, SCALED differences between a future opponent's profile and
    each of a fighter's past opponents (from create_fighter_career_dataset's
    opponent_* columns), summed into one total_difference per past
    opponent and sorted smallest-first (most similar first).

    Fixes a real bug in the notebook this was adapted from: it summed raw,
    unscaled absolute differences across columns living on wildly
    different scales (0-1 accuracy rates alongside tens-of-strikes raw
    counts and 60-90 inches of reach), so whichever column happened to
    have the largest raw numeric range dominated total_difference
    regardless of how meaningful the gap actually was. Each compared
    column is min-max scaled to [0, 1] here -- using the combined range of
    the past-opponent population AND the future opponent -- before
    differencing, so every column contributes comparably.

    Separately, the notebook's default column selector used lowercase
    keyword substrings ('height', 'reach') against columns actually named
    'Height (m)'/'Reach (in)' (capitalized) -- a case-sensitive `in` check
    that silently matched neither, so despite the stated intent, physical
    measurables never actually made it into either comparison the
    notebook ran. This function takes columns_to_compare explicitly
    instead of an auto-detected default, so that mismatch can't recur.

    If both future_profile and career_dataset carry stance data (Stance /
    opponent_Stance), a stance_match column ('same'/'different'/'unknown')
    is attached too -- informational only, not part of total_difference or
    the sort. Stance is categorical, not a subtractable numeric quantity,
    so it's surfaced as a flag next to the score rather than blended into
    it (see generate_fighter_profile). Missing on a df built before the
    Stance pipeline addition (e.g. plain v2_6.csv) -- the column is simply
    omitted in that case, not an error.
    """
    if future_profile.shape[0] != 1:
        raise ValueError("Future opponent profile must contain exactly one row.")

    result = career_dataset[['opponent', 'opponent_fighter_url']].copy()

    if 'Stance' in future_profile.columns and 'opponent_Stance' in career_dataset.columns:
        future_stance = future_profile['Stance'].values[0]
        result['stance_match'] = career_dataset['opponent_Stance'].apply(
            lambda past_stance: _stance_match_label(future_stance, past_stance)
        )

    abs_diffs = []

    for column in columns_to_compare:
        career_column = f'opponent_{column}'
        if career_column not in career_dataset.columns:
            raise ValueError(f"Column '{career_column}' not found in fighter career dataset.")

        future_value = future_profile[column].values[0]
        past_values = career_dataset[career_column]

        combined = pd.concat([past_values, pd.Series([future_value])], ignore_index=True)
        col_min, col_max = combined.min(), combined.max()
        col_range = col_max - col_min

        if pd.isna(col_range) or col_range == 0:
            # Every value on this dimension (including the future
            # opponent's) is identical or missing -- no discriminating
            # information here, so this column contributes nothing rather
            # than a divide-by-zero or an arbitrary tie-breaker.
            scaled_diff = pd.Series(0.0, index=career_dataset.index)
        else:
            scaled_diff = (past_values - col_min) / col_range - (future_value - col_min) / col_range

        result[f'diff_{column}'] = scaled_diff.values
        abs_diffs.append(scaled_diff.abs())

    # total_difference is the MEAN of the available |diff| values per row,
    # not a sum with missing columns filled to 0. Filling-to-0 would make
    # a past opponent with NO data on any compared dimension score a
    # perfect 0.0 "total difference" -- ranking them as the single most
    # similar opponent despite there being zero actual information to
    # support that. Averaging only the columns that ARE present avoids
    # crediting missing data as agreement, and a row missing every
    # comparison column correctly becomes NaN (which pandas sorts to the
    # end), rather than a false top rank.
    result['total_difference'] = pd.concat(abs_diffs, axis=1).mean(axis=1, skipna=True)
    return result.sort_values(by='total_difference').reset_index(drop=True)


def calculate_physical_similarity(future_profile, career_dataset):
    """Physical-similarity ranking (age, height, reach) of a fighter's past opponents against a future opponent."""
    return calculate_similarity_differences(future_profile, career_dataset, PHYSICAL_SIMILARITY_COLUMNS)


def calculate_style_similarity(future_profile, career_dataset):
    """Fighting-style-similarity ranking (striking/TD accuracy & defense) of past opponents against a future opponent."""
    return calculate_similarity_differences(future_profile, career_dataset, STYLE_SIMILARITY_COLUMNS)


def find_most_similar_past_opponents(df, fighter_name, future_opponent_name, exclude_future_opponent=True):
    """
    For fighter_name's upcoming fight against future_opponent_name, find
    which of fighter_name's PAST opponents most resemble future_opponent_name
    -- physically and stylistically, reported SEPARATELY (see module-level
    note on PHYSICAL_SIMILARITY_COLUMNS/STYLE_SIMILARITY_COLUMNS for why).

    exclude_future_opponent: if True (default), drops future_opponent_name's
    own past meeting(s) with fighter_name from the comparison. Without
    this, a fighter who has already fought their upcoming opponent before
    will trivially rank their own past fight(s) against that person as
    the "most similar" match -- correct (it IS the same person) but not
    useful if the point is finding a genuinely DIFFERENT stand-in/analog
    fighter. Matched on fighter_url, not name, for the same
    identity-safety reason used throughout this project (two different
    fighters can share a name; URLs can't collide).

    Returns (physical_similarity_df, style_similarity_df), each sorted
    most-similar-first via total_difference.
    """
    future_profile = generate_fighter_profile(df, future_opponent_name)
    career_dataset = create_fighter_career_dataset(df, fighter_name)

    if exclude_future_opponent:
        future_opponent_url = future_profile['fighter_url'].values[0]
        career_dataset = career_dataset[career_dataset['opponent_fighter_url'] != future_opponent_url]

    physical = calculate_physical_similarity(future_profile, career_dataset)
    style = calculate_style_similarity(future_profile, career_dataset)
    return physical, style
