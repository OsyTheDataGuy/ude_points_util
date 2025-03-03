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
