# ./services/pattern_recognition.py

import pandas as pd
import numpy as np
from typing import List
import logging
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def classify_pattern(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classifies detected anomalies into bullish or bearish patterns.

    :param df: DataFrame containing 'Reconstruction_Error', 'Anomaly', and 'Timestamp' columns.
    :return: DataFrame with an additional 'Pattern' column.
    """
    patterns = []
    for i in range(len(df)):
        if df.iloc[i]['Anomaly']:
            pattern = identify_bullish_patterns(df, i)
            if not pattern:
                pattern = identify_bearish_patterns(df, i)
            if not pattern:
                pattern = 'Unknown'
            patterns.append(pattern)
        else:
            patterns.append('Normal')
    df['Pattern'] = patterns
    return df


def identify_bullish_patterns(df: pd.DataFrame, anomaly_idx: int) -> str:
    """
    Identifies bullish indicators within anomaly data.

    :param df: DataFrame containing 'Reconstruction_Error', 'Anomaly', and 'Timestamp' columns.
    :param anomaly_idx: The integer index of the anomaly in the DataFrame.
    :return: String indicating the bullish pattern, if any.
    """
    window = 30  # e.g., 30 data points before the anomaly
    if anomaly_idx < window:
        return ''

    window_data = df.iloc[anomaly_idx - window:anomaly_idx]

    # Detect if there's an uptrend in reconstruction error
    trend = window_data['Reconstruction_Error'].values
    slope = np.polyfit(range(len(trend)), trend, 1)[0]
    if slope > 0.0005:
        return 'Bullish Flag'

    return ''


def identify_bearish_patterns(df: pd.DataFrame, anomaly_idx: int) -> str:
    """
    Identifies bearish indicators within anomaly data.

    :param df: DataFrame containing 'Reconstruction_Error', 'Anomaly', and 'Timestamp' columns.
    :param anomaly_idx: The integer index of the anomaly in the DataFrame.
    :return: String indicating the bearish pattern, if any.
    """
    window = 30  # e.g., 30 data points before the anomaly
    if anomaly_idx < window:
        return ''

    window_data = df.iloc[anomaly_idx - window:anomaly_idx]

    # Detect if there's a downtrend in reconstruction error
    trend = window_data['Reconstruction_Error'].values
    slope = np.polyfit(range(len(trend)), trend, 1)[0]
    if slope < -0.0005:
        return 'Bearish Pennant'

    return ''


def visualize_anomalies(df: pd.DataFrame, symbol: str, output_path: str):
    """
    Visualizes anomalies and their classification.

    :param df: DataFrame containing 'Reconstruction_Error', 'Anomaly', 'Pattern', and 'Category' columns.
    :param symbol: Stock symbol for the title.
    :param output_path: Path to save the visualization.
    """
    plt.figure(figsize=(14, 7))
    
    # Plot Reconstruction Error
    sns.lineplot(x=df.index, y='Reconstruction_Error', data=df, label='Reconstruction Error')
    
    # Highlight anomalies
    anomalies = df[df['Anomaly'] == True]
    plt.scatter(anomalies.index, anomalies['Reconstruction_Error'], color='red', label='Anomalies')
    
    # Annotate patterns
    for idx, row in anomalies.iterrows():
        if row['Pattern'] not in ['Unknown', 'Normal']:
            plt.annotate(row['Pattern'], (idx, row['Reconstruction_Error']),
                         textcoords="offset points", xytext=(0,10), ha='center', color='purple')
    
    plt.title(f'Anomaly Detection and Pattern Classification for {symbol}')
    plt.xlabel('Timestamp')
    plt.ylabel('Reconstruction Error')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Visualization saved to {output_path}")


def categorize_trading_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a column categorizing the nature of the anomaly into bullish or bearish.

    :param df: DataFrame containing 'Pattern' column.
    :return: DataFrame with an additional 'Category' column.
    """
    categories = []
    for idx, row in df.iterrows():
        if row['Pattern'].startswith('Bullish'):
            categories.append('Bullish')
        elif row['Pattern'].startswith('Bearish'):
            categories.append('Bearish')
        else:
            categories.append('Normal')
    df['Category'] = categories
    return df
