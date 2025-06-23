# ./services/data_loader.py

"""Data preprocessing utilities."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from .technical_indicators import add_indicators


# Function to preprocess stock data for anomaly detection model
def preprocess_data(
    data: pd.DataFrame, sequence_length: int
) -> Tuple[np.ndarray, MinMaxScaler]:
    """Return normalized sliding windows with technical indicators.

    Args:
        data: Raw OHLCV dataframe.
        sequence_length: Number of timesteps per sequence.

    Returns:
        Tuple of ``np.ndarray`` of shape ``(n, sequence_length, 8)`` and the scaler used.
    """

    features = add_indicators(data)
    feature_cols = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "MA20",
        "MA50",
        "RSI",
    ]
    cleaned = features[feature_cols].dropna()
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(cleaned)

    sequences = [
        scaled[i : i + sequence_length] for i in range(len(scaled) - sequence_length)
    ]
    return np.array(sequences), scaler


# Function to scale data (for inference or further preprocessing)
def scale_data(data: pd.DataFrame) -> MinMaxScaler:
    """
    Scales the stock data using MinMaxScaler.

    :param data: DataFrame containing raw stock data.
    :return: MinMaxScaler object used to scale the data.

    :usage:
           from app/services/preprocess.py import scale_data
           scaler = scale_data(data)
    """
    enriched = add_indicators(data)
    feature_cols = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "MA20",
        "MA50",
        "RSI",
    ]
    cleaned = enriched[feature_cols].dropna()
    scaler = MinMaxScaler()
    scaler.fit(cleaned)
    return scaler


# Function to inverse scale data (after prediction for interpretation)
def inverse_scale(scaled_data: np.ndarray, scaler: MinMaxScaler) -> np.ndarray:
    """Inverse transform ``scaled_data`` back to the original scale."""

    return scaler.inverse_transform(scaled_data)
