# ./services/data_loader.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple

# Function to preprocess stock data for anomaly detection model
def preprocess_data(data: pd.DataFrame, sequence_length: int) -> Tuple[np.ndarray, MinMaxScaler]:
    """
    Cleans, scales, and creates input sequences from raw stock data.
    
    :param data: DataFrame containing raw stock data (e.g., 'Open', 'High', 'Low', 'Close', 'Volume').
    :param sequence_length: Number of time steps in each input sequence.
    :return: Tuple containing numpy array of sequences and the MinMaxScaler object.
    
    :usage:
           from app/services/preprocess.py import preprocess_data
           sequences, scaler = preprocess_data(data, 60)
    """

    data_cleaned = data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data_cleaned)
    sequences = []
    for i in range(len(scaled_data) - sequence_length):
        sequences.append(scaled_data[i:i + sequence_length])

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
    data_cleaned = data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    scaler = MinMaxScaler()
    scaler.fit(data_cleaned)

    return scaler


# Function to inverse scale data (after prediction for interpretation)
def inverse_scale(scaled_data: np.ndarray, scaler: MinMaxScaler) -> np.ndarray:
    """
    Reverses the scaling applied to the stock data, to retrieve original values.
    
    :param scaled_data: Scaled data to be inversely transformed.
    :param scaler: MinMaxScaler object that was used for scaling.
    :return: Data in original scale.
    
    :usage:
           from app/services/preprocess.py import inverse_scale
           original_data = inverse_scale(scaled_data, scaler)
    """
    return scaler.inverse_transform(scaled_data)
