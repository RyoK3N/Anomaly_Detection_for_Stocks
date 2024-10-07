# ./services/evaluate.py

import os
import argparse
import logging
import torch
import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.preprocessing import MinMaxScaler
import joblib
from tqdm import tqdm

# Import the VQVAE model and prepare_model function
from model import VQVAE, prepare_model
from preprocess import preprocess_data
from data_loader import load_real_time_data

# Import pattern recognition functions
from pattern_recognition import classify_pattern, visualize_anomalies, categorize_trading_patterns

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_mse(original: np.ndarray, reconstructed: np.ndarray) -> np.ndarray:
    """
    Calculates the Mean Squared Error between original and reconstructed sequences.

    :param original: Original data sequences, shape [num_samples, features, sequence_length]
    :param reconstructed: Reconstructed data sequences, same shape as original
    :return: Array of MSE values for each sample
    """
    mse = np.mean((original - reconstructed) ** 2, axis=(1, 2))
    return mse


def detect_anomalies(
    model: VQVAE,
    data: torch.Tensor,
    device: torch.device,
    threshold: float,
    batch_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detects anomalies in the data using the trained VQ-VAE model.

    :param model: Trained VQ-VAE model
    :param data: Preprocessed data tensor, shape [num_samples, features, sequence_length]
    :param device: Torch device (CPU or CUDA)
    :param threshold: Threshold for anomaly detection
    :param batch_size: Batch size for processing data
    :return: Tuple of (anomalies array, reconstruction errors array)
    """
    model.eval()
    anomalies = []
    recon_errors = []

    with torch.no_grad():
        for i in tqdm(range(0, data.size(0), batch_size), desc="Detecting Anomalies"):
            batch = data[i:i + batch_size].to(device)
            recon, _ = model(batch)
        
            recon = recon.cpu().numpy()
            batch = batch.cpu().numpy()
            # Calculate MSE for each sample in the batch
            mse = calculate_mse(batch, recon)
            recon_errors.extend(mse)
            # Determine anomalies
            batch_anomalies = mse > threshold
            anomalies.extend(batch_anomalies)

    return np.array(anomalies), np.array(recon_errors)


def load_new_data(symbol: str, start: str, end: str) -> pd.DataFrame:
    """
    Loads new data using the load_real_time_data function.

    :param symbol: Stock symbol to fetch data for.
    :param start: Start date in 'YYYY-MM-DD' format.
    :param end: End date in 'YYYY-MM-DD' format.
    :return: DataFrame containing the new data
    """
    logger.info(f"Loading new data for symbol: {symbol}, from {start} to {end}.")
    df = load_real_time_data(symbol, start, end)
    if df.empty:
        logger.error("No data fetched. Please check the symbol and date range.")
        raise ValueError("No data fetched. Please check the symbol and date range.")
    logger.info(f"Loaded new data with shape: {df.shape}")
    return df


def save_results(
    output_path: str,
    df_original: pd.DataFrame,
    recon_errors: np.ndarray,
    anomalies: np.ndarray,
    sequence_length: int
):
    """
    Saves the reconstruction errors and anomaly flags to a CSV file, along with the original timestamps.

    :param output_path: Path to save the output CSV
    :param df_original: Original DataFrame containing timestamps
    :param recon_errors: Array of reconstruction error values
    :param anomalies: Array of boolean anomaly flags
    :param sequence_length: The window size used for sequence generation
    """
    # Align reconstruction errors and anomalies with the corresponding timestamps
    if len(df_original) < sequence_length:
        logger.error("Original data length is less than sequence_length.")
        raise ValueError("Original data length is less than sequence_length.")

    timestamps = df_original.index[sequence_length:].tolist()  # sequence_length = window size
    results = pd.DataFrame({
        'Timestamp': timestamps,
        'Reconstruction_Error': recon_errors,
        'Anomaly': anomalies
    })
    results.set_index('Timestamp', inplace=True)

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results.to_csv(output_path)
    logger.info(f"Saved evaluation results to {output_path}")


def main(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load the trained model and scaler
    logger.info("Loading the trained model and scaler.")
    model, scaler = prepare_model(
        checkpoint_dir=args.checkpoint_dir,
        symbol=args.symbol,
        data_start_date=args.train_start,
        data_end_date=args.train_end
    )
    model.to(device)

    # Load new data
    logger.info("Loading new data for evaluation.")
    new_data_df = load_new_data(args.symbol, args.start, args.end)

    # Preprocess the data
    logger.info("Preprocessing the new data.")
    sequences, _ = preprocess_data(new_data_df, sequence_length=args.sequence_length)
    logger.info(f"Data preprocessed into {sequences.shape} sequences.")

    # Convert sequences to tensor
    sequences_tensor = torch.tensor(sequences, dtype=torch.float32)
    # Check current shape
    logger.info(f"Sequences tensor shape before permutation: {sequences_tensor.shape}")
    # Permute to [batch, features, sequence_length]
    sequences_tensor = sequences_tensor.permute(0, 2, 1)
    logger.info(f"Sequences tensor shape after permutation: {sequences_tensor.shape}")

    # Detect anomalies
    logger.info("Starting anomaly detection on new data.")
    anomalies, recon_errors = detect_anomalies(
        model,
        sequences_tensor,
        device,
        args.threshold,
        args.batch_size
    )
    logger.info("Anomaly detection on new data completed.")

    # Save results with timestamps
    logger.info("Saving the results.")
    save_results(
        args.output_path,
        new_data_df,
        recon_errors,
        anomalies,
        sequence_length=args.sequence_length
    )

    # Load the saved results
    results_df = pd.read_csv(args.output_path, parse_dates=['Timestamp'], index_col='Timestamp')

    # Classify patterns
    logger.info("Classifying detected anomalies into trading patterns.")
    results_df = classify_pattern(results_df)

    # Categorize patterns
    results_df = categorize_trading_patterns(results_df)

    # Save updated results with patterns
    results_with_patterns_path = os.path.splitext(args.output_path)[0] + '_with_patterns.csv'
    results_df.to_csv(results_with_patterns_path)
    logger.info(f"Saved evaluation results with patterns to {results_with_patterns_path}")

    # Visualize anomalies and patterns
    visualization_path = os.path.splitext(args.output_path)[0] + '_visualization.png'
    visualize_anomalies(results_df, args.symbol, visualization_path)

    logger.info("Evaluation process finished successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate VQ-VAE model for anomaly detection on historical stock data."
    )

    # Required arguments
    parser.add_argument('--symbol', type=str, required=True, help='Stock symbol to evaluate (e.g., AAPL)')
    parser.add_argument('--start', type=str, required=True, help='Start date for evaluation data in YYYY-MM-DD format')
    parser.add_argument('--end', type=str, required=True, help='End date for evaluation data in YYYY-MM-DD format')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the evaluation results CSV')
    parser.add_argument('--train_start', type=str, default='2014-01-01', help='Start date for training data used in scaler fitting')
    parser.add_argument('--train_end', type=str, default='2024-01-01', help='End date for training data used in scaler fitting')
    # Kwargs arguments
    parser.add_argument('--checkpoint_dir', type=str, default='./services/checkpoints', help='Directory containing model checkpoints and scaler')
    parser.add_argument('--threshold', type=float, default=0.05, help='Threshold for anomaly detection based on MSE')
    parser.add_argument('--sequence_length', type=int, default=60, help='Number of time steps in each input sequence')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for processing data')

    args = parser.parse_args()

    main(args)
