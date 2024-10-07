# ./services/model.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple
import yfinance as yf
from data_loader import load_real_time_data
from preprocess import preprocess_data
from torch.utils.data import DataLoader, TensorDataset, random_split
import joblib
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float):
        """
        Initializes the Vector Quantizer.
        
        :param num_embeddings: Number of embeddings in the codebook.
        :param embedding_dim: Dimension of each embedding vector.
        :param commitment_cost: Weight for the commitment loss.
        """
        super(VectorQuantizer, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        # Initialize embeddings with uniform distribution
        self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embeddings.weight.data.uniform_(-1 / self.num_embeddings, 1 / self.num_embeddings)

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the Vector Quantizer.
        
        :param inputs: Encoder outputs of shape [batch, embedding_dim, sequence_length]
        :return: Quantized tensor and the quantization loss.
        """
        # Reshape input to [batch*sequence_length, embedding_dim]
        batch_size, embedding_dim, sequence_length = inputs.shape
        flat_inputs = inputs.permute(0, 2, 1).contiguous().view(-1, self.embedding_dim)

        # Calculate distances between encoder outputs and embeddings
        distances = (
            torch.sum(flat_inputs**2, dim=1, keepdim=True) +
            torch.sum(self.embeddings.weight**2, dim=1) -
            2 * torch.matmul(flat_inputs, self.embeddings.weight.t())
        )

        # Encoding indices for minimum distance
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.size(0), self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and reshape back to [batch, embedding_dim, sequence_length]
        quantized = torch.matmul(encodings, self.embeddings.weight).view(batch_size, sequence_length, self.embedding_dim)
        quantized = quantized.permute(0, 2, 1).contiguous()

        # Calculate losses
        e_latent_loss = torch.mean((quantized.detach() - inputs) ** 2)
        q_latent_loss = torch.mean((quantized - inputs.detach()) ** 2)
        loss = self.commitment_cost * e_latent_loss + q_latent_loss

        # Apply Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()

        return quantized, loss


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        """
        Initializes a Residual Block.
        
        :param channels: Number of input and output channels.
        """
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Residual Block.
        
        :param x: Input tensor of shape [batch, channels, sequence_length]
        :return: Output tensor of the same shape.
        """
        return x + self.block(x)


class VQVAE(nn.Module):
    def __init__(self, input_dim: int, embedding_dim: int = 64, num_embeddings: int = 512, commitment_cost: float = 0.25):
        """
        Initializes the VQ-VAE model with a residual encoder and decoder.
        
        :param input_dim: Number of input features (e.g., 5 for ['Open', 'High', 'Low', 'Close', 'Volume']).
        :param embedding_dim: Dimension of the embedding vectors.
        :param num_embeddings: Number of embeddings in the codebook.
        :param commitment_cost: Weight for the commitment loss.
        """
        super(VQVAE, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings

        # Encoder with Residual Blocks
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            ResidualBlock(64),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            ResidualBlock(128),
            nn.Conv1d(in_channels=128, out_channels=embedding_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True)
        )

        # Vector Quantizer
        self.quantizer = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)

        # Decoder with Residual Blocks
        self.decoder = nn.Sequential(
            ResidualBlock(embedding_dim),
            nn.ConvTranspose1d(in_channels=embedding_dim, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            ResidualBlock(128),
            nn.ConvTranspose1d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            ResidualBlock(64),
            nn.Conv1d(in_channels=64, out_channels=input_dim, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()  # Assuming input data is normalized between 0 and 1
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the VQ-VAE model.
        
        :param x: Input tensor of shape [batch, input_dim, sequence_length]
        :return: Reconstructed tensor and the quantization loss.
        """
        z = self.encoder(x)
        quantized, vq_loss = self.quantizer(z)
        recon = self.decoder(quantized)
        return recon, vq_loss


def train_vqvae(
    model: VQVAE,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 50,
    lr: float = 1e-3,
    checkpoint_dir: str = './services/checkpoints'
) -> VQVAE:
    """
    Train the VQ-VAE model on given data and save the best checkpoint.
    
    :param model: VQ-VAE model.
    :param train_loader: DataLoader for training data.
    :param val_loader: DataLoader for validation data.
    :param epochs: Number of training epochs.
    :param lr: Learning rate.
    :param checkpoint_dir: Directory to save model checkpoints.
    :return: Trained VQ-VAE model.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, 'vqvae_best.pth')

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)

    best_val_loss = float('inf')
    patience_counter = 0
    patience = 10  # Early stopping patience

    logger.info("Starting training of VQ-VAE model.")
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        vq_loss_total = 0.0
        recon_loss_total = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} - Training"):
            batch = batch[0].to(device).float()  # [batch, features, sequence_length]

            # Assert correct shape
            if batch.shape[1] != 5 or batch.shape[2] != 60:
                logger.error(f"Expected input shape [batch, 5, 60], but got {batch.shape}")
                raise AssertionError(f"Expected input shape [batch, 5, 60], but got {batch.shape}")

            optimizer.zero_grad()
            recon, vq_loss = model(batch)
            recon_loss = criterion(recon, batch)
            loss = recon_loss + vq_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()

            running_loss += loss.item()
            vq_loss_total += vq_loss.item()
            recon_loss_total += recon_loss.item()

        avg_train_loss = running_loss / len(train_loader)
        avg_vq_loss = vq_loss_total / len(train_loader)
        avg_recon_loss = recon_loss_total / len(train_loader)

        # Validation
        model.eval()
        val_running_loss = 0.0
        val_vq_loss_total = 0.0
        val_recon_loss_total = 0.0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} - Validation"):
                batch = batch[0].to(device).float()

                # Assert correct shape
                if batch.shape[1] != 5 or batch.shape[2] != 60:
                    logger.error(f"Expected input shape [batch, 5, 60], but got {batch.shape}")
                    raise AssertionError(f"Expected input shape [batch, 5, 60], but got {batch.shape}")

                recon, vq_loss = model(batch)
                recon_loss = criterion(recon, batch)
                loss = recon_loss + vq_loss

                val_running_loss += loss.item()
                val_vq_loss_total += vq_loss.item()
                val_recon_loss_total += recon_loss.item()

        avg_val_loss = val_running_loss / len(val_loader)
        avg_val_vq_loss = val_vq_loss_total / len(val_loader)
        avg_val_recon_loss = val_recon_loss_total / len(val_loader)

        logger.info(f"Epoch [{epoch}/{epochs}], Train Loss: {avg_train_loss:.4f}, Recon Loss: {avg_recon_loss:.4f}, VQ Loss: {avg_vq_loss:.4f}")
        logger.info(f"Epoch [{epoch}/{epochs}], Val Loss: {avg_val_loss:.4f}, Recon Loss: {avg_val_recon_loss:.4f}, VQ Loss: {avg_val_vq_loss:.4f}")

        # Scheduler step
        scheduler.step(avg_val_loss)

        # Early Stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_val_loss,
            }, checkpoint_path)
            logger.info(f"Best model saved at epoch {epoch} with validation loss {avg_val_loss:.4f}")
        else:
            patience_counter += 1
            logger.info(f"No improvement in validation loss for {patience_counter} epoch(s).")
            if patience_counter >= patience:
                logger.info("Early stopping triggered.")
                break

    logger.info(f"Training completed. Best model saved at {checkpoint_path}")
    return model


def detect_anomalies(
    model: VQVAE,
    data: torch.Tensor,
    threshold: float = 0.05,
    batch_size: int = 32
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect anomalies in data using reconstruction error.
    
    :param model: Trained VQ-VAE model.
    :param data: Data to check for anomalies (Tensor of shape [batch_size, feature_dim, sequence_length]).
    :param threshold: Threshold for anomaly detection based on reconstruction error.
    :param batch_size: Batch size for processing data.
    :return: Tuple containing a boolean array indicating anomalies and the reconstruction errors.
    """
    model.eval()
    anomalies = []
    recon_errors = []

    with torch.no_grad():
        for i in tqdm(range(0, data.size(0), batch_size), desc="Detecting Anomalies"):
            batch = data[i:i+batch_size].to(device)
            recon, _ = model(batch)
            # Move tensors to CPU and convert to numpy
            recon = recon.cpu().numpy()
            batch = batch.cpu().numpy()
            # Calculate MSE for each sample in the batch
            mse = np.mean((batch - recon) ** 2, axis=(1, 2))
            recon_errors.extend(mse)
            # Determine anomalies
            batch_anomalies = mse > threshold
            anomalies.extend(batch_anomalies)

    return np.array(anomalies), np.array(recon_errors)


def get_stock_data(symbol: str, start: str, end: str) -> pd.DataFrame:
    """
    Fetch stock data from Yahoo Finance.
    
    :param symbol: Stock symbol.
    :param start: Start date in 'YYYY-MM-DD' format.
    :param end: End date in 'YYYY-MM-DD' format.
    :return: DataFrame containing historical stock data.
    """
    logger.info(f"Fetching data for {symbol} from {start} to {end}.")
    data = load_real_time_data(symbol, start, end)
    if data.empty:
        logger.error("No data fetched. Please check the symbol and date range.")
    return data


def prepare_model(
    checkpoint_dir: str = './services/checkpoints',
    symbol: str = 'AAPL',
    data_start_date: str = '2014-01-01',
    data_end_date: str = '2024-01-01'
) -> Tuple[VQVAE, MinMaxScaler]:
    """
    Prepare the VQ-VAE model by loading existing weights or training a new model.
    
    :param checkpoint_dir: Directory where checkpoints are stored.
    :param symbol: Stock symbol for training or scaler fitting.
    :param data_start_date: Start date for training data.
    :param data_end_date: End date for training data.
    :return: Tuple containing the model and the scaler used for data preprocessing.
    """
    checkpoint_path = os.path.join(checkpoint_dir, 'vqvae_best.pth')
    input_shape = (5, 60)  # feature_dim=5, sequence_length=60

    if os.path.exists(checkpoint_path):
        logger.info("Pre-trained model weights found. Loading the model.")
        model = VQVAE(
            input_dim=input_shape[0],
            embedding_dim=64,
            num_embeddings=512,
            commitment_cost=0.25
        ).to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded VQ-VAE model from {checkpoint_path}")

        # Load scaler
        scaler_path = os.path.join(checkpoint_dir, 'scaler.save')
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            logger.info("Loaded scaler for data preprocessing.")
        else:
            logger.warning("Scaler not found. Please ensure scaler is saved during training.")
            # Refit scaler if not found
            data = get_stock_data(symbol=symbol, start=data_start_date, end=data_end_date)
            if data.empty:
                raise ValueError("Failed to fetch stock data for scaler fitting.")
            _, scaler = preprocess_data(data, sequence_length=60)
    else:
        logger.info("No pre-trained weights found. Training the VQ-VAE model.")
        # Fetch stock data
        data = get_stock_data(symbol=symbol, start=data_start_date, end=data_end_date)
        if data.empty:
            raise ValueError("Failed to fetch stock data for training.")

        # Preprocess data
        sequences, scaler = preprocess_data(data, sequence_length=60)
        logger.info(f"Data preprocessed into {sequences.shape} sequences.")

        # Save scaler
        os.makedirs(checkpoint_dir, exist_ok=True)
        scaler_path = os.path.join(checkpoint_dir, 'scaler.save')
        joblib.dump(scaler, scaler_path)
        logger.info(f"Scaler saved at {scaler_path}")

        # Convert sequences to tensors
        sequences_tensor = torch.tensor(sequences, dtype=torch.float32)
        # Transpose to [batch, feature_dim, sequence_length]
        sequences_tensor = sequences_tensor.permute(0, 2, 1)
        logger.info(f"Transposed sequences_tensor shape: {sequences_tensor.shape}")
        train_dataset = TensorDataset(sequences_tensor)

        # Split into training and validation sets
        val_size = int(0.1 * len(train_dataset))
        train_size = len(train_dataset) - val_size
        train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

        train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)

        # Initialize model
        model = VQVAE(
            input_dim=input_shape[0],
            embedding_dim=64,
            num_embeddings=512,
            commitment_cost=0.25
        ).to(device)
        logger.info("VQ-VAE model built successfully.")

        # Train model
        model = train_vqvae(
            model,
            train_loader,
            val_loader,
            epochs=50,
            lr=1e-3,
            checkpoint_dir=checkpoint_dir
        )

    return model, scaler
