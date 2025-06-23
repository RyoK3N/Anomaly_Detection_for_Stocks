"""WaveNet architecture and helpers."""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class WaveNetBlock(nn.Module):
    """Single dilated convolution block."""

    def __init__(self, channels: int, dilation: int, skip_channels: int) -> None:
        super().__init__()
        padding = dilation
        self.filter_conv = nn.Conv1d(
            channels, channels, kernel_size=2, dilation=dilation, padding=padding
        )
        self.gate_conv = nn.Conv1d(
            channels, channels, kernel_size=2, dilation=dilation, padding=padding
        )
        self.res_conv = nn.Conv1d(channels, channels, kernel_size=1)
        self.skip_conv = nn.Conv1d(channels, skip_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        filter_out = torch.tanh(self.filter_conv(x))
        gate_out = torch.sigmoid(self.gate_conv(x))
        out = filter_out * gate_out
        skip = self.skip_conv(out)
        residual = self.res_conv(out)
        return x + residual, skip


class WaveNet(nn.Module):
    """Minimal WaveNet implementation."""

    def __init__(
        self,
        in_channels: int,
        residual_channels: int = 32,
        skip_channels: int = 64,
        dilations: list[int] | None = None,
    ) -> None:
        super().__init__()
        if dilations is None:
            dilations = [1, 2, 4, 8, 16]

        self.device = device
        self.initial = nn.Conv1d(in_channels, residual_channels, kernel_size=1)
        self.blocks = nn.ModuleList(
            [WaveNetBlock(residual_channels, d, skip_channels) for d in dilations]
        )
        self.final = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(skip_channels, skip_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(skip_channels, in_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.initial(x)
        skips = []
        for block in self.blocks:
            x, skip = block(x)
            skips.append(skip)
        out = sum(skips)
        return self.final(out)


def train_wavenet(
    model: WaveNet,
    loader: DataLoader,
    epochs: int = 20,
    lr: float = 1e-3,
    checkpoint: str = "./services/checkpoints/wavenet.pth",
) -> WaveNet:
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    best_loss = float("inf")
    model.to(device)

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for batch in tqdm(loader, desc=f"Epoch {epoch}/{epochs}"):
            batch = batch[0].to(device).float()
            optimizer.zero_grad()
            recon = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            running += loss.item()
        avg = running / len(loader)
        if avg < best_loss:
            best_loss = avg
            torch.save(model.state_dict(), checkpoint)
    return model


def load_wavenet(
    input_dim: int, checkpoint: str = "./services/checkpoints/wavenet.pth"
) -> WaveNet:
    """Instantiate ``WaveNet`` and load weights if present."""

    model = WaveNet(in_channels=input_dim)
    if os.path.exists(checkpoint):
        model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.to(device)
    model.device = device
    return model
