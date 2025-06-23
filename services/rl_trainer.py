"""Reinforcement-style weight update helpers."""

from __future__ import annotations

import asyncio

import torch
from wavenet_model import WaveNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


async def rl_update(
    model: WaveNet,
    optimizer: torch.optim.Optimizer,
    predicted: torch.Tensor,
    actual: torch.Tensor,
    steps: int = 10,
) -> float:
    """Perform a simplistic RL-style update loop.

    Args:
        model: The network to update.
        optimizer: Optimizer instance.
        predicted: Predicted tensor.
        actual: Ground-truth tensor.
        steps: Number of update iterations.

    Returns:
        Final loss value.
    """

    model.train()
    criterion = torch.nn.MSELoss()
    loss = torch.tensor(0.0)
    for _ in range(steps):
        loss = criterion(predicted, actual)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        await asyncio.sleep(0)
    return float(loss.item())
