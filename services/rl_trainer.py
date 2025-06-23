import asyncio
import torch
from wavenet_model import WaveNet


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


async def rl_update(model: WaveNet, optimizer: torch.optim.Optimizer, predicted: torch.Tensor, actual: torch.Tensor, steps: int = 10):
    """Simple reinforcement-style update over incoming data."""
    model.train()
    criterion = torch.nn.MSELoss()
    for _ in range(steps):
        loss = criterion(predicted, actual)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        await asyncio.sleep(0)
    return loss.item()
