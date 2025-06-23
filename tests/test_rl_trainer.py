import asyncio
import importlib

import pytest

torch_spec = importlib.util.find_spec("torch")
torch = importlib.import_module("torch") if torch_spec else None


@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_rl_update_changes_weights():
    from services.rl_trainer import rl_update
    from services.wavenet_model import WaveNet

    model = WaveNet(in_channels=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    predicted = torch.zeros(1, 8, 10)
    actual = torch.ones(1, 8, 10)

    orig = [p.clone() for p in model.parameters()]
    asyncio.run(rl_update(model, optimizer, predicted, actual, steps=1))
    changed = any(not torch.equal(p0, p1) for p0, p1 in zip(orig, model.parameters()))
    assert changed
