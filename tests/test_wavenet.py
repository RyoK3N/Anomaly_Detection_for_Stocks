import importlib

import pytest

torch_spec = importlib.util.find_spec("torch")
torch = importlib.import_module("torch") if torch_spec else None


@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_wavenet_forward_shape():
    from services.wavenet_model import WaveNet

    model = WaveNet(in_channels=8)
    x = torch.randn(2, 8, 60)
    out = model(x)
    assert out.shape == x.shape
