from typing import Any, Dict

import pytest
import torch

from astfnet.models.cnn import PLCNN


@pytest.fixture
def dummy_batch() -> Dict[str, Any]:
    batch_size = 2
    seq_len = 256
    output_length = 501
    return {
        "target": torch.randn(batch_size, seq_len),
        "egf": torch.randn(batch_size, seq_len),
        "astf": torch.abs(torch.randn(batch_size, output_length)),
    }


def test_simplecnn_forward(dummy_batch: Dict[str, Any]) -> None:
    config = {"in_channels": 2, "output_length": 501, "model_name": "simplecnn", "loss": "mse"}
    model = PLCNN(config)
    out = model(dummy_batch["target"], dummy_batch["egf"])
    assert out.shape == (2, 501)


def test_simplecnn_training_step(
    dummy_batch: Dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = {"in_channels": 2, "output_length": 501, "model_name": "simplecnn", "loss": "mse"}
    model = PLCNN(config)

    monkeypatch.setattr(model, "print", lambda *a, **kw: None)

    monkeypatch.setattr(model, "safe_log", lambda *a, **kw: None)

    class DummyOptimizer:
        def __init__(self) -> None:
            self.param_groups = [{"lr": 1e-3}]

    class DummyTrainer:
        def __init__(self) -> None:
            self.optimizers = [DummyOptimizer()]
            self.global_step = 0

    model._trainer = DummyTrainer()

    loss = model.training_step(dummy_batch, 0)
    assert loss.requires_grad
    assert loss.item() >= 0


def test_simplecnn_validation_step(dummy_batch: Dict[str, Any]) -> None:
    config = {"in_channels": 2, "output_length": 501, "model_name": "simplecnn", "loss": "mse"}
    model = PLCNN(config)
    loss = model.validation_step(dummy_batch, 0)
    assert loss.item() >= 0
