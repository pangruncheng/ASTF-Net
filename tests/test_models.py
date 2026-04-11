from typing import Any, Dict

import pytest
import torch

from astfnet.models.base import ASTFModule


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
    model = ASTFModule(config)
    out = model(dummy_batch["target"], dummy_batch["egf"])
    assert out.shape == (2, 501)


def test_simplecnn_training_step(dummy_batch: Dict[str, Any]) -> None:
    config = {"in_channels": 2, "output_length": 501, "model_name": "simplecnn", "loss": "mse"}
    model = ASTFModule(config)
    loss = model.training_step(dummy_batch, 0)
    assert loss.requires_grad
    assert loss.item() >= 0


def test_simplecnn_validation_step(dummy_batch: Dict[str, Any]) -> None:
    config = {"in_channels": 2, "output_length": 501, "model_name": "simplecnn", "loss": "mse"}
    model = ASTFModule(config)
    model.on_validation_start()
    loss = model.validation_step(dummy_batch, 0)
    assert loss.item() >= 0
