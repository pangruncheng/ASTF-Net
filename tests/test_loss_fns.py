import torch
from astfnet.models.loss_fns import (
    WeightedMSE,
    EffectiveRegionWeightedMSELoss,
    NonZeroWeightedMSE,
)


def test_weighted_mse():
    loss_fn = WeightedMSE()
    y_pred = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    y_true = torch.tensor([[1.0, 1.0], [2.0, 2.0]])
    weights = torch.tensor([1.0, 2.0])
    loss = loss_fn(y_pred, y_true, weights)
    expected = (1.0 * ((0.0 + 1.0) / 2) + 2.0 * ((1.0 + 4.0) / 2)) / 2
    assert torch.isclose(loss, torch.tensor(expected)), (
        f"Expected {expected}, got {loss.item()}"
    )


def test_effective_region_weighted_mse():
    loss_fn = EffectiveRegionWeightedMSELoss()
    y_pred = torch.tensor([[0.0, 2.0], [0.0, 0.0]])
    y_true = torch.tensor([[1.0, 0.0], [2.0, 0.0]])
    weights = torch.tensor([1.0, 1.0])
    # For first sample: nonzero mask = [1,1], mse = ((0-1)^2 + (2-0)^2)/2 = (1+4)/2=2.5
    # For second sample: nonzero mask = [1,0], mse = ((0-2)^2 + (0-0)^2)/2 = (4+0)/1=4.0
    expected = (1.0 * 2.5 + 1.0 * 4.0) / 2
    loss = loss_fn(y_pred, y_true, weights)
    assert torch.isclose(loss, torch.tensor(expected)), (
        f"Expected {expected}, got {loss.item()}"
    )


def test_nonzero_weighted_mse():
    loss_fn = NonZeroWeightedMSE()
    y_pred = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    y_true = torch.tensor([[0.0, 0.0], [1.0, 2.0]])
    weights = torch.tensor([2.0, 1.0])
    # For first sample: nonzero mask = [1,0], mse = ((1-0)^2 + (0-0)^2)/1 = 1.0
    # For second sample: nonzero mask = [1,1], mse = ((0-1)^2 + (0-2)^2)/2 = (1+4)/2=2.5
    expected = (2.0 * 1.0 + 1.0 * 2.5) / 2
    loss = loss_fn(y_pred, y_true, weights)
    assert torch.isclose(loss, torch.tensor(expected)), (
        f"Expected {expected}, got {loss.item()}"
    )
