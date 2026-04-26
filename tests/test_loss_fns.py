import torch

from astfnet.models.loss_fns import AMSELoss, WeightedMSE


def test_weighted_mse() -> None:
    loss_fn = WeightedMSE()
    y_pred = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    y_true = torch.tensor([[1.0, 1.0], [2.0, 2.0]])
    weights = torch.tensor([1.0, 2.0])
    loss = loss_fn(y_pred, y_true, weights)
    expected = (1.0 * ((0.0 + 1.0) / 2) + 2.0 * ((1.0 + 4.0) / 2)) / 2
    assert torch.isclose(loss, torch.tensor(expected)), f"Expected {expected}, got {loss.item()}"

def test_amse_loss() -> None:
    """
    Unit test for AMSELoss (Amplitude + Phase Spectral Loss).

    The AMSE formula:
        AMSE = Σ_k [ (√PSD_x - √PSD_y)^2 + 2 * max(PSD_x, PSD_y) * (1 - cosΔφ_k) ]
    """

    # Case 1: Identical signals → expect near-zero loss
    x_true = torch.tensor([[1.0, 0.0, -1.0, 0.0]])
    x_pred = torch.tensor([[1.0, 0.0, -1.0, 0.0]])
    loss_fn = AMSELoss(fft_dim=-1, reduction="mean", eps=1e-12)
    loss_same = loss_fn(x_pred, x_true)
    assert torch.allclose(
        loss_same, torch.tensor(0.0), atol=1e-6
    ), f"Expected 0 for identical signals, got {loss_same.item()}"

    # Case 2: Amplitude mismatch → expect positive loss
    x_pred_amp = 2 * x_true
    loss_amp = loss_fn(x_pred_amp, x_true)
    assert loss_amp > 0, "Expected positive AMSE for amplitude mismatch"

    # Case 3: Phase mismatch → expect positive loss
    x_pred_phase = torch.roll(x_true, shifts=1, dims=-1)
    loss_phase = loss_fn(x_pred_phase, x_true)
    assert loss_phase > 0, "Expected positive AMSE for phase shift"

    # Case 4: Numerical stability → no NaN or Inf
    x_zero = torch.zeros_like(x_true)
    loss_stable = loss_fn(x_zero, x_true)
    assert not torch.isnan(loss_stable), "Loss returned NaN for zero input"
    assert not torch.isinf(loss_stable), "Loss returned Inf for zero input"
