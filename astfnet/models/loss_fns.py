import logging
from typing import Optional

import torch
import torch.fft
import torch.nn as nn

logger = logging.getLogger(__name__)

class WeightedMSE(nn.Module):
    """Weighted Mean Squared Error loss module."""

    def __init__(self) -> None:
        """Initialize the WeightedMSE loss module."""
        super().__init__()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Calculate weighted mean squared error.

        Args:
            y_pred: Predicted values with shape (batch, seq_len).
            y_true: True values with shape (batch, seq_len).
            weights: Optional weights for each batch with shape (batch,).

        Returns:
            Weighted MSE for each batch with shape (batch,).
        """
        mse = torch.pow(y_pred - y_true, 2)  # (batch, seq_len)
        mse = mse.mean(dim=1)  # (batch,)
        if weights is not None:
            weighted_mse = weights * mse
            weighted_mse_batch = weighted_mse.mean(dim=0)
        else:
            weighted_mse_batch = mse.mean(dim=0)
        return weighted_mse_batch


###################AMSELoss####################
def _unitary_fft(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Unitary FFT (norm='ortho') along `dim`."""
    return torch.fft.fft(x, dim=dim, norm="ortho")


def _per_bin_psd_unitary(X: torch.Tensor, N: int) -> torch.Tensor:
    """Per-bin PSD so that sum_k PSD_k = average power per sample."""
    return torch.pow(X.abs(), 2) / float(N)


def _per_bin_cos_phase(X: torch.Tensor, Y: torch.Tensor, eps: float) -> torch.Tensor:
    """cos(Δφ_k) = Re{X Y*} / (|X||Y| + eps), elementwise over the FFT axis.

    Returns:
        Real tensor in [-1, 1].
    """
    num = torch.real(X * torch.conj(Y))
    den = X.abs() * Y.abs() + eps
    cphi = num / den
    return torch.clamp(cphi, -1.0, 1.0)


class AMSELoss(nn.Module):
    """AMSE loss under unitary DFT.

    AMSE = Σ_k [ (√PSD_x - √PSD_y)^2 + 2*max(PSD_x,PSD_y)*(1 - cosΔφ_k) ]

    Args:
        fft_dim: FFT transform axis (default: -1)
        reduction: 'mean' | 'sum' | 'none'
        eps: small constant for numerical stability
    """

    def __init__(self, fft_dim: int = -1, reduction: str = "mean", eps: float = 1e-12) -> None:
        """Initialize AMSELoss with FFT configuration."""
        super().__init__()
        self.fft_dim = fft_dim
        self.reduction = reduction
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute AMSE loss between predicted and target tensors."""
        if pred.shape != target.shape:
            raise ValueError("pred and target must have identical shapes")

        N = pred.size(self.fft_dim)
        # Unitary FFT
        X = _unitary_fft(pred, dim=self.fft_dim)
        Y = _unitary_fft(target, dim=self.fft_dim)

        # Per-bin PSDs
        PSDx = _per_bin_psd_unitary(X, N)
        PSDy = _per_bin_psd_unitary(Y, N)

        # cos(Δφ_k)
        cphi = _per_bin_cos_phase(X, Y, self.eps)

        # Amplitude term
        term_amp = torch.pow(torch.sqrt(PSDx + self.eps) - torch.sqrt(PSDy + self.eps), 2)
        # Phase term
        term_phase = 2.0 * torch.maximum(PSDx, PSDy) * (1.0 - cphi)

        amse_per_example = (term_amp + term_phase).sum(dim=self.fft_dim)

        # Reduction
        if self.reduction == "mean":
            return amse_per_example.mean()
        elif self.reduction == "sum":
            return amse_per_example.sum()
        else:
            return amse_per_example


def load_loss(
    config: dict,
) -> torch.nn.Module:  # pragma: no cover
    """Instantiate the loss.

    Args:
        config: config dict from the config file

    Returns:
        loss_fn: The loss for the given model
    """
    loss_name = config["loss"]
    if loss_name == "mse":
        logger.info("MSELoss is loaded as the loss function.")
        loss_fn = torch.nn.MSELoss()
    elif loss_name == "amse":
        logger.info("AMSELoss (Amplitude and Phase Spectral Loss) is loaded as the loss function.")
        # Read parameters from config file (provide default values)
        fft_dim = config.get("loss_fft_dim", -1)
        reduction = config.get("loss_reduction", "mean")
        eps = config.get("loss_eps", 1e-6)

        loss_fn = AMSELoss(
            fft_dim=fft_dim,
            reduction=reduction,
            eps=eps,
        )
    else:
        raise ValueError("loss {} not supported".format(loss_name))
    return loss_fn
