import torch
import torch.fft
import torch.nn as nn
from torch import Tensor


class WeightedMSE(nn.Module):
    """Weighted Mean Squared Error loss module."""

    def __init__(self: "WeightedMSE") -> None:
        """Initialize the WeightedMSE loss module."""
        super(WeightedMSE, self).__init__()

    def forward(
        self: "WeightedMSE", y_pred: torch.Tensor, y_true: torch.Tensor, weights: torch.Tensor = None
    ) -> torch.Tensor:
        """Calculate weighted mean squared error.

        Args:
            y_pred: Predicted values with shape (batch, seq_len).
            y_true: True values with shape (batch, seq_len).
            weights: Optional weights for each batch with shape (batch,).

        Returns:
            Weighted MSE for each batch with shape (batch,).
        """
        mse = (y_pred - y_true) ** 2  # (batch, seq_len)
        mse = mse.mean(dim=1)  # (batch,)
        if weights is not None:
            weighted_mse = weights * mse
            weighted_mse_batch = weighted_mse.mean(dim=0)
        else:
            weighted_mse_batch = mse.mean(dim=0)
        return weighted_mse_batch


class EffectiveRegionWeightedMSELoss(nn.Module):
    """MSE loss weighted by effective regions where either prediction or target is non-zero."""

    def __init__(self: "EffectiveRegionWeightedMSELoss") -> None:
        """Initialize the EffectiveRegionWeightedMSELoss module."""
        super(EffectiveRegionWeightedMSELoss, self).__init__()

    def forward(
        self: "EffectiveRegionWeightedMSELoss", y_pred: torch.Tensor, y_true: torch.Tensor, weights: torch.Tensor = None
    ) -> torch.Tensor:
        """Calculate MSE weighted by effective regions.

        Args:
            y_pred: Predicted values with shape (batch, seq_len).
            y_true: True values with shape (batch, seq_len).
            weights: Optional weights for each batch with shape (batch,).

        Returns:
            Weighted MSE for each batch with shape (batch,).
        """
        mask = ((y_pred != 0) | (y_true != 0)).float()  # (batch, seq_len)
        diff = (y_pred - y_true) ** 2 * mask
        region_len = mask.sum(dim=1).clamp(min=1)
        mse = diff.sum(dim=1) / region_len  # (batch,)
        if weights is not None:
            weighted_mse = weights * mse
            weighted_mse_batch = weighted_mse.mean(dim=0)
        else:
            weighted_mse_batch = mse.mean(dim=0)
        return weighted_mse_batch


class NonZeroWeightedMSE(nn.Module):
    """MSE loss weighted by non-zero error regions."""

    def __init__(self: "NonZeroWeightedMSE") -> None:
        """Initialize the NonZeroWeightedMSE module."""
        super(NonZeroWeightedMSE, self).__init__()

    def forward(
        self: "NonZeroWeightedMSE", y_pred: torch.Tensor, y_true: torch.Tensor, weights: torch.Tensor = None
    ) -> torch.Tensor:
        """Calculate MSE weighted by non-zero error regions.

        Args:
            y_pred: Predicted values with shape (batch, seq_len).
            y_true: True values with shape (batch, seq_len).
            weights: Optional weights for each batch with shape (batch,).

        Returns:
            Weighted MSE for each batch with shape (batch,).
        """
        diff = (y_pred - y_true) ** 2  # (batch, seq_len)
        mask = (diff != 0).float()
        num_nonzero = mask.sum(dim=1).clamp(min=1)  # (batch,)
        mse = diff.sum(dim=1) / num_nonzero  # (batch,)
        if weights is not None:
            weighted_mse = weights * mse
            weighted_mse_batch = weighted_mse.mean(dim=0)
        else:
            weighted_mse_batch = mse.mean(dim=0)
        return weighted_mse_batch


class AmplitudeWeightedMSELoss(nn.Module):
    """MSE loss weighted by the amplitude of the target signal."""

    def __init__(self: "AmplitudeWeightedMSELoss", epsilon: float, a: float) -> None:
        """Initialize the AmplitudeWeightedMSELoss module.

        Args:
            epsilon: Small value to avoid division by zero.
            a: Exponent for amplitude weighting.
        """
        super(AmplitudeWeightedMSELoss, self).__init__()
        self.epsilon = epsilon
        self.a = a

    def forward(
        self: "AmplitudeWeightedMSELoss", y_pred: torch.Tensor, y_true: torch.Tensor, weights: torch.Tensor = None
    ) -> torch.Tensor:
        """Calculate amplitude-weighted MSE.

        Args:
            y_pred: Predicted values with shape (batch, seq_len).
            y_true: True values with shape (batch, seq_len).
            weights: Optional weights for each sample with shape (batch,).

        Returns:
            Scalar amplitude-weighted MSE loss.
        """
        # Compute the amplitude (max absolute value) for each sample
        amplitude = torch.max(torch.abs(y_true), dim=1, keepdim=True)[0] + self.epsilon  # (batch, 1)

        # Normalize the error by amplitude^a
        normalized_error = (y_pred - y_true) / amplitude**self.a  # (batch, seq_len)

        # Compute per-sample MSE
        mse = (normalized_error**2).mean(dim=1)  # (batch,)

        # Apply weights if provided
        if weights is not None:
            weighted_mse = weights * mse  # (batch,)
            loss = weighted_mse.mean()  # scalar
        else:
            loss = mse.mean()  # scalar

        return loss


###################AMSELoss####################


def _unitary_fft(x: Tensor, dim: int = -1) -> Tensor:
    """Unitary FFT (norm='ortho') along `dim`."""
    return torch.fft.fft(x, dim=dim, norm="ortho")


def _per_bin_psd_unitary(X: Tensor, N: int) -> Tensor:
    """Per-bin PSD so that sum_k PSD_k = average power per sample."""
    return (X.abs() ** 2) / float(N)


def _per_bin_cos_phase(X: Tensor, Y: Tensor, eps: float) -> Tensor:
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

    def __init__(self: "AMSELoss", fft_dim: int = -1, reduction: str = "mean", eps: float = 1e-12) -> None:
        """Initialize AMSELoss with FFT configuration."""
        super().__init__()
        self.fft_dim = fft_dim
        self.reduction = reduction
        self.eps = eps

    def forward(self: "AMSELoss", pred: Tensor, target: Tensor) -> Tensor:
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
        term_amp = (torch.sqrt(PSDx + self.eps) - torch.sqrt(PSDy + self.eps)) ** 2
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


# Optional helpers
def mse_time(x: Tensor, y: Tensor) -> Tensor:
    """Time-domain mean squared error."""
    e = x - y
    return torch.mean(torch.abs(e) ** 2, dim=-1)


def mse_freq_unitary_torch(a: Tensor, b: Tensor) -> Tensor:
    """Frequency-domain MSE using unitary FFT."""
    A = _unitary_fft(a, dim=-1)
    B = _unitary_fft(b, dim=-1)
    return ((A - B).abs() ** 2).sum(dim=-1) / a.size(-1)
