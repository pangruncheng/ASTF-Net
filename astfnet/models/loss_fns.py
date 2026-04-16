import logging

import torch
import torch.fft
import torch.nn as nn

logger = logging.getLogger(__name__)


class WeightedMSE(nn.Module):
    """Weighted Mean Squared Error loss module."""

    def __init__(self) -> None:
        """Initialize the WeightedMSE loss module."""
        super().__init__()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, weights: torch.Tensor = None) -> torch.Tensor:
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

    def __init__(self) -> None:
        """Initialize the EffectiveRegionWeightedMSELoss module."""
        super().__init__()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, weights: torch.Tensor = None) -> torch.Tensor:
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

    def __init__(self) -> None:
        """Initialize the NonZeroWeightedMSE module."""
        super().__init__()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, weights: torch.Tensor = None) -> torch.Tensor:
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

    def __init__(self, epsilon: float, a: float) -> None:
        """Initialize the AmplitudeWeightedMSELoss module.

        Args:
            epsilon: Small value to avoid division by zero.
            a: Exponent for amplitude weighting.
        """
        super().__init__()
        self.epsilon = epsilon
        self.a = a

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, weights: torch.Tensor = None) -> torch.Tensor:
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
def _unitary_fft(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Unitary FFT (norm='ortho') along `dim`."""
    return torch.fft.fft(x, dim=dim, norm="ortho")


def _per_bin_psd_unitary(X: torch.Tensor, N: int) -> torch.Tensor:
    """Per-bin PSD so that sum_k PSD_k = average power per sample."""
    return (X.abs() ** 2) / float(N)


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
    if loss_name == "weighted_mse":
        logger.info("WeightedMSE is loaded as the loss function.")
        loss_fn = WeightedMSE()
    elif loss_name == "effective_region_weighted_mse":
        logger.info("EffectiveRegionWeightedMSELoss is loaded as the loss function.")
        loss_fn = EffectiveRegionWeightedMSELoss()
    elif loss_name == "nonzero_weighted_mse":
        logger.info("NonZeroWeightedMSE is loaded as the loss function.")
        loss_fn = NonZeroWeightedMSE()
    elif loss_name == "amplitude_weighted_mse":
        logger.info("AmplitudeWeightedMSELoss is loaded as the loss function.")
        loss_fn = AmplitudeWeightedMSELoss(epsilon=1e-6, a=0.8)
    elif loss_name == "mse":
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
