import torch
import torch.nn as nn


class WeightedMSE(nn.Module):
    def __init__(self):
        super(WeightedMSE, self).__init__()

    def forward(
        self, y_pred: torch.Tensor, y_true: torch.Tensor, weights: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            y_pred: (batch, seq_len) predicted values
            y_true: (batch, seq_len) true values
            weights: (batch,) weights for each batch
        Returns:
            weighted_mse_batch: (batch,) weighted mse for each batch
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
    def __init__(self):
        super(EffectiveRegionWeightedMSELoss, self).__init__()

    def forward(
        self, y_pred: torch.Tensor, y_true: torch.Tensor, weights: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            y_pred: (batch, seq_len) predicted values
            y_true: (batch, seq_len) true values
            weights: (batch,) weights for each batch
        Returns:
            weighted_mse_batch: (batch,) weighted mse for each batch
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
    def __init__(self):
        super(NonZeroWeightedMSE, self).__init__()

    def forward(
        self, y_pred: torch.Tensor, y_true: torch.Tensor, weights: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            y_pred: (batch, seq_len) predicted values
            y_true: (batch, seq_len) true values
            weights: (batch,) weights for each batch
        Returns:
            weighted_mse_batch: (batch,) weighted mse for each batch
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
