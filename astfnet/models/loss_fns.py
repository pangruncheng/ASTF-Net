import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedMSE(nn.Module):
    """Weighted Mean Squared Error loss module."""

    def __init__(self) -> None:
        """Initialize the WeightedMSE loss module."""
        super(WeightedMSE, self).__init__()

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
        super(EffectiveRegionWeightedMSELoss, self).__init__()

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
        super(NonZeroWeightedMSE, self).__init__()

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
        super(AmplitudeWeightedMSELoss, self).__init__()
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


class ConvAlignLoss(nn.Module):
    """Loss combining ASTF MSE and convolution alignment loss."""

    def __init__(self, alpha: float = 1.0, crop_len: int = 256) -> None:
        """Initialize the ConvAlignLoss module.

        Args:
            alpha: Weight for the convolution alignment loss term.
            crop_len: Length of waveform segment to compare (center cropped).
        """
        super().__init__()
        self.alpha = alpha
        self.crop_len = crop_len
        self.mse = nn.MSELoss()

    def forward(
        self,
        pred_astf: torch.Tensor,
        true_astf: torch.Tensor,
        egf: torch.Tensor,
        target_waveform: torch.Tensor,
        roll: torch.Tensor = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Calculate the combined ASTF and convolution alignment loss.

        Args:
            pred_astf: Predicted ASTF with shape (B, L).
            true_astf: True ASTF with shape (B, L).
            egf: EGF with shape (B, L).
            target_waveform: Target waveform with shape (B, L3).
            roll: Precomputed shift (alignment) values with shape (B,) or None.

        Returns:
            A tuple containing:
                - total_loss: Combined loss value.
                - Dictionary with individual loss terms ("astf" and "conv").
        """
        B, L = pred_astf.shape
        L_target = target_waveform.shape[1]

        # 1. Standard MSE loss for ASTF
        loss_astf = self.mse(pred_astf, true_astf)

        # 2. Batch convolution: (B,1,L) * (B,1,L) groups=B implements B 1D convolutions
        pred_astf_ = pred_astf.unsqueeze(1)  # (B, 1, L)
        egf_ = egf.unsqueeze(1)  # (B, 1, L)

        # Reshape for grouped convolution
        input_reshaped = pred_astf_.transpose(0, 1)  # (1, B, L)
        weight_reshaped = egf_  # (B, 1, L)

        # Use grouped convolution manually
        conv_result = F.conv1d(input_reshaped, weight_reshaped, groups=pred_astf.shape[0])
        conv_result = conv_result.transpose(0, 1).squeeze(1)  # (B, L_out)

        # 3. Compute alignment shifts
        if roll is None:
            conv_len = conv_result.shape[1]
            # Pad target_waveform to match conv_result length for FFT calculation
            target_padded = F.pad(target_waveform, (0, conv_len - L_target))  # (B, conv_len)

            # Compute cross-correlation using FFT
            fft_conv = torch.fft.rfft(conv_result, n=2 * conv_len - 1)
            fft_target = torch.fft.rfft(target_padded, n=2 * conv_len - 1)
            cc = torch.fft.irfft(fft_conv * torch.conj(fft_target), n=2 * conv_len - 1)  # (B, 2*conv_len -1)

            # Find maximum correlation position to get shift vector (B,)
            shifts = torch.argmax(cc, dim=1) - (conv_len - 1)
        else:
            shifts = roll.to(torch.long)

        # 4. Roll and center crop the aligned results
        conv_aligned = self._batch_roll(conv_result, -shifts)  # Negative shift for left shift
        conv_cropped = self._batch_center_crop(conv_aligned, self.crop_len)
        target_cropped = self._batch_center_crop(target_waveform, self.crop_len)

        # 5. MSE loss between convolution result and target
        loss_conv = F.mse_loss(conv_cropped, target_cropped)

        total_loss = loss_astf + self.alpha * loss_conv
        return total_loss, {"astf": loss_astf, "conv": loss_conv}

    def _batch_center_crop(self, x: torch.Tensor, crop_len: int) -> torch.Tensor:
        """Center crop batch data along the last dimension.

        Args:
            x: Input tensor with shape (B, L).
            crop_len: Length to crop to.

        Returns:
            Center-cropped tensor with shape (B, crop_len).
        """
        L = x.size(1)
        start = (L - crop_len) // 2
        return x[:, start : start + crop_len]

    def _batch_roll(self, x: torch.Tensor, shifts: torch.Tensor) -> torch.Tensor:
        """Roll each element in batch x by corresponding shift value.

        Args:
            x: Input tensor with shape (B, L).
            shifts: Shift values with shape (B,).

        Returns:
            Rolled tensor with same shape as input.
        """
        B, L = x.size()
        rolled = torch.empty_like(x)
        for i in range(B):
            rolled[i] = torch.roll(x[i], shifts[i].item())
        return rolled
