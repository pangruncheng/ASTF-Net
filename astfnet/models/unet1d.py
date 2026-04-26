"""UNet backbone models for ASTF-net."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from astfnet.models.backbone import register_backbone


class DoubleConv1D(nn.Module):
    """Apply two blocks of Conv1d, BatchNorm1d, ReLU, and Dropout."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: Optional[int] = None,
        dropout: float = 0.0,
    ) -> None:
        """Initialize the double convolution block.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            mid_channels: Number of intermediate channels. If None, use out_channels.
            dropout: Dropout probability.
        """
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the forward pass.

        Args:
            x: Input tensor.

        Returns:
            Output tensor after double convolution.
        """
        return self.double_conv(x)


class Down1D(nn.Module):
    """Downscale with max pooling followed by double convolution."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        """Initialize the downsampling block.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            dropout: Dropout probability.
        """
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(2),
            DoubleConv1D(in_channels, out_channels, dropout=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the forward pass.

        Args:
            x: Input tensor.

        Returns:
            Downsampled output tensor.
        """
        return self.maxpool_conv(x)


class Up1D(nn.Module):
    """Upscale and then apply double convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        linear: bool = True,
        dropout: float = 0.0,
    ) -> None:
        """Initialize the upsampling block.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            linear: Whether to use linear upsampling. If False, use transposed convolution.
            dropout: Dropout probability.
        """
        super().__init__()
        if linear:
            self.up = nn.Upsample(scale_factor=2, mode="linear", align_corners=True)
            self.conv = DoubleConv1D(
                in_channels,
                out_channels,
                in_channels // 2,
                dropout=dropout,
            )
        else:
            self.up = nn.ConvTranspose1d(
                in_channels,
                in_channels // 2,
                kernel_size=2,
                stride=2,
            )
            self.conv = DoubleConv1D(in_channels, out_channels, dropout=dropout)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Run the forward pass.

        Args:
            x1: Input tensor from the decoder branch.
            x2: Skip connection tensor from the encoder branch.

        Returns:
            Output tensor after upsampling, concatenation, and convolution.
        """
        x1 = self.up(x1)
        diff = x2.size(2) - x1.size(2)
        x1 = F.pad(x1, [diff // 2, diff - diff // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv1D(nn.Module):
    """Apply the output convolution and Softplus activation."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Initialize the output block.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
        """
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.softplus = nn.Softplus()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the forward pass.

        Args:
            x: Input tensor.

        Returns:
            Non-negative output tensor.
        """
        x = self.conv(x)
        return self.softplus(x)


@register_backbone("unet1d")
class UNet1D(nn.Module):
    """A 1D U-Net for ASTF prediction from target waveform and EGF."""

    def __init__(
        self,
        in_channels: int = 2,
        n_classes: int = 1,
        linear: bool = True,
        dropout_shallow: float = 0.1,
        dropout_deep: float = 0.3,
    ) -> None:
        """Initialize the 1D U-Net model.

        Args:
            in_channels: Number of input channels, usually 2 for EGF and target waveform.
            n_classes: Number of output channels, usually 1 for ASTF.
            linear: Whether to use linear upsampling.
            dropout_shallow: Dropout probability in shallow layers.
            dropout_deep: Dropout probability in deep layers.
        """
        super().__init__()
        self.n_channels = in_channels
        self.n_classes = n_classes
        self.linear = linear

        self.inc = DoubleConv1D(in_channels, 64, dropout=dropout_shallow)
        self.down1 = Down1D(64, 128, dropout=dropout_shallow)
        self.down2 = Down1D(128, 256, dropout=dropout_shallow)
        self.down3 = Down1D(256, 512, dropout=dropout_shallow)
        factor = 2 if linear else 1
        self.down4 = Down1D(512, 1024 // factor, dropout=dropout_deep)

        self.up1 = Up1D(1024, 512 // factor, linear, dropout=dropout_deep)
        self.up2 = Up1D(512, 256 // factor, linear, dropout=dropout_shallow)
        self.up3 = Up1D(256, 128 // factor, linear, dropout=dropout_shallow)
        self.up4 = Up1D(128, 64, linear, dropout=dropout_shallow)

        self.outc = OutConv1D(64, n_classes)

    def forward(self, target_waveform: torch.Tensor, egf: torch.Tensor) -> torch.Tensor:
        """Run the forward pass.

        Args:
            target_waveform: Input target waveform with shape [B, L] or [B, 1, L].
            egf: Input EGF with shape [B, L] or [B, 1, L].

        Returns:
            Predicted ASTF with shape [B, L].
        """
        if target_waveform.ndim == 2:
            target_waveform = target_waveform.unsqueeze(1)
        if egf.ndim == 2:
            egf = egf.unsqueeze(1)

        x = torch.cat([target_waveform, egf], dim=1)

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        out = self.outc(x)
        return out.squeeze(1)
