"""
UNet models for ASTF-net.
"""

import torch
import torch.nn as nn


class UNet1D(nn.Module):
    """
    1D UNet model for seismic data processing.
    """

    def __init__(self):
        super(UNet1D, self).__init__()

        # Down: Encoder
        self.down1 = self.conv_block(2, 32)  # (B, 32, 256)
        self.pool1 = nn.MaxPool1d(2)  # → (B, 32, 128)

        self.down2 = self.conv_block(32, 64)  # (B, 64, 128)
        self.pool2 = nn.MaxPool1d(2)  # → (B, 64, 64)

        self.down3 = self.conv_block(64, 128)  # (B, 128, 64)
        self.pool3 = nn.MaxPool1d(2)  # → (B, 128, 32)

        # Bottom
        self.bottom = self.conv_block(128, 256)  # (B, 256, 32)

        # Up: Decoder
        self.up3 = self.up_block(256, 128)  # upsample to (B, 128, 64)
        self.dec3 = self.conv_block(256, 128)  # concat with down3 → (B, 128, 64)

        self.up2 = self.up_block(128, 64)
        self.dec2 = self.conv_block(128, 64)

        self.up1 = self.up_block(64, 32)
        self.dec1 = self.conv_block(64, 32)

        # Final output layer
        self.final_conv = nn.Conv1d(32, 1, kernel_size=1)

    def conv_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        """
        Create a convolutional block with batch normalization and ReLU.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels

        Returns:
            Sequential module with conv layers
        """
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def up_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        """
        Create an upsampling block.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels

        Returns:
            Sequential module with upsampling and conv
        """
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode="linear", align_corners=True),
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x1: First input tensor (e.g., target waveform)
            x2: Second input tensor (e.g., EGF)

        Returns:
            Output tensor
        """
        x = torch.stack([x1, x2], dim=1)  # (B, 2, 256)

        # Down path
        d1 = self.down1(x)  # (B, 32, 256)
        p1 = self.pool1(d1)  # (B, 32, 128)

        d2 = self.down2(p1)  # (B, 64, 128)
        p2 = self.pool2(d2)  # (B, 64, 64)

        d3 = self.down3(p2)  # (B, 128, 64)
        p3 = self.pool3(d3)  # (B, 128, 32)

        # Bottom
        bn = self.bottom(p3)  # (B, 256, 32)

        # Up path
        up3 = self.up3(bn)  # (B, 128, 64)
        up3 = torch.cat([up3, d3], dim=1)
        dec3 = self.dec3(up3)  # (B, 128, 64)

        up2 = self.up2(dec3)  # (B, 64, 128)
        up2 = torch.cat([up2, d2], dim=1)
        dec2 = self.dec2(up2)  # (B, 64, 128)

        up1 = self.up1(dec2)  # (B, 32, 256)
        up1 = torch.cat([up1, d1], dim=1)
        dec1 = self.dec1(up1)  # (B, 32, 256)

        out = self.final_conv(dec1)  # (B, 1, 256)
        return out.squeeze(1)  # (B, 256)
