"""CNN backbone models for ASTF-net."""

import torch
import torch.nn as nn

from astfnet.models.backbone import register_backbone


@register_backbone("simplecnn")
class SimpleCNN(nn.Module):
    """A simple CNN network for deconvolution learning tasks."""

    def __init__(self, in_channels: int = 2, output_length: int = 501) -> None:
        """Initialize SimpleCNN.

        Args:
            in_channels: Number of input channels (e.g., 2 for target waveform and EGF)
            output_length: Length of output sequence (e.g., source time function length)
        """
        super().__init__()

        # Feature extraction module
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )

        # Fully connected module
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, output_length),
            nn.Softplus(),
        )

    def forward(self, target_waveform: torch.Tensor, egf: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            target_waveform: Target waveform (batch_size, seq_len)
            egf: Empirical Green's function (batch_size, seq_len)

        Returns:
            Predicted source time function (batch_size, output_length)
        """
        x = torch.stack([target_waveform, egf], dim=1)
        x = self.feature_extractor(x)
        x = self.regressor(x)
        return x
