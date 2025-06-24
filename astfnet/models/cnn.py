"""
CNN models for ASTF-net.
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl


class SimpleCNN(nn.Module):
    """
    A simple CNN network for deconvolution learning tasks.
    """

    def __init__(self, in_channels: int = 2, output_length: int = 501):
        """
        Initialize SimpleCNN.

        Args:
            in_channels: Number of input channels (e.g., 2 for target waveform and EGF)
            output_length: Length of output sequence (e.g., source time function length)
        """
        super(SimpleCNN, self).__init__()

        # Feature extraction module
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(
                in_channels, 32, kernel_size=3, padding=1
            ),  # 32 output channels, kernel size 3
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # Pooling, sequence length halved
            nn.Conv1d(32, 64, kernel_size=3, padding=1),  # 64 output channels
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # Pooling again, sequence length halved again
            nn.Conv1d(64, 128, kernel_size=3, padding=1),  # 128 output channels
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # Final pooling
        )

        # Fully connected module
        self.regressor = nn.Sequential(
            nn.Flatten(),  # Flatten features
            nn.Linear(
                4096, 1024
            ),  # Assume sequence length reduced to output_length // 8 after 3 poolings
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, output_length),  # Output source time function length
            nn.Softplus(),  # Ensure non-negative output
        )

    def forward(self, target_waveform: torch.Tensor, egf: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            target_waveform: Target waveform (batch_size, seq_len)
            egf: Empirical Green's function (batch_size, seq_len)

        Returns:
            Predicted source time function (batch_size, output_length)
        """
        # Concatenate target_waveform and egf as (batch_size, 2, seq_len)
        x = torch.stack(
            [target_waveform, egf], dim=1
        )  # Concatenate in channel dimension
        x = self.feature_extractor(x)  # Feature extraction
        x = self.regressor(x)  # Fully connected regressor
        return x


class PLSimpleCNN(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        in_channels = config.get("in_channels", 2)
        output_length = config.get("output_length", 501)
        lr = config.get("lr", 1e-3)
        self.save_hyperparameters(config)
        self.model = SimpleCNN(in_channels=in_channels, output_length=output_length)
        self.loss_fn = nn.MSELoss()
        self.lr = lr

    def forward(self, target_waveform, egf):
        return self.model(target_waveform, egf)

    def training_step(self, batch, batch_idx):
        y_hat = self(batch["target"], batch["egf"])
        loss = self.loss_fn(y_hat, batch["astf"])
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        y_hat = self(batch["target"], batch["egf"])
        loss = self.loss_fn(y_hat, batch["astf"])
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
