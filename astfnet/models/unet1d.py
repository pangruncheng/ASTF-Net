"""UNet models for ASTF-net."""

from typing import Dict, List, Optional, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing_extensions import Literal

from astfnet.models.optim import load_loss as optim


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


class UNet1D_2(nn.Module):
    """A 1D U-Net for ASTF prediction from target waveform and EGF."""

    def __init__(
        self,
        n_channels: int = 2,
        n_classes: int = 1,
        linear: bool = True,
        dropout_shallow: float = 0.1,
        dropout_deep: float = 0.3,
    ) -> None:
        """Initialize the 1D U-Net model.

        Args:
            n_channels: Number of input channels, usually 2 for EGF and target waveform.
            n_classes: Number of output channels, usually 1 for ASTF.
            linear: Whether to use linear upsampling.
            dropout_shallow: Dropout probability in shallow layers.
            dropout_deep: Dropout probability in deep layers.
        """
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.linear = linear

        self.inc = DoubleConv1D(n_channels, 64, dropout=dropout_shallow)
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


class PLUNet1D(pl.LightningModule):
    """PyTorch Lightning module for UNet1D training, validation, and testing."""

    def __init__(self, config: Dict[str, Union[str, float, int, Dict[str, object]]]) -> None:
        """Initialize the Lightning module.

        Args:
            config: Configuration dictionary for model, optimizer, and training settings.
        """
        super().__init__()
        in_channels = int(config.get("in_channels", 2))
        lr = float(config.get("lr", 1e-3))

        self.save_hyperparameters(config)
        model_type = config.get("model_name", "unet1d")

        if model_type == "unet1d_2":
            self.model = UNet1D_2(
                n_channels=in_channels,
                n_classes=1,
                linear=True,
                dropout_shallow=float(config.get("dropout_shallow", 0.1)),
                dropout_deep=float(config.get("dropout_deep", 0.3)),
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        self.loss_fn = optim(config)
        self.lr = lr
        self.test_preds: List[torch.Tensor] = []
        self.test_trues: List[torch.Tensor] = []

    def forward(self, target_waveform: torch.Tensor, egf: torch.Tensor) -> torch.Tensor:
        """Forward pass through the wrapped model.

        Args:
            target_waveform: Input target waveform.
            egf: Input empirical Green's function.

        Returns:
            Model prediction.
        """
        return self.model(target_waveform, egf)

    def safe_log(
        self,
        name: str,
        value: Union[torch.Tensor, float],
        *,
        prog_bar: bool = False,
        logger: bool = True,
        on_step: Optional[bool] = None,
        on_epoch: Optional[bool] = None,
        reduce_fx: Literal["mean", "sum", "min", "max"] = "mean",
        sync_dist: bool = False,
        sync_dist_group: Optional[object] = None,
        add_dataloader_idx: bool = True,
        batch_size: Optional[int] = None,
        metric_attribute: Optional[str] = None,
        rank_zero_only: bool = False,
    ) -> None:
        """Safely log a value only when it is finite.

        Args:
            name: Name of the metric to log.
            value: Value to log.
            prog_bar: Whether to show the metric in the progress bar.
            logger: Whether to send the metric to the logger.
            on_step: Whether to log on each step.
            on_epoch: Whether to log on each epoch.
            reduce_fx: Reduction function over step values.
            sync_dist: Whether to synchronize the metric across devices.
            sync_dist_group: Process group for distributed synchronization.
            add_dataloader_idx: Whether to append the dataloader index to the metric name.
            batch_size: Batch size for proper averaging.
            metric_attribute: Metric attribute name for Lightning internals.
            rank_zero_only: Whether to log only on rank zero.
        """
        tensor_value = value if isinstance(value, torch.Tensor) else torch.tensor(value, device=self.device)

        if torch.isfinite(tensor_value):
            self.log(
                name,
                value,
                prog_bar=prog_bar,
                logger=logger,
                on_step=on_step,
                on_epoch=on_epoch,
                reduce_fx=reduce_fx,
                sync_dist=sync_dist,
                sync_dist_group=sync_dist_group,
                add_dataloader_idx=add_dataloader_idx,
                batch_size=batch_size,
                metric_attribute=metric_attribute,
                rank_zero_only=rank_zero_only,
            )
        else:
            self.print(f"[WARNING] {name} is NaN or Inf. Skipped logging.")

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Run one training step.

        Args:
            batch: Input batch containing target, egf, and astf.
            batch_idx: Index of the current batch.

        Returns:
            Loss tensor.
        """
        del batch_idx

        for key in ["target", "egf", "astf"]:
            if not torch.isfinite(batch[key]).all():
                self.print(f"[ERROR] batch[{key}] contains NaN or Inf at step {self.global_step}")
                self.print(f"  stats: min={batch[key].min()}, max={batch[key].max()}, mean={batch[key].mean()}")
                return torch.tensor(0.0, requires_grad=True, device=self.device)

        y_hat = self(batch["target"], batch["egf"])
        if not torch.isfinite(y_hat).all():
            self.print(f"[ERROR] Model output y_hat contains NaN or Inf at step {self.global_step}")
            return torch.tensor(0.0, requires_grad=True, device=self.device)

        loss_name = self.hparams.get("loss", "mse")
        if loss_name == "convalignLoss":
            loss, parts = self.loss_fn(
                pred_astf=y_hat,
                true_astf=batch["astf"],
                egf=batch["egf"],
                target_waveform=batch["target"],
            )
            self.safe_log("train/loss_astf", parts["astf"], on_epoch=True, prog_bar=False)
            self.safe_log("train/loss_conv", parts["conv"], on_epoch=True, prog_bar=False)
        else:
            loss = self.loss_fn(y_hat, batch["astf"])

        if not torch.isfinite(loss):
            self.print(f"[ERROR] Loss is NaN at step {self.global_step}")
            self.print(f"  y_hat stats: min={y_hat.min()}, max={y_hat.max()}, mean={y_hat.mean()}")
            self.print(
                f"  astf stats: min={batch['astf'].min()}, max={batch['astf'].max()}, mean={batch['astf'].mean()}"
            )
            return torch.tensor(0.0, requires_grad=True, device=self.device)

        self.safe_log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.safe_log("lr-Adam", lr, on_step=False, on_epoch=True, prog_bar=True)

        if self.global_step % 1000 == 0:
            self.print(f"[step {self.global_step}] loss: {loss.item():.5f}")

        return loss if torch.isfinite(loss) else torch.tensor(0.0, requires_grad=True, device=self.device)

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Run one validation step.

        Args:
            batch: Input batch containing target, egf, and astf.
            batch_idx: Index of the current batch.

        Returns:
            Loss tensor.
        """
        del batch_idx

        y_hat = self(batch["target"], batch["egf"])
        loss_name = self.hparams.get("loss", "mse")

        if loss_name == "convalignLoss":
            loss, parts = self.loss_fn(
                pred_astf=y_hat,
                true_astf=batch["astf"],
                egf=batch["egf"],
                target_waveform=batch["target"],
            )
            self.safe_log("val/loss_astf", parts["astf"], on_epoch=True)
            self.safe_log("val/loss_conv", parts["conv"], on_epoch=True)
        else:
            loss = self.loss_fn(y_hat, batch["astf"])

        self.safe_log("val/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def on_test_start(self) -> None:
        """Initialize containers for test predictions and labels."""
        self.test_preds = []
        self.test_trues = []

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Run one test step.

        Args:
            batch: Input batch containing target, egf, and astf.
            batch_idx: Index of the current batch.

        Returns:
            Loss tensor.
        """
        del batch_idx

        y_hat = self(batch["target"], batch["egf"])
        self.test_preds.append(y_hat.detach().cpu())
        self.test_trues.append(batch["astf"].detach().cpu())

        loss_name = self.hparams.get("loss", "mse")
        if loss_name == "convalignLoss":
            loss, _ = self.loss_fn(
                pred_astf=y_hat,
                true_astf=batch["astf"],
                egf=batch["egf"],
                target_waveform=batch["target"],
            )
        else:
            loss = self.loss_fn(y_hat, batch["astf"])

        self.safe_log("test/loss", loss, prog_bar=True)
        return loss

    def on_test_epoch_end(self) -> None:
        """Concatenate predictions and labels collected during testing."""
        preds = torch.cat(self.test_preds, dim=0)
        trues = torch.cat(self.test_trues, dim=0)
        self.test_preds = preds
        self.test_trues = trues

    def configure_optimizers(self) -> Dict[str, object]:
        """Configure the optimizer and learning-rate scheduler.

        Returns:
            Optimizer and scheduler configuration dictionary.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams["lr"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=self.hparams["callbacks"]["lr_scheduler"]["mode"],
            factor=self.hparams["callbacks"]["lr_scheduler"]["factor"],
            patience=self.hparams["callbacks"]["lr_scheduler"]["patience"],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": self.hparams["callbacks"]["lr_scheduler"]["monitor"],
                "interval": "epoch",
                "frequency": 1,
            },
        }
