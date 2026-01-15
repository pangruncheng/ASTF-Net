"""CNN models for ASTF-net."""

import logging
from typing import Dict, List, Literal, Optional, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn

from astfnet.models.optim import load_loss as optim

logger = logging.getLogger(__name__)


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


class SimpleCNN_Tunable(nn.Module):
    """Tunable version of the original SimpleCNN.

    When parameters take default values, this model is EXACTLY
    equivalent to the original SimpleCNN.
    """

    def __init__(
        self,
        in_channels: int = 2,
        output_length: int = 256,
        *,
        base_channels: int = 32,
        fc_hidden_dim: int = 1024,
        dropout_rate: float = 0.5,
        output_activation: str = "softplus",
    ) -> None:
        """Initialize the tunable SimpleCNN.

        Args:
            in_channels: Number of input channels.
            output_length: Length of output sequence.
            base_channels: Number of channels in the first convolution layer.
            fc_hidden_dim: Hidden dimension of the fully connected layer.
            dropout_rate: Dropout rate applied in the fully connected layer.
            output_activation: Activation function of the output layer.
        """
        super().__init__()

        # -------- Feature extractor (identical topology) --------
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(base_channels, base_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(base_channels * 4, base_channels * 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(base_channels * 8, base_channels * 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        # -------- Fully connected head --------
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base_channels * 16 * 8, fc_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(fc_hidden_dim, output_length),
            self._get_activation(output_activation),
        )

    def _get_activation(self, name: str) -> nn.Module:
        """Return output activation module.

        Args:
            name: Name of activation function.

        Returns:
            PyTorch activation module.
        """
        name = name.lower()
        if name == "softplus":
            return nn.Softplus()
        if name == "relu":
            return nn.ReLU()
        if name == "identity":
            return nn.Identity()
        raise ValueError(f"Unsupported activation: {name}")

    def forward(
        self,
        target_waveform: torch.Tensor,
        egf: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            target_waveform: Target waveform tensor.
            egf: Empirical Green's function tensor.

        Returns:
            Predicted source time function.
        """
        x = torch.stack([target_waveform, egf], dim=1)
        x = self.feature_extractor(x)
        return self.regressor(x)


class PLCNN(pl.LightningModule):
    """PyTorch Lightning module for CNN with training/validation/test logic."""

    def __init__(self, config: Dict[str, Union[str, float, int]]) -> None:
        """Initialize PLCNN.

        Args:
            config: Configuration dictionary containing model hyperparameters
        """
        super().__init__()
        in_channels = config.get("in_channels", 2)
        output_length = config.get("output_length", 501)
        lr = config.get("lr", 1e-3)
        self.save_hyperparameters(config)
        # self.model = SimpleCNN(in_channels=in_channels, output_length=output_length)
        model_type = config.get("model_name", "simple")
        if model_type == "simplecnn":
            self.model = SimpleCNN(in_channels=in_channels, output_length=output_length)
        elif model_type == "simplecnn_tunable":
            self.model = SimpleCNN_Tunable(
                in_channels=config["in_channels"],
                output_length=config["output_length"],
                base_channels=config["base_channels"],
                fc_hidden_dim=config["fc_hidden_dim"],
                dropout_rate=config["dropout_rate"],
                output_activation=config["output_activation"],
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        self.loss_fn = optim(config)
        self.lr = lr
        self.test_preds: List[torch.Tensor] = []
        self.test_trues: List[torch.Tensor] = []

    def forward(self, target_waveform: torch.Tensor, egf: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.

        Args:
            target_waveform: Input target waveform
            egf: Input empirical Green's function

        Returns:
            Model output
        """
        return self.model(target_waveform, egf)

    def safe_log(
        self,
        name: str,
        value: Union[torch.Tensor, float],  # float or Tensor
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
        """Safely log a value, checking for finite values first.

        Args:
            name: Name of the metric to log
            value: Value to log
            prog_bar: If True logs to the progress bar
            logger: If True logs to the logger
            on_step: If True logs at each step
            on_epoch: If True logs epoch accumulated metrics
            reduce_fx: Reduction function over step values for end of epoch
            sync_dist: If True reduces the metric across devices
            sync_dist_group: The ddp group to sync across
            add_dataloader_idx: If True adds dataloader index to metric name
            batch_size: Current batch size
            metric_attribute: Attribute to store the metric
            rank_zero_only: If True logs only on rank 0
        """
        # Tensor
        tensor_value = value if isinstance(value, torch.Tensor) else torch.tensor(value)

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
        """Training step with loss computation and logging.

        Args:
            batch: Input batch containing target, egf and astf
            batch_idx: Index of the current batch

        Returns:
            Computed loss value
        """
        for key in ["target", "egf", "astf"]:
            if not torch.isfinite(batch[key]).all():
                self.print(f"[ERROR] batch[{key}] contains NaN or Inf at step {self.global_step}")
                self.print(f"  stats: min={batch[key].min()}, max={batch[key].max()}, mean={batch[key].mean()}")
                return torch.tensor(0.0, requires_grad=True, device=self.device)

        y_hat = self(batch["target"], batch["egf"])
        if not torch.isfinite(y_hat).all():
            self.print(f"[ERROR] Model output y_hat contains NaN or Inf at step {self.global_step}")
            return torch.tensor(0.0, requires_grad=True, device=self.device)

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

    def validation_step(self: "PLCNN", batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step with loss computation and logging.

        Args:
            batch: Input batch containing target, egf and astf
            batch_idx: Index of the current batch

        Returns:
            Computed loss value
        """
        y_hat = self(batch["target"], batch["egf"])

        loss = self.loss_fn(y_hat, batch["astf"])

        self.safe_log("val/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def on_test_start(self) -> None:
        """Initialize test predictions and ground truth lists."""
        self.test_preds = []
        self.test_trues = []

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Test step with loss computation and result collection.

        Args:
            batch: Input batch containing target, egf and astf
            batch_idx: Index of the current batch

        Returns:
            Computed loss value
        """
        y_hat = self(batch["target"], batch["egf"])
        self.test_preds.append(y_hat.detach().cpu())
        self.test_trues.append(batch["astf"].detach().cpu())

        loss = self.loss_fn(y_hat, batch["astf"])

        self.safe_log("test/loss", loss, prog_bar=True)
        return loss

    def on_test_epoch_end(self) -> None:
        """Concatenate all test predictions and ground truth values."""
        preds = torch.cat(self.test_preds, dim=0)
        trues = torch.cat(self.test_trues, dim=0)
        self.test_preds = preds
        self.test_trues = trues

    def configure_optimizers(self) -> Dict[str, object]:
        """Configure optimizers and learning rate schedulers.

        Returns:
            Dictionary containing optimizer and scheduler configuration
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
