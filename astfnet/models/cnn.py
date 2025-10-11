"""CNN models for ASTF-net."""

from typing import Dict, List, Literal, Optional, Union
from typing_extensions import Literal

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from astfnet.models.optim import load_loss as optim


class SimpleCNN(nn.Module):
    """A simple CNN network for deconvolution learning tasks."""

    def __init__(self, in_channels: int = 2, output_length: int = 501) -> None:
        """Initialize SimpleCNN.

        Args:
            in_channels: Number of input channels (e.g., 2 for target waveform and EGF)
            output_length: Length of output sequence (e.g., source time function length)
        """
        super(SimpleCNN, self).__init__()

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

    
class DeepCNN(nn.Module):
    """A deeper CNN network for deconvolution learning tasks with input length 256."""

    def __init__(self, in_channels: int = 2, output_length: int = 501) -> None:
        super(DeepCNN, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),  # 256 -> 128

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),  # 128 -> 64

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),  # 64 -> 32

            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),  # 32 -> 16

            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),  # 16 -> 8

            nn.Conv1d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),  # 8 -> 4

            nn.Conv1d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),  # 4 -> 2
        )

        # flatten size: 512 * 2 = 1024
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, output_length),
            nn.Softplus(),
        )

    def forward(self, target_waveform: torch.Tensor, egf: torch.Tensor) -> torch.Tensor:
        x = torch.stack([target_waveform, egf], dim=1)  # (B, 2, 256)
        x = self.feature_extractor(x)  # (B, 512, 2)
        x = self.regressor(x)  # (B, output_length)
        return x


class VGGStyleCNN(nn.Module):
    """A deeper VGG-style CNN for deconvolution learning."""

    def __init__(self, in_channels: int = 2, output_length: int = 501) -> None:
        super(VGGStyleCNN, self).__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv1d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            # Block 2
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            # Block 3
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            # Block 4
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(8),  # Make feature dimension fixed
        )

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 8, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, output_length),
            nn.Softplus(),
        )

    def forward(self, target_waveform: torch.Tensor, egf: torch.Tensor) -> torch.Tensor:
        x = torch.stack([target_waveform, egf], dim=1)  # (B, 2, L)
        x = self.features(x)
        x = self.regressor(x)
        return x
    
####FMnet######
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, activation=nn.ReLU):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.activation = activation()

    def forward(self, x):
        return self.activation(self.conv(x))


class FMNet1D(nn.Module):
    def __init__(self, in_channels=2, output_length=256):
        super().__init__()
        self.output_length = output_length

        # Encoder
        self.encoder = nn.Sequential(
            ConvBlock(in_channels, 64),
            nn.MaxPool1d(2),  # 256 -> 128

            ConvBlock(64, 128),
            nn.MaxPool1d(2),  # 128 -> 64

            ConvBlock(128, 256),
            nn.MaxPool1d(2),  # 64 -> 32

            ConvBlock(256, 512),
            nn.MaxPool1d(2),  # 32 -> 16
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='linear', align_corners=True),  # 16 -> 32
            ConvBlock(512, 256),

            nn.Upsample(scale_factor=2, mode='linear', align_corners=True),  # 32 -> 64
            ConvBlock(256, 128),

            nn.Upsample(scale_factor=2, mode='linear', align_corners=True),  # 64 -> 128
            ConvBlock(128, 64),

            nn.Upsample(scale_factor=2, mode='linear', align_corners=True),  # 128 -> 256
            ConvBlock(64, 32),
        )

        # Final conv to adjust output size
        self.final_conv = nn.Conv1d(32, 1, kernel_size=1)
        self.output_activation = nn.Softplus()

    def forward(self, target, egf):
        target = target.unsqueeze(1)  # (B, 1, 256)
        egf = egf.unsqueeze(1)        # (B, 1, 256)
        x = torch.cat([target, egf], dim=1)  # (B, 1, L) + (B, 1, L) -> (B, 2, L)
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.final_conv(x)
        if x.shape[-1] != self.output_length:
            x = F.interpolate(x, size=self.output_length, mode='linear', align_corners=True)
        return self.output_activation(x)
    
class AmplitudeFusionCNN(nn.Module):
    """CNN for two waveforms + MLP for amplitude/magnitude fusion."""

    def __init__(self, in_channels: int = 2, output_length: int = 501, aux_dim: int = 2) -> None:
        """
        Args:
            in_channels: number of input waveform channels (default=2)
            output_length: output sequence length
            aux_dim: number of auxiliary features (e.g., amplitude, magnitude)
        """
        super().__init__()
        self.output_length = output_length

        # CNN feature extractor for waveforms
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(8),  # fixed-length feature
        )

        # Linear extractor for auxiliary info (amplitude, magnitude)
        self.aux_extractor = nn.Sequential(
            nn.Linear(aux_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
        )

        # Fusion + regression
        self.regressor = nn.Sequential(
            nn.Linear(128 * 8 + 64, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, output_length),
            nn.Softplus(),
        )

    def forward(self, waveform1: torch.Tensor, waveform2: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform1: (B, L)
            waveform2: (B, L)
            aux: (B, aux_dim), e.g., amplitude, magnitude
        """
        x = torch.stack([waveform1, waveform2], dim=1)  # (B, 2, L)
        feat = self.feature_extractor(x)  # (B, 128, 8)
        feat = feat.flatten(1)  # (B, 128*8)

        aux_feat = self.aux_extractor(aux)  # (B, 64)

        fused = torch.cat([feat, aux_feat], dim=1)  # (B, feature_dim)
        out = self.regressor(fused)  # (B, output_length)
        return out


class CNNLSTMFusion(nn.Module):
    """CNN + LSTM fusion model: both inputs go through CNN and LSTM, then fused."""

    def __init__(self, hidden_dim: int = 128, num_layers: int = 1, output_length: int = 501):
        super().__init__()
        self.output_length = output_length

        # CNN branch (shared for both inputs)
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(16),
        )

        # LSTM branch (shared for both inputs)
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

        # Fusion + regression
        self.regressor = nn.Sequential(
            nn.Linear((64 * 16 + hidden_dim * 2) * 2, 256),  # ×2 because two inputs
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, output_length),
            nn.Softplus(),
        )

    def extract_features(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Extract features from one waveform using CNN and LSTM.
        Args:
            waveform: (B, L)
        Returns:
            feat: (B, 64*16 + hidden_dim*2)
        """
        # CNN branch
        x_cnn = waveform.unsqueeze(1)  # (B, 1, L)
        cnn_feat = self.cnn(x_cnn).flatten(1)  # (B, 64*16)

        # LSTM branch
        x_lstm = waveform.unsqueeze(-1)  # (B, L, 1)
        lstm_out, _ = self.lstm(x_lstm)  # (B, L, hidden*2)
        lstm_feat = lstm_out[:, -1, :]   # take last hidden state (B, hidden*2)

        # concat CNN + LSTM
        feat = torch.cat([cnn_feat, lstm_feat], dim=1)
        return feat

    def forward(self, waveform1: torch.Tensor, waveform2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform1: (B, L)
            waveform2: (B, L)
        Returns:
            out: (B, output_length)
        """
        feat1 = self.extract_features(waveform1)  # (B, feature_dim)
        feat2 = self.extract_features(waveform2)  # (B, feature_dim)

        fused = torch.cat([feat1, feat2], dim=1)  # (B, 2*feature_dim)
        out = self.regressor(fused)
        return out


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
        elif model_type == "vgg":
            self.model = VGGStyleCNN(in_channels=in_channels, output_length=output_length)
        elif model_type == "deepcnn":
            self.model = DeepCNN(in_channels=in_channels, output_length=output_length)
        elif model_type == "fmnet1d":
            self.model = FMNet1D(in_channels=in_channels, output_length=output_length)
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
        """Validation step with loss computation and logging.

        Args:
            batch: Input batch containing target, egf and astf
            batch_idx: Index of the current batch

        Returns:
            Computed loss value
        """
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

        self.safe_log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
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

        loss_name = self.hparams.get("loss", "mse")
        if loss_name == "convalignLoss":
            loss, _ = self.loss_fn(
                pred_astf=y_hat, true_astf=batch["astf"], egf=batch["egf"], target_waveform=batch["target"]
            )
        else:
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
