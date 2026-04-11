"""Unified PyTorch Lightning module for all ASTF-net backbones."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import pytorch_lightning as pl
import torch

from astfnet.models.backbone import build_backbone
from astfnet.models.optim import load_loss

logger = logging.getLogger(__name__)


class ASTFModule(pl.LightningModule):
    """Backbone-agnostic Lightning module for ASTF regression.

    Args:
        config: Flat configuration dictionary.  ``model_name`` selects the
            backbone; ``loss`` selects the loss function.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialise the module from a config dict."""
        super().__init__()
        self.save_hyperparameters(config)

        self.model = build_backbone(config)
        self.loss_fn = load_loss(config)
        self.lr = float(config.get("lr", 1e-3))

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, target_waveform: torch.Tensor, egf: torch.Tensor) -> torch.Tensor:
        """Run the backbone forward pass."""
        return self.model(target_waveform, egf)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Compute and log training loss."""
        y_hat = self(batch["target"], batch["egf"])
        loss = self.loss_fn(y_hat, batch["astf"])
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def on_validation_start(self) -> None:
        """Reset validation loss buffer."""
        self._val_losses: List[torch.Tensor] = []

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Compute validation loss and accumulate for epoch averaging."""
        y_hat = self(batch["target"], batch["egf"])
        loss = self.loss_fn(y_hat, batch["astf"])
        self._val_losses.append(loss)
        return loss

    def on_validation_epoch_end(self) -> None:
        """Log mean validation loss and clear the buffer."""
        avg_loss = torch.stack(self._val_losses).mean()
        self.log("val/loss_epoch", avg_loss, prog_bar=True)
        self._val_losses.clear()

    # ------------------------------------------------------------------
    # Testing
    # ------------------------------------------------------------------

    def on_test_start(self) -> None:
        """Reset prediction buffers."""
        self.test_preds: List[torch.Tensor] = []
        self.test_trues: List[torch.Tensor] = []

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Collect predictions and compute test loss."""
        y_hat = self(batch["target"], batch["egf"])
        self.test_preds.append(y_hat.detach().cpu())
        self.test_trues.append(batch["astf"].detach().cpu())

        loss = self.loss_fn(y_hat, batch["astf"])
        self.log("test/loss", loss, prog_bar=True)
        return loss

    def on_test_epoch_end(self) -> None:
        """Concatenate collected predictions and ground truths."""
        self.test_preds = torch.cat(self.test_preds, dim=0)
        self.test_trues = torch.cat(self.test_trues, dim=0)

    # ------------------------------------------------------------------
    # Optimiser
    # ------------------------------------------------------------------

    def configure_optimizers(self) -> Dict[str, Any]:
        """Build Adam + optional ReduceLROnPlateau from config."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        callbacks_cfg = self.hparams.get("callbacks", {})
        lr_sched_cfg = callbacks_cfg.get("lr_scheduler") if isinstance(callbacks_cfg, dict) else None

        if lr_sched_cfg is None:
            return {"optimizer": optimizer}

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=lr_sched_cfg.get("mode", "min"),
            factor=lr_sched_cfg.get("factor", 0.5),
            patience=lr_sched_cfg.get("patience", 10),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": lr_sched_cfg.get("monitor", "val/loss_epoch"),
                "interval": "epoch",
                "frequency": 1,
            },
        }
