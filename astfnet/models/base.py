"""Unified PyTorch Lightning module for all ASTF-net backbones."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import pytorch_lightning as pl
import torch

from astfnet.models.backbone import build_backbone
from astfnet.models.optim import load_loss
from astfnet.models.optimizer import OptimizerFactory
from astfnet.models.scheduler import SchedulerFactory

logger = logging.getLogger(__name__)


class ASTFModule(pl.LightningModule):
    """Backbone-agnostic Lightning module for ASTF regression.

    Args:
        config: Top-level configuration dictionary.  ``model_name`` selects
            the backbone; ``loss`` selects the loss function.  Optimizer and
            scheduler settings are read from the ``optimizer`` and
            ``callbacks.lr_scheduler`` sub-dicts respectively (see
            :class:`~astfnet.models.optimizer.OptimizerFactory` and
            :class:`~astfnet.models.scheduler.SchedulerFactory`).
        optimizer_factory: Pre-built optimizer factory.
            :meth:`~astfnet.models.optimizer.OptimizerFactory.from_config`.
        scheduler_factory: Pre-built scheduler factory.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        optimizer_factory: Optional[OptimizerFactory] = None,
        scheduler_factory: Optional[SchedulerFactory] = None,
    ) -> None:
        """Initialise the module from a config dict."""
        super().__init__()
        self.save_hyperparameters(config)

        self.model = build_backbone(config)
        self.loss_fn = load_loss(config)

        if optimizer_factory is None:
            optimizer_factory = OptimizerFactory()

        self._optimizer_factory = optimizer_factory

        self._scheduler_factory = scheduler_factory

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
    # Optimizer
    # ------------------------------------------------------------------

    def configure_optimizers(self) -> Dict[str, Any]:
        """Build optimizer + optional LR scheduler from the stored factories.

        Both factories are constructed from the config dict passed to
        :meth:`__init__` unless explicit factory objects were supplied.
        Supported optimizers and schedulers are listed in
        :data:`~astfnet.models.optimizer.OPTIMIZER_REGISTRY` and
        :data:`~astfnet.models.scheduler.SCHEDULER_REGISTRY` respectively.
        """
        optimizer = self._optimizer_factory.build(self.parameters())

        if self._scheduler_factory is None:
            return {"optimizer": optimizer}

        return {
            "optimizer": optimizer,
            "lr_scheduler": self._scheduler_factory.lr_lightning_dict(optimizer),
        }
