"""LR-scheduler factory."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union

import torch
from omegaconf import DictConfig, OmegaConf

from astfnet.models._factory_utils import build_from_config

SCHEDULER_REGISTRY: Dict[str, Any] = {
    "ReduceLROnPlateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
}

# Keys consumed by Lightning / this wrapper — not forwarded to the scheduler.
_LIGHTNING_SCHED_KEYS = {"name", "monitor", "interval", "frequency"}


@dataclass
class SchedulerFactory:
    """Factory that instantiates a PyTorch LR scheduler.

    Configure via ``callbacks.lr_scheduler`` block in YAML::

        callbacks:
          lr_scheduler:
            name: ReduceLROnPlateau  # one of SCHEDULER_REGISTRY keys
            monitor: "val/loss_epoch"
            interval: epoch
            frequency: 1
            mode: min          # scheduler-specific kwargs below this line
            factor: 0.5
            patience: 10

    Attributes:
        name: Key in :data:`SCHEDULER_REGISTRY`.
        monitor: Metric name passed to Lightning (used by ``ReduceLROnPlateau``).
        interval: ``"epoch"`` or ``"step"`` — passed to Lightning.
        frequency: How often to call the scheduler — passed to Lightning.
        kwargs: Scheduler-constructor keyword arguments (everything in the
            YAML block that is not ``name``, ``monitor``, ``interval``, or
            ``frequency``).
    """

    name: str = "ReduceLROnPlateau"
    monitor: str = "val/loss_epoch"
    interval: str = "epoch"
    frequency: int = 1
    kwargs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the config values."""
        if self.name not in SCHEDULER_REGISTRY:
            raise ValueError(f"Unknown lr_scheduler {self.name!r}. Supported: {list(SCHEDULER_REGISTRY)}.")
        if self.interval not in {"epoch", "step"}:
            raise ValueError(f"interval must be 'epoch' or 'step', got {self.interval!r}.")
        if self.frequency < 1:
            raise ValueError(f"frequency must be >= 1, got {self.frequency!r}.")

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    def build(self, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler.LRScheduler:
        """Instantiate the scheduler wrapping *optimizer*.

        Args:
            optimizer: The optimizer to wrap.

        Returns:
            A configured LR scheduler.

        Raises:
            ValueError: If :attr:`name` is not in :data:`SCHEDULER_REGISTRY`.
        """
        return SCHEDULER_REGISTRY[self.name](optimizer, **self.kwargs)

    def lr_lightning_dict(self, optimizer: torch.optim.Optimizer) -> Dict[str, Any]:
        """Return a Lightning-compatible scheduler configuration dict.

        Calls :meth:`build` internally and wraps the result with the
        Lightning-specific keys (``interval``, ``frequency``, optionally
        ``monitor``).

        Args:
            optimizer: The optimizer to wrap.

        Returns:
            Dict suitable for inclusion in ``configure_optimizers`` return value.
        """
        scheduler = self.build(optimizer)
        d: Dict[str, Any] = {
            "scheduler": scheduler,
            "interval": self.interval,
            "frequency": self.frequency,
        }
        cls = SCHEDULER_REGISTRY[self.name]
        if issubclass(cls, torch.optim.lr_scheduler.ReduceLROnPlateau):
            d["monitor"] = self.monitor
        return d

    @classmethod
    def from_config(cls, config: Union[DictConfig, Dict[str, Any]]) -> Optional["SchedulerFactory"]:
        """Construct from a config (DictConfig or plain dict).

        Reads ``config["callbacks"]["lr_scheduler"]``.  Returns ``None``
        when that key is absent so callers can skip scheduler setup.

        Args:
            config: Top-level configuration (OmegaConf DictConfig or dict).

        Returns:
            A :class:`SchedulerFactory`, or ``None`` if no scheduler is
            configured.
        """
        if not isinstance(config, DictConfig):
            config = OmegaConf.create(config)
        if OmegaConf.select(config, "lr_scheduler") is None:
            return None
        args = build_from_config(
            config,
            path="lr_scheduler",
            own_keys=_LIGHTNING_SCHED_KEYS,
            defaults={"name": "ReduceLROnPlateau", "monitor": "val/loss_epoch", "interval": "epoch", "frequency": 1},
        )
        args["frequency"] = int(args["frequency"])
        return cls(**args)
