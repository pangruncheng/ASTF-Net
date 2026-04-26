"""Optimizer factory backed by a dataclass, bridging YAML config to PyTorch."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Union

import torch
from omegaconf import DictConfig, OmegaConf

from astfnet.models._factory_utils import build_from_config

OPTIMIZER_REGISTRY: Dict[str, Any] = {
    "Adam": torch.optim.Adam,
    "AdamW": torch.optim.AdamW,
}

_OPT_OWN_KEYS = {"name", "lr"}


@dataclass
class OptimizerFactory:
    """Factory that instantiates a :class:`torch.optim.Optimizer`.

    Configure via ``optimizer:`` block in YAML::

        optimizer:
          name: Adam        # one of OPTIMIZER_REGISTRY keys
          lr: 0.0001
          weight_decay: 0.0 # any extra kwarg forwarded to the constructor

    ``lr`` may also be specified at the top level of the config for
    backward compatibility (``optimizer.lr`` takes precedence when present).

    Attributes:
        name: Key in :data:`OPTIMIZER_REGISTRY`.
        lr: Base learning rate.
        kwargs: Extra keyword arguments forwarded verbatim to the optimizer
            constructor (e.g. ``weight_decay``, ``betas``, ``momentum``).
    """

    name: str = "Adam"
    lr: float = 1e-3
    kwargs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the config values."""
        if self.name not in OPTIMIZER_REGISTRY:
            raise ValueError(f"Unknown optimizer {self.name!r}. Supported: {list(OPTIMIZER_REGISTRY)}.")
        if self.lr <= 0:
            raise ValueError(f"lr must be positive, got {self.lr!r}.")

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    def build(self, params: Iterable) -> torch.optim.Optimizer:
        """Instantiate the optimizer for *params*.

        Args:
            params: Iterable of parameters (e.g. ``model.parameters()``).

        Returns:
            A configured :class:`torch.optim.Optimizer` instance.

        Raises:
            ValueError: If :attr:`name` is not in :data:`OPTIMIZER_REGISTRY`.
        """
        return OPTIMIZER_REGISTRY[self.name](params, lr=self.lr, **self.kwargs)

    # ------------------------------------------------------------------
    # YAML bridge
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: Union[DictConfig, Dict[str, Any]]) -> "OptimizerFactory":
        """Construct from a config (DictConfig or plain dict).

        Reads the ``optimizer`` sub-dict when present; falls back to the
        top-level ``lr`` key for backward compatibility.

        Args:
            config: Top-level configuration (OmegaConf DictConfig or dict).

        Returns:
            An :class:`OptimizerFactory` populated from *config*.
        """
        if not isinstance(config, DictConfig):
            config = OmegaConf.create(config)
        # top-level ``lr`` is the backward-compat fallback when optimizer.lr is absent
        fallback_lr = float(OmegaConf.select(config, "lr", default=1e-3))
        args = build_from_config(
            config,
            path="optimizer",
            own_keys=_OPT_OWN_KEYS,
            defaults={"name": "Adam", "lr": fallback_lr},
        )
        args["lr"] = float(args["lr"])
        return cls(**args)
