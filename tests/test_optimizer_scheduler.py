"""Unit tests for OptimizerFactory and SchedulerFactory."""

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from astfnet.models.optimizer import OptimizerFactory
from astfnet.models.scheduler import SchedulerFactory

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture()
def simple_params() -> torch.nn.ParameterList:
    """A tiny parameter list to pass to optimizer constructors."""
    return nn.Linear(4, 2).parameters()


@pytest.fixture()
def adam_optimizer(simple_params: torch.nn.ParameterList) -> torch.optim.Adam:
    return torch.optim.Adam(simple_params, lr=1e-3)


# ===========================================================================
# OptimizerFactory — dataclass construction
# ===========================================================================


class TestOptimizerFactoryInit:
    def test_defaults(self) -> None:
        factory = OptimizerFactory()
        assert factory.name == "Adam"
        assert factory.lr == pytest.approx(1e-3)
        assert factory.kwargs == {}

    def test_custom_values(self) -> None:
        factory = OptimizerFactory(name="AdamW", lr=5e-4, kwargs={"weight_decay": 1e-2})
        assert factory.name == "AdamW"
        assert factory.lr == pytest.approx(5e-4)
        assert factory.kwargs["weight_decay"] == pytest.approx(1e-2)

    def test_unknown_name_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown optimizer"):
            OptimizerFactory(name="SGD")

    def test_zero_lr_raises(self) -> None:
        with pytest.raises(ValueError, match="lr must be positive"):
            OptimizerFactory(lr=0.0)

    def test_negative_lr_raises(self) -> None:
        with pytest.raises(ValueError, match="lr must be positive"):
            OptimizerFactory(lr=-1e-3)


# ===========================================================================
# OptimizerFactory.build
# ===========================================================================


class TestOptimizerFactoryBuild:
    def test_build_adam(self, simple_params: torch.nn.parameter.Parameter) -> None:
        factory = OptimizerFactory(name="Adam", lr=1e-3)
        opt = factory.build(simple_params)
        assert isinstance(opt, torch.optim.Adam)
        assert opt.defaults["lr"] == pytest.approx(1e-3)

    def test_build_adamw(self) -> None:
        params = nn.Linear(4, 2).parameters()
        factory = OptimizerFactory(name="AdamW", lr=2e-4, kwargs={"weight_decay": 1e-4})
        opt = factory.build(params)
        assert isinstance(opt, torch.optim.AdamW)
        assert opt.defaults["lr"] == pytest.approx(2e-4)
        assert opt.defaults["weight_decay"] == pytest.approx(1e-4)


# ===========================================================================
# OptimizerFactory.from_config
# ===========================================================================


class TestOptimizerFactoryFromConfig:
    def test_plain_dict_minimal(self) -> None:
        cfg = {"optimizer": {"name": "Adam", "lr": 1e-3}}
        factory = OptimizerFactory.from_config(cfg)
        assert factory.name == "Adam"
        assert factory.lr == pytest.approx(1e-3)
        assert factory.kwargs == {}

    def test_plain_dict_with_extra_kwargs(self) -> None:
        cfg = {"optimizer": {"name": "AdamW", "lr": 3e-4, "weight_decay": 1e-2}}
        factory = OptimizerFactory.from_config(cfg)
        assert factory.name == "AdamW"
        assert factory.kwargs == {"weight_decay": pytest.approx(1e-2)}

    def test_omegaconf_config(self) -> None:
        cfg = OmegaConf.create({"optimizer": {"name": "Adam", "lr": 5e-4}})
        factory = OptimizerFactory.from_config(cfg)
        assert factory.name == "Adam"
        assert factory.lr == pytest.approx(5e-4)

    def test_fallback_top_level_lr(self) -> None:
        """When optimizer.lr is absent, top-level lr is used as fallback."""
        cfg = {"lr": 7e-5, "optimizer": {"name": "Adam"}}
        factory = OptimizerFactory.from_config(cfg)
        assert factory.lr == pytest.approx(7e-5)

    def test_optimizer_lr_takes_precedence_over_top_level(self) -> None:
        cfg = {"lr": 1e-1, "optimizer": {"name": "Adam", "lr": 1e-4}}
        factory = OptimizerFactory.from_config(cfg)
        assert factory.lr == pytest.approx(1e-4)

    def test_missing_optimizer_block_uses_defaults(self) -> None:
        factory = OptimizerFactory.from_config({})
        assert factory.name == "Adam"
        assert factory.lr == pytest.approx(1e-3)

    def test_from_config_build_roundtrip(self) -> None:
        cfg = {"optimizer": {"name": "AdamW", "lr": 1e-3, "weight_decay": 5e-4}}
        factory = OptimizerFactory.from_config(cfg)
        params = nn.Linear(4, 2).parameters()
        opt = factory.build(params)
        assert isinstance(opt, torch.optim.AdamW)


# ===========================================================================
# SchedulerFactory — dataclass construction
# ===========================================================================


class TestSchedulerFactoryInit:
    def test_defaults(self) -> None:
        factory = SchedulerFactory()
        assert factory.name == "ReduceLROnPlateau"
        assert factory.monitor == "val/loss_epoch"
        assert factory.interval == "epoch"
        assert factory.frequency == 1
        assert factory.kwargs == {}

    def test_custom_values(self) -> None:
        factory = SchedulerFactory(
            name="ReduceLROnPlateau",
            monitor="val/my_metric",
            interval="step",
            frequency=2,
            kwargs={"mode": "min", "factor": 0.5, "patience": 5},
        )
        assert factory.monitor == "val/my_metric"
        assert factory.interval == "step"
        assert factory.frequency == 2
        assert factory.kwargs["factor"] == pytest.approx(0.5)

    def test_unknown_name_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown lr_scheduler"):
            SchedulerFactory(name="CosineAnnealingLR")

    def test_invalid_interval_raises(self) -> None:
        with pytest.raises(ValueError, match="interval must be"):
            SchedulerFactory(interval="batch")

    def test_zero_frequency_raises(self) -> None:
        with pytest.raises(ValueError, match="frequency must be"):
            SchedulerFactory(frequency=0)

    def test_negative_frequency_raises(self) -> None:
        with pytest.raises(ValueError, match="frequency must be"):
            SchedulerFactory(frequency=-1)


# ===========================================================================
# SchedulerFactory.build
# ===========================================================================


class TestSchedulerFactoryBuild:
    def test_build_reduce_lr_on_plateau(self, adam_optimizer: torch.optim.Adam) -> None:
        factory = SchedulerFactory(kwargs={"mode": "min", "factor": 0.5, "patience": 5})
        sched = factory.build(adam_optimizer)
        assert isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau)
        assert sched.factor == pytest.approx(0.5)
        assert sched.patience == 5


# ===========================================================================
# SchedulerFactory.lr_lightning_dict
# ===========================================================================


class TestSchedulerFactoryLrLightningDict:
    def test_contains_required_keys(self, adam_optimizer: torch.optim.Adam) -> None:
        factory = SchedulerFactory(
            monitor="val/loss_epoch",
            interval="epoch",
            frequency=1,
            kwargs={"mode": "min", "factor": 0.5, "patience": 10},
        )
        d = factory.lr_lightning_dict(adam_optimizer)
        assert "scheduler" in d
        assert isinstance(d["scheduler"], torch.optim.lr_scheduler.ReduceLROnPlateau)
        assert d["interval"] == "epoch"
        assert d["frequency"] == 1
        assert d["monitor"] == "val/loss_epoch"

    def test_monitor_key_present_for_reduce_lr_on_plateau(self, adam_optimizer: torch.optim.Adam) -> None:
        factory = SchedulerFactory(monitor="custom_metric")
        d = factory.lr_lightning_dict(adam_optimizer)
        assert d["monitor"] == "custom_metric"


# ===========================================================================
# SchedulerFactory.from_config
# ===========================================================================


class TestSchedulerFactoryFromConfig:
    def test_returns_none_when_absent(self) -> None:
        assert SchedulerFactory.from_config({}) is None
        assert SchedulerFactory.from_config({"optimizer": {"name": "Adam", "lr": 1e-3}}) is None

    def test_plain_dict_minimal(self) -> None:
        cfg = {
            "lr_scheduler": {
                "name": "ReduceLROnPlateau",
                "monitor": "val/loss_epoch",
                "interval": "epoch",
                "frequency": 1,
                "mode": "min",
                "factor": 0.5,
                "patience": 10,
            }
        }
        factory = SchedulerFactory.from_config(cfg)
        assert factory is not None
        assert factory.name == "ReduceLROnPlateau"
        assert factory.monitor == "val/loss_epoch"
        assert factory.interval == "epoch"
        assert factory.frequency == 1
        assert factory.kwargs["mode"] == "min"
        assert factory.kwargs["factor"] == pytest.approx(0.5)
        assert factory.kwargs["patience"] == 10

    def test_omegaconf_config(self) -> None:
        cfg = OmegaConf.create(
            {
                "lr_scheduler": {
                    "name": "ReduceLROnPlateau",
                    "monitor": "val/loss_epoch",
                    "interval": "epoch",
                    "frequency": 1,
                }
            }
        )
        factory = SchedulerFactory.from_config(cfg)
        assert factory is not None
        assert factory.name == "ReduceLROnPlateau"

    def test_frequency_is_int(self) -> None:
        cfg = {"lr_scheduler": {"name": "ReduceLROnPlateau", "frequency": 2}}
        factory = SchedulerFactory.from_config(cfg)
        assert isinstance(factory.frequency, int)
        assert factory.frequency == 2

    def test_from_config_build_roundtrip(self, adam_optimizer: torch.optim.Adam) -> None:
        cfg = {
            "lr_scheduler": {
                "name": "ReduceLROnPlateau",
                "monitor": "val/loss_epoch",
                "interval": "epoch",
                "frequency": 1,
                "mode": "min",
                "factor": 0.1,
                "patience": 3,
            }
        }
        factory = SchedulerFactory.from_config(cfg)
        sched = factory.build(adam_optimizer)
        assert isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau)
        assert sched.factor == pytest.approx(0.1)
