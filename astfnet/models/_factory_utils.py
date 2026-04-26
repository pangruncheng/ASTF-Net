"""Utility functions for building model components from config."""

from typing import Any, Dict, Set, Union

from omegaconf import DictConfig, OmegaConf


def build_from_config(
    cfg: Union[DictConfig, Dict[str, Any]],
    path: str,
    own_keys: Set[str],
    defaults: Dict[str, Any],
) -> Dict[str, Any]:
    """Extract factory constructor args from a (possibly nested) config.

    Navigates *cfg* to the dot-separated *path*, applies *defaults* for
    each key in *own_keys*, and collects all remaining keys into a
    ``"kwargs"`` entry.

    Args:
        cfg: Top-level config — either an :class:`~omegaconf.DictConfig`
            (from ``OmegaConf.load``) or a plain :class:`dict`.
        path: Dot-separated key path to the sub-config block
            (e.g. ``"optimizer"`` or ``"callbacks.lr_scheduler"``).
        own_keys: Keys consumed by the factory itself and *not* forwarded
            as extra kwargs.
        defaults: Default value for every key in *own_keys*.  Must cover
            all keys in *own_keys*.

    Returns:
        Dict mapping each key in *own_keys* to its resolved value, plus a
        ``"kwargs"`` key containing every remaining key/value pair.
    """
    if not isinstance(cfg, DictConfig):
        cfg = OmegaConf.create(cfg)

    sub = OmegaConf.select(cfg, path, default=OmegaConf.create({}))
    _raw = OmegaConf.to_container(sub, resolve=True) or {}
    flat: Dict[str, Any] = dict(_raw) if isinstance(_raw, dict) else {}

    result: Dict[str, Any] = {k: flat.get(k, defaults[k]) for k in own_keys}
    result["kwargs"] = {k: v for k, v in flat.items() if k not in own_keys}
    return result
