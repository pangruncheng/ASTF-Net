import logging
import torch

from astfnet.models.loss_fns import (
    EffectiveRegionWeightedMSELoss,
    NonZeroWeightedMSE,
    WeightedMSE,
)

logger = logging.getLogger(__name__)


def load_loss(
    config: dict,
) -> torch.nn.Module:  # pragma: no cover
    r"""Instantiate the loss.

    Args:
        config: config dict from the config file

    Returns:
        loss_fn: The loss for the given model
    """
    loss_name = config["loss"]
    if loss_name == "weighted_mse":
        logger.info("WeightedMSE is loaded as the loss function.")
        loss_fn = WeightedMSE()
    elif loss_name == "effective_region_weighted_mse":
        logger.info("EffectiveRegionWeightedMSELoss is loaded as the loss function.")
        loss_fn = EffectiveRegionWeightedMSELoss()
    elif loss_name == "nonzero_weighted_mse":
        logger.info("NonZeroWeightedMSE is loaded as the loss function.")
        loss_fn = NonZeroWeightedMSE()
    else:
        raise ValueError("loss {} not supported".format(loss_name))
    return loss_fn
