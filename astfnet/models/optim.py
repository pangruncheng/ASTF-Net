import logging

import torch

from astfnet.models.loss_fns import (
    AmplitudeWeightedMSELoss,
    ConvAlignLoss,
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
    elif loss_name == "amplitude_weighted_mse":
        logger.info("AmplitudeWeightedMSELoss is loaded as the loss function.")
        loss_fn = AmplitudeWeightedMSELoss(epsilon=1e-6, a=0.8)
    elif loss_name == "mse":
        logger.info("MSELoss is loaded as the loss function.")
        loss_fn = torch.nn.MSELoss()
    elif loss_name == "convalignLoss":
        logger.info("ConvAlignLoss is loaded as the loss function.")
        alpha = config.get("loss_weight")
        loss_fn = ConvAlignLoss(alpha=alpha, crop_len=256)
    else:
        raise ValueError("loss {} not supported".format(loss_name))
    return loss_fn
