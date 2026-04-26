"""Data augmentation modules for ASTF-net."""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from speechbrain.augment.augmenter import Augmenter
from speechbrain.augment.time_domain import AddNoise, DropChunk

logger = logging.getLogger(__name__)


class BaseWaveformAugmentation(nn.Module, ABC):
    """Base class for waveform augmentations."""

    def __init__(self) -> None:
        """Initialize the class."""
        super().__init__()

    @abstractmethod
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Forward method."""


class AddRandomNoise(BaseWaveformAugmentation):
    """Add random Gaussian noise to a batch of waveform tensors."""

    def __init__(self, noise_level: float = 0.05) -> None:
        """Initialize AddRandomNoise.

        Args:
            noise_level (float): Standard deviation of the noise as a fraction of the max absolute value of the waveform.
        """
        super().__init__()
        self.noise_level = noise_level

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Add random noise to the batch of waveforms.

        Args:
            waveform: Input waveform tensor of shape (batch, seq_len).

        Returns:
            noisy_waveform: Noisy waveform tensor of shape (batch, seq_len).
        """
        max_abs = torch.amax(torch.abs(waveform), dim=1, keepdim=True)
        noise = torch.randn_like(waveform) * self.noise_level * max_abs
        noisy_waveform = waveform + noise
        return noisy_waveform


class DropRegion(BaseWaveformAugmentation):
    """Randomly zero out a region of each waveform in the batch."""

    def __init__(self, max_drop_length: int = 10) -> None:
        """Initialize DropRegion.

        Args:
            max_drop_length: Maximum length of the region to drop.
        """
        super().__init__()
        self.max_drop_length = max_drop_length

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Randomly zero out a region in each waveform in the batch.

        Args:
            waveform: Input waveform tensor of shape (batch, seq_len).

        Returns:
            dropped_waveform: waveform with dropped regions.
        """
        batch, seq_len = waveform.shape
        dropped_waveform = waveform.clone()
        for i in range(batch):
            drop_start = torch.randint(0, seq_len // 2, (1,)).item()
            drop_len = torch.randint(0, self.max_drop_length, (1,)).item()
            drop_end = min(drop_start + drop_len, seq_len)
            dropped_waveform[i, drop_start:drop_end] = 0

        return dropped_waveform


class AddTrend(BaseWaveformAugmentation):
    """Add a linear trend to each waveform in the batch."""

    def __init__(self, min_deg: float = -2.0, max_deg: float = 2.0) -> None:
        """Initialize AddTrend."""
        super().__init__()
        self.min_deg = min_deg
        self.max_deg = max_deg

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Add a linear trend to each waveform in the batch.

        Args:
            waveform: Input waveform tensor of shape (batch, seq_len).

        Returns:
            trended_waveform: waveform with trend.
        """
        batch, seq_len = waveform.shape
        trended_waveform = waveform.clone()
        x = torch.arange(seq_len, dtype=waveform.dtype, device=waveform.device)
        for i in range(batch):
            trend_deg = float((self.max_deg - self.min_deg) * torch.rand(1).item() + self.min_deg)
            slope = torch.tan(torch.tensor(trend_deg * torch.pi / 180.0))
            trend = slope * x
            max_abs = torch.max(torch.abs(waveform[i]))
            if max_abs > 0:
                trend = trend / (torch.max(torch.abs(trend)) + 1e-6) * max_abs
            trended_waveform[i] = waveform[i] + trend
        return trended_waveform


class RandomShift(BaseWaveformAugmentation):
    """Randomly shift the waveform in time (to the right) by up to max_shift samples."""

    def __init__(self: "RandomShift", max_shift: int = 10) -> None:
        """Initialize RandomShift."""
        super().__init__()
        self.max_shift = max_shift

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Randomly shift the waveform in time.

        Args:
            waveform: Input waveform tensor of shape (batch, seq_len).

        Returns:
            shifted_waveform: Time-shifted waveform with same shape.
        """
        batch_size, _ = waveform.shape
        shifted_waveform = torch.zeros_like(waveform)

        for i in range(batch_size):
            shift = torch.randint(0, self.max_shift + 1, (1,)).item()
            if shift == 0:
                shifted_waveform[i] = waveform[i]
            else:
                shifted_waveform[i, shift:] = waveform[i, :-shift]
                shifted_waveform[i, :shift] = 0  # pad left with zeros

        return shifted_waveform


def load_augmenter(augmentation_params: Dict[str, Any]) -> Optional[Augmenter]:
    """Load an augmenter from a dictionary of augmentation parameters."""
    augmentations = load_augmentations(augmentation_params)
    max_augmentations = int(augmentation_params.get("max_augmentations", 1))
    if not augmentations or len(augmentations) == 0:
        logger.warning("No augmentations found.")
        return None

    logger.info(f"Loaded {len(augmentations)} augmentations, max_augmentations={max_augmentations}")

    augmenter = Augmenter(
        augmentations=augmentations,
        max_augmentations=max_augmentations,
        shuffle_augmentations=True,
        concat_original=False,
    )
    return augmenter


def load_augmentations(augmentation_params: Dict[str, Any]) -> List[torch.nn.Module]:
    """Load a list of augmentation modules from a dictionary of parameters."""
    augmentations = []
    data_augmentations = augmentation_params.get("data_augmentations", augmentation_params.get("augmentations", []))

    for data_augmentation in data_augmentations:
        for aug_name, aug_params in data_augmentation.items():
            if aug_name == "AddNoise":
                augmentation = AddNoise(**aug_params)
            elif aug_name == "DropChunk":
                augmentation = DropChunk(**aug_params)
            elif aug_name == "AddRandomNoise":
                augmentation = AddRandomNoise(**aug_params)
            elif aug_name == "DropRegion":
                augmentation = DropRegion(**aug_params)
            elif aug_name == "AddTrend":
                augmentation = AddTrend(**aug_params)
            elif aug_name == "RandomShift":
                augmentation = RandomShift(**aug_params)
            else:
                raise ValueError(f"Unknown augmentation: {aug_name}")
            augmentations.append(augmentation)
    return augmentations


if __name__ == "__main__":
    augmentation_params = {
        "data_augmentations": [
            {"AddRandomNoise": {"noise_level": 0.5}},
            {"DropRegion": {"max_drop_length": 10}},
            {"AddTrend": {"min_deg": -2, "max_deg": 2}},
            {"AddNoise": {"snr_low": 10, "snr_high": 20}},
            {"DropChunk": {"drop_length_low": 5, "drop_length_high": 10}},
            {"RandomShift": {"max_shift": 10}},
        ]
    }
    augmenter = load_augmenter(augmentation_params)
    assert augmenter is not None, "Expected augmenter to be non-None"
    waveform = torch.randn(1, 100)
    augmented_waveform, _ = augmenter(waveform, lengths=torch.tensor([1.0]))
    print(augmented_waveform.shape)
