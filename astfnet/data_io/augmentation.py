from typing import Any, Dict, List

import torch
import torch.nn as nn
from speechbrain.augment.augmenter import Augmenter
from speechbrain.augment.time_domain import AddNoise, DropChunk


class AddRandomNoise(nn.Module):
    """Add random Gaussian noise to a batch of waveform tensors.

    Args:
        noise_level (float): Standard deviation of the noise as a fraction of the max absolute value of the waveform.

    Input shape:
        waveform: (batch, seq_len)
    Output shape:
        (batch, seq_len)
    """

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
            waveform (torch.Tensor): Input waveform tensor of shape (batch, seq_len).

        Returns:
            torch.Tensor: Noisy waveform tensor of shape (batch, seq_len).
        """
        max_abs = torch.amax(torch.abs(waveform), dim=1, keepdim=True)
        noise = torch.randn_like(waveform) * self.noise_level * max_abs
        return waveform + noise


class DropRegion(nn.Module):
    """Randomly zero out a region of each waveform in the batch.

    Args:
        max_drop_length (int): Maximum length of the region to drop.

    Input shape:
        waveform: (batch, seq_len)
    Output shape:
        (batch, seq_len)
    """

    def __init__(self, max_drop_length: int = 10) -> None:
        """Initialize DropRegion.

        Args:
            max_drop_length (int): Maximum length of the region to drop.
        """
        super().__init__()
        self.max_drop_length = max_drop_length

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Randomly zero out a region in each waveform in the batch.

        Args:
            waveform (torch.Tensor): Input waveform tensor of shape (batch, seq_len).

        Returns:
            torch.Tensor: waveform with dropped regions
        """
        batch, seq_len = waveform.shape
        dropped_waveform = waveform.clone()
        for i in range(batch):
            drop_start = torch.randint(0, seq_len // 2, (1,)).item()
            drop_len = torch.randint(0, self.max_drop_length, (1,)).item()
            drop_end = min(drop_start + drop_len, seq_len)
            dropped_waveform[i, drop_start:drop_end] = 0
        return dropped_waveform


class AddTrend(nn.Module):
    """Add a linear trend to each waveform in the batch.

    Args:
        min_deg (float): Minimum degree of the trend angle.
        max_deg (float): Maximum degree of the trend angle.

    Input shape:
        waveform: (batch, seq_len)
    Output shape:
        (batch, seq_len)
    """

    def __init__(self, min_deg: float = -2, max_deg: float = 2) -> None:
        """Initialize AddTrend.

        Args:
            min_deg (float): Minimum degree of the trend angle.
            max_deg (float): Maximum degree of the trend angle.
        """
        super().__init__()
        self.min_deg = min_deg
        self.max_deg = max_deg

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Add a linear trend to each waveform in the batch.

        Args:
            waveform (torch.Tensor): Input waveform tensor of shape (batch, seq_len).

        Returns:
            torch.Tensor: waveform with trend
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

class RandomShift(nn.Module):
    """
    Randomly shift the waveform in time (to the right) by up to max_shift samples.
    The shifted part is padded with zeros.

    Args:
        max_shift (int): Maximum number of samples to shift.
    """

    def __init__(self, max_shift: int = 10):
        super().__init__()
        self.max_shift = max_shift

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform (torch.Tensor): Input waveform tensor of shape (batch, seq_len)

        Returns:
            torch.Tensor: Time-shifted waveform with same shape
        """
        batch_size, seq_len = waveform.shape
        shifted_waveform = torch.zeros_like(waveform)

        for i in range(batch_size):
            shift = torch.randint(0, self.max_shift + 1, (1,)).item()
            if shift == 0:
                shifted_waveform[i] = waveform[i]
            else:
                shifted_waveform[i, shift:] = waveform[i, :-shift]
                shifted_waveform[i, :shift] = 0  # pad left with zeros

        return shifted_waveform


def load_augmenter(augmentation_params: Dict[str, Any]) -> Augmenter:
    """Load an augmenter from a dictionary of augmentation parameters.

    Args:
        augmentation_params (Dict[str, Any]): A dictionary of augmentation parameters.

    Returns:
        Augmenter: An augmenter object.
    """
    augmentations = load_augmentations(augmentation_params)
    augmenter_kwargs = augmentation_params.get("augmenter", {})
    augmenter = Augmenter(augmentations=augmentations, **augmenter_kwargs)
    return augmenter


def load_augmentations(
    augmentation_params: Dict[str, Any],
) -> List[torch.nn.Module]:
    """Load a list of augmentation modules from a dictionary of parameters.

    Args:
        augmentation_params (Dict[str, Any]): A dictionary of augmentation parameters.

    Returns:
        List[torch.nn.Module]: List of augmentation modules.
    """
    augmentations = []
    data_augmentations = augmentation_params.get("data_augmentations", [])
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
    waveform = torch.randn(1, 100)
    augmented_waveform, _ = augmenter(waveform, lengths=torch.tensor([1.0]))
    print(augmented_waveform.shape)
