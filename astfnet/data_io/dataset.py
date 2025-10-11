"""Dataset classes for ASTF-net."""

from typing import Any, Dict

import h5py
import torch
import numpy as np
from torch.utils.data import Dataset

from astfnet.data_io.augmentation import load_augmenter


class SeismicDatasetHDF5(Dataset):
    """Dataset class for loading seismic data from HDF5 files with data augmentation using SpeechBrain's Augmenter."""

    def __init__(
        self,
        hdf5_file: str,
        augmentation_params: Dict[str, Any] = None,
        log_normalize_astf: bool = True,
        log_normalize_input: bool = True,
    ) -> None:
        """
        Args:
            hdf5_file: Path to HDF5 file containing seismic data
            augmentation_params: Dictionary of augmentation parameters
            log_normalize_astf: Whether to apply signed log normalization to ASTF
            log_normalize_input: Whether to apply signed log normalization to input waveforms
        """
        self.hdf5_file = hdf5_file
        with h5py.File(self.hdf5_file, "r") as hf:
            self.target_waveforms = hf["target_waveforms"][:]
            self.egfs = hf["egfs"][:]
            self.astfs = hf["astfs"][:]
        self.augmenter = load_augmenter(augmentation_params or {})
        self.log_normalize_astf = log_normalize_astf
        self.log_normalize_input = log_normalize_input
        self.epsilon = 1

    def log_normalize(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sign(x) * torch.log(torch.abs(x) + self.epsilon)

    def __len__(self) -> int:
        return len(self.target_waveforms)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        target_waveform = torch.tensor(self.target_waveforms[idx], dtype=torch.float32).unsqueeze(0)
        egf = torch.tensor(self.egfs[idx], dtype=torch.float32).unsqueeze(0)
        astf = self.astfs[idx]

        # Normalize inputs if specified
        if self.log_normalize_input:
            target_waveform = self.log_normalize(target_waveform)
            egf = self.log_normalize(egf)

        # Normalize ASTF if specified
        if self.log_normalize_astf:
            astf = np.sign(astf) * np.log(np.abs(astf) + self.epsilon)

        astf = torch.tensor(astf, dtype=torch.float32)

        # Data augmentation (only on waveform, not ASTF)
        if self.augmenter is not None:
            target_waveform, _ = self.augmenter(target_waveform, lengths=torch.tensor([1.0]))
            egf, _ = self.augmenter(egf, lengths=torch.tensor([1.0]))

        return {
            "target": target_waveform.squeeze(0),
            "egf": egf.squeeze(0),
            "astf": astf,
        }
    
class SeismicDatasetHDF5_mask(Dataset):
    """Dataset class for loading seismic data from HDF5 files with data augmentation using SpeechBrain's Augmenter."""

    def __init__(
        self, 
        hdf5_file: str, 
        augmentation_params: Dict[str, Any] = None,
        log_normalize_astf: bool = True,
        log_normalize_input: bool = True
    ) -> None:
        """Initialize the dataset.

        Args:
            hdf5_file: Path to HDF5 file containing seismic data
            augmentation_params: Dictionary of augmentation parameters
            log_normalize_astf: Whether to apply signed log normalization to ASTF
            log_normalize_input: Whether to apply signed log normalization to input waveforms
        """
        self.hdf5_file = hdf5_file
        with h5py.File(self.hdf5_file, "r") as hf:
            self.target_waveforms = hf["target_waveforms"][:]
            self.egfs = hf["egfs"][:]
            self.astfs = hf["astfs"][:]
            self.masks = hf["masks"][:]  # 读取 mask 数据
        self.augmenter = load_augmenter(augmentation_params or {})
        self.log_normalize_astf = log_normalize_astf
        self.log_normalize_input = log_normalize_input
        self.epsilon = 1

    def log_normalize(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sign(x) * torch.log(torch.abs(x) + self.epsilon)
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.target_waveforms)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a sample from the dataset.

        Args:
            idx: Index of the sample

        Returns:
            Dictionary containing target, egf, astf, and mask tensors
        """
        target_waveform = torch.tensor(self.target_waveforms[idx], dtype=torch.float32)
        egf = torch.tensor(self.egfs[idx], dtype=torch.float32)
        # astf = torch.tensor(self.astfs[idx], dtype=torch.float32)
        mask = torch.tensor(self.masks[idx], dtype=torch.float32)

        astf = self.astfs[idx]

        # Normalize inputs if specified
        if self.log_normalize_input:
            target_waveform = self.log_normalize(target_waveform)
            egf = self.log_normalize(egf)

        # Normalize ASTF if specified
        if self.log_normalize_astf:
            astf = np.sign(astf) * np.log(np.abs(astf) + self.epsilon)

        astf = torch.tensor(astf, dtype=torch.float32)

        # Apply augmentation (only to target & egf)
        if self.augmenter is not None:
            target_waveform_aug, _ = self.augmenter(target_waveform.unsqueeze(0), lengths=torch.tensor([1.0]))
            egf_aug, _ = self.augmenter(egf.unsqueeze(0), lengths=torch.tensor([1.0]))
            target_waveform = target_waveform_aug.squeeze(0)
            egf = egf_aug.squeeze(0)

        return {
            "target": target_waveform,
            "egf": egf,
            "astf": astf,
            "mask": mask,
        }
