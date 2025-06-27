"""Dataset classes for ASTF-net."""

from typing import Any, Dict

import h5py
import torch
from torch.utils.data import Dataset

from astfnet.data_io.augmentation import load_augmenter


class SeismicDatasetHDF5(Dataset):
    """Dataset class for loading seismic data from HDF5 files with data augmentation using SpeechBrain's Augmenter."""

    def __init__(self, hdf5_file: str, augmentation_params: Dict[str, Any] = None) -> None:
        """Initialize the dataset.

        Args:
            hdf5_file: Path to HDF5 file containing seismic data
            augmentation_params: Dictionary of augmentation parameters
        """
        self.hdf5_file = hdf5_file
        with h5py.File(self.hdf5_file, "r") as hf:
            self.target_waveforms = hf["target_waveforms"][:]
            self.egfs = hf["egfs"][:]
            self.astfs = hf["astfs"][:]
        self.augmenter = load_augmenter(augmentation_params or {})

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.target_waveforms)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a sample from the dataset.

        Args:
            idx: Index of the sample

        Returns:
            Dictionary containing target, egf, astf tensors and metadata
        """
        target_waveform = torch.tensor(self.target_waveforms[idx], dtype=torch.float32).unsqueeze(0)
        egf = torch.tensor(self.egfs[idx], dtype=torch.float32).unsqueeze(0)
        astf = torch.tensor(self.astfs[idx], dtype=torch.float32)
        if self.augmenter is not None:
            target_waveform, _ = self.augmenter(target_waveform, lengths=torch.tensor([1.0]))
            egf, _ = self.augmenter(egf, lengths=torch.tensor([1.0]))
        return {
            "target": target_waveform.squeeze(0),
            "egf": egf.squeeze(0),
            "astf": astf,
        }
