"""
Dataset classes for ASTF-net.
"""

import h5py
import torch
from torch.utils.data import Dataset
from typing import Dict, Any, List
from speechbrain.augment.time_domain import AddNoise, DropChunk
from speechbrain.augment.augmenter import Augmenter


class SeismicDatasetHDF5(Dataset):
    """
    Dataset class for loading seismic data from HDF5 files with data augmentation using SpeechBrain's Augmenter.
    """

    def __init__(self, hdf5_file: str, augmentation_params: Dict[str, Any] = None):
        """
        Initialize the dataset.
        Args:
            hdf5_file: Path to HDF5 file containing seismic data
            augmentation_params: Dictionary of augmentation parameters
        """
        self.hdf5_file = hdf5_file
        with h5py.File(self.hdf5_file, "r") as hf:
            self.target_waveforms = hf["target_waveforms"][:]
            self.egfs = hf["egfs"][:]
            self.astfs = hf["astfs"][:]
        self.augmenter = self.load_augmenter(augmentation_params or {})

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.target_waveforms)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a sample from the dataset.

        Args:
            idx: Index of the sample

        Returns:
            Dictionary containing target, egf, astf tensors and metadata
        """
        target_waveform = torch.tensor(
            self.target_waveforms[idx], dtype=torch.float32
        ).unsqueeze(0)
        egf = torch.tensor(self.egfs[idx], dtype=torch.float32).unsqueeze(0)
        astf = torch.tensor(self.astfs[idx], dtype=torch.float32)
        if self.augmenter is not None:
            target_waveform, _ = self.augmenter(
                target_waveform, lengths=torch.tensor([1.0])
            )
            egf, _ = self.augmenter(egf, lengths=torch.tensor([1.0]))
        return {
            "target": target_waveform.squeeze(0),
            "egf": egf.squeeze(0),
            "astf": astf,
        }

    @staticmethod
    def load_augmenter(augmentation_params: Dict[str, Any]) -> Augmenter:
        augmentations = SeismicDatasetHDF5.load_augmentations(augmentation_params)
        augmenter_kwargs = augmentation_params.get("augmenter", {})
        if augmentations:
            return Augmenter(augmentations=augmentations, **augmenter_kwargs)
        return None

    @staticmethod
    def load_augmentations(
        augmentation_params: Dict[str, Any],
    ) -> List[torch.nn.Module]:
        augmentations = []
        data_augmentations = augmentation_params.get("data_augmentations", [])
        for data_augmentation in data_augmentations:
            for aug_name, aug_params in data_augmentation.items():
                if aug_name == "AddNoise":
                    augmentation = AddNoise(**aug_params)
                elif aug_name == "DropChunk":
                    augmentation = DropChunk(**aug_params)
                else:
                    raise ValueError(f"Unknown augmentation: {aug_name}")
                augmentations.append(augmentation)
        return augmentations
