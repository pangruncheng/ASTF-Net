"""Dataset classes for ASTF-net."""

import os
import sys
from typing import Any, Dict

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from astfnet.data_io.augmentation import load_augmenter


class SeismicDatasetHDF5(Dataset):
    """Dataset class for loading seismic data from HDF5 files.

    Supports optional data augmentation using SpeechBrain's Augmenter.
    """

    _cached_augment_params = None

    def __init__(
        self: "SeismicDatasetHDF5",
        hdf5_file: str,
        augmentation_params: Dict[str, Any] = None,
        log_normalize_astf: bool = True,
        log_normalize_input: bool = True,
    ) -> None:
        """Initialize the dataset.

        Args:
            hdf5_file (str): Path to HDF5 file containing seismic data.
            augmentation_params (Dict[str, Any], optional): Augmentation config.
            log_normalize_astf (bool): Whether to log-normalize ASTF.
            log_normalize_input (bool): Whether to log-normalize input waveforms.
        """
        self.hdf5_file = hdf5_file

        with h5py.File(self.hdf5_file, "r") as hf:
            self.target_waveforms = hf["target_waveforms"][:]
            self.egfs = hf["egfs"][:]
            self.astfs = hf["astfs"][:]

        self._augmentation_params = augmentation_params
        self.augmenter = None
        self._has_announced_augmentation = False

        self.log_normalize_astf = log_normalize_astf
        self.log_normalize_input = log_normalize_input
        self.epsilon = 1.0

        print("🔥 DEBUG augmentation_params:", augmentation_params)

    def _ensure_augmenter(self: "SeismicDatasetHDF5") -> None:
        """Lazy-create augmenter (only once per worker, DDP-safe)."""
        if self.augmenter is None and self._augmentation_params:
            print(f"🧩 [Worker PID={os.getpid()}] Initializing augmenter...", flush=True)
            self.augmenter = load_augmenter(self._augmentation_params)
            sys.stdout.flush()

    def log_normalize(self: "SeismicDatasetHDF5", x: torch.Tensor) -> torch.Tensor:
        """Signed log normalization."""
        return torch.sign(x) * torch.log(torch.abs(x) + self.epsilon)

    def __len__(self: "SeismicDatasetHDF5") -> int:
        """Return dataset length."""
        return len(self.target_waveforms)

    def __getitem__(self: "SeismicDatasetHDF5", idx: int) -> Dict[str, Any]:
        """Get a sample from the dataset.

        Args:
            idx (int): Sample index.

        Returns:
            Dict[str, Any]: Dict containing target, egf, and astf tensors.
        """
        self._ensure_augmenter()

        target_waveform = torch.tensor(self.target_waveforms[idx], dtype=torch.float32).unsqueeze(0)
        egf = torch.tensor(self.egfs[idx], dtype=torch.float32).unsqueeze(0)
        astf = self.astfs[idx]

        # normalize inputs
        if self.log_normalize_input:
            target_waveform = self.log_normalize(target_waveform)
            egf = self.log_normalize(egf)

        # normalize ASTF
        if self.log_normalize_astf:
            astf = np.sign(astf) * np.log(np.abs(astf) + self.epsilon)

        astf = torch.tensor(astf, dtype=torch.float32)

        # augmentation
        if self.augmenter is not None:
            if not self._has_announced_augmentation:
                print(f"✅ Augmentation active for worker PID={os.getpid()}", flush=True)
                self._has_announced_augmentation = True

            target_waveform, _ = self.augmenter(target_waveform, lengths=torch.tensor([1.0]))
            egf, _ = self.augmenter(egf, lengths=torch.tensor([1.0]))

        return {
            "target": target_waveform.squeeze(0),
            "egf": egf.squeeze(0),
            "astf": astf,
        }


class SeismicDatasetHDF5_mask(Dataset):
    """Dataset class for loading seismic data with additional mask outputs.

    Supports data augmentation and log normalization for waveforms and ASTF.
    """

    def __init__(
        self: "SeismicDatasetHDF5_mask",
        hdf5_file: str,
        augmentation_params: Dict[str, Any] = None,
        log_normalize_astf: bool = True,
        log_normalize_input: bool = True,
    ) -> None:
        """Initialize dataset with masks.

        Args:
            hdf5_file (str): Path to HDF5 file.
            augmentation_params (Dict[str, Any], optional): Augmentation config.
            log_normalize_astf (bool): Whether to log-normalize ASTF.
            log_normalize_input (bool): Whether to log-normalize input waveforms.
        """
        self.hdf5_file = hdf5_file

        with h5py.File(self.hdf5_file, "r") as hf:
            self.target_waveforms = hf["target_waveforms"][:]
            self.egfs = hf["egfs"][:]
            self.astfs = hf["astfs"][:]
            self.masks = hf["masks"][:]

        self.augmenter = load_augmenter(augmentation_params or {})
        self.log_normalize_astf = log_normalize_astf
        self.log_normalize_input = log_normalize_input
        self.epsilon = 1.0

    def log_normalize(self: "SeismicDatasetHDF5_mask", x: torch.Tensor) -> torch.Tensor:
        """Signed log normalization."""
        return torch.sign(x) * torch.log(torch.abs(x) + self.epsilon)

    def __len__(self: "SeismicDatasetHDF5_mask") -> int:
        """Return dataset size."""
        return len(self.target_waveforms)

    def __getitem__(self: "SeismicDatasetHDF5_mask", idx: int) -> Dict[str, Any]:
        """Retrieve a sample with mask.

        Args:
            idx (int): Sample index.

        Returns:
            Dict[str, Any]: target, egf, astf, mask
        """
        target_waveform = torch.tensor(self.target_waveforms[idx], dtype=torch.float32)
        egf = torch.tensor(self.egfs[idx], dtype=torch.float32)
        mask = torch.tensor(self.masks[idx], dtype=torch.float32)

        astf = self.astfs[idx]

        if self.log_normalize_input:
            target_waveform = self.log_normalize(target_waveform)
            egf = self.log_normalize(egf)

        if self.log_normalize_astf:
            astf = np.sign(astf) * np.log(np.abs(astf) + self.epsilon)

        astf = torch.tensor(astf, dtype=torch.float32)

        # augment only waveform
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
