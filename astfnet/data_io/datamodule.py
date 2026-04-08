"""DataModule for seismic dataset loading and augmentation in ASTF-net."""

import logging
from typing import Any, Dict, Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from astfnet.data_io.dataset import (
    SeismicDatasetHDF5,
    SeismicDatasetHDF5_mask,
)

logger = logging.getLogger(__name__)


class SeismicDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for seismic data loading and processing.

    It handles loading of seismic data from HDF5 files and provides DataLoader instances
    for training, validation, and testing.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the SeismicDataModule with configuration."""
        super().__init__()
        self.config = config
        self.batch_size = config.get("batch_size", 32)
        self.num_workers = config.get("num_workers", 2)
        self.train_hdf5_file = config.get("train_hdf5_file")
        self.val_hdf5_file = config.get("val_hdf5_file")
        self.test_hdf5_file = config.get("test_hdf5_file")
        self.log_normalize_astf = config.get("log_normalize_astf", True)
        self.log_normalize_input = config.get("log_normalize_input", True)
        self.augmentation_params = {
            "augmentations": config.get("augmentations", config.get("data_augmentations", [])),
            "max_augmentations": int(config.get("max_augmentations", 0)),
        }

        model_name = config.get("model_name", "").lower()

        if "mask" in model_name:
            self.dataset_class = SeismicDatasetHDF5_mask
        else:
            self.dataset_class = SeismicDatasetHDF5

    def setup(self, stage: Optional[str] = None, test_name: Optional[str] = None) -> None:
        """Set up training, validation, or test datasets based on the stage.

        Args:
            stage: One of {"fit", "validate", "test"} or None.
            test_name: Optional key for selecting a specific test dataset.
        """
        if stage == "fit" or stage is None:
            logger.info("Setting up training and validation datasets...")
            self.train_dataset = self.dataset_class(
                self.train_hdf5_file,
                augmentation_params=self.augmentation_params,
                log_normalize_astf=self.log_normalize_astf,
                log_normalize_input=self.log_normalize_input,
            )

            self.val_dataset = self.dataset_class(
                self.val_hdf5_file,
                augmentation_params=None,
                log_normalize_astf=self.log_normalize_astf,
                log_normalize_input=self.log_normalize_input,
            )

        elif stage == "validate":
            logger.info("Setting up validation dataset...")
            self.val_dataset = self.dataset_class(
                self.val_hdf5_file,
                augmentation_params=None,
                log_normalize_astf=self.log_normalize_astf,
                log_normalize_input=self.log_normalize_input,
            )

        elif stage == "test":
            logger.info("Setting up test dataset...")
            if test_name is None:
                test_name = self.test_name if hasattr(self, "test_name") else None

            test_file_key = f"{test_name}_hdf5_file" if test_name else "test_hdf5_file"
            test_file = self.config.get(test_file_key)

            if test_file is None:
                raise ValueError(f"Cannot find HDF5 path for test_name: '{test_name}' in config.")

            self.test_dataset = self.dataset_class(
                test_file,
                augmentation_params=None,
                log_normalize_astf=self.log_normalize_astf,
                log_normalize_input=self.log_normalize_input,
            )
            self.test_name = test_name

    def train_dataloader(self) -> DataLoader:
        """Create and return the training DataLoader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Create and return the validation DataLoader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        """Create and return the test DataLoader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )
