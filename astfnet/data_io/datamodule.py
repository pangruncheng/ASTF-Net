from typing import Any, Dict, Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from astfnet.data_io.dataset import (
    SeismicDatasetHDF5,
    SeismicDatasetHDF5_mask,
    # SeismicDatasetHDF5_dual,       # input:(x1, x2)
    # SeismicDatasetHDF5_ampfusion,   # input:(x, amp)
)


class SeismicDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for seismic data loading and processing.

    Handles loading of seismic data from HDF5 files and provides DataLoader instances
    for training, validation, and testing.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the SeismicDataModule with configuration.

        Args:
            config: Configuration dictionary containing:
                - batch_size: Batch size for DataLoaders
                - num_workers: Number of workers for DataLoaders
                - train_hdf5_file: Path to training HDF5 file
                - val_hdf5_file: Path to validation HDF5 file
                - test_hdf5_file: Path to test HDF5 file
                - data_augmentations: List of data augmentation strategies
        """
        super().__init__()
        self.config = config
        self.batch_size = config.get("batch_size", 32)
        self.num_workers = config.get("num_workers", 2)
        self.train_hdf5_file = config.get("train_hdf5_file")
        self.val_hdf5_file = config.get("val_hdf5_file")
        self.test_hdf5_file = config.get("test_hdf5_file")
        self.log_normalize_astf = config.get("log_normalize_astf", True)
        self.log_normalize_input = config.get("log_normalize_input", True)
        self.augmentation_params = {"data_augmentations": config.get("data_augmentations", [])}
        
        model_name = config.get("model_name", "").lower()

        if "mask" in model_name:
            self.DatasetClass = SeismicDatasetHDF5_mask
        elif "lstm_fusion" in model_name:
            self.DatasetClass = SeismicDatasetHDF5_dual
        elif "amp_fusion" in model_name:
            self.DatasetClass = SeismicDatasetHDF5_ampfusion
        else:
            self.DatasetClass = SeismicDatasetHDF5

    def setup(self, stage: Optional[str] = None, test_name: Optional[str] = None):
        if stage == "fit" or stage is None:
            print("✅ Setting up training and validation datasets")
            self.train_dataset = self.DatasetClass(
                self.train_hdf5_file,
                augmentation_params=self.augmentation_params,
                log_normalize_astf=self.log_normalize_astf,
                log_normalize_input=self.log_normalize_input,
            )
            self.val_dataset = self.DatasetClass(
                self.val_hdf5_file,
                augmentation_params=None,
                log_normalize_astf=self.log_normalize_astf,
                log_normalize_input=self.log_normalize_input,
            )

        elif stage == "validate":
            print("✅ Setting up validation dataset")
            self.val_dataset = self.DatasetClass(
                self.val_hdf5_file,
                augmentation_params=None,
                log_normalize_astf=self.log_normalize_astf,
                log_normalize_input=self.log_normalize_input,
            )

        elif stage == "test":
            if test_name is None:
                test_name = self.test_name if hasattr(self, "test_name") else None

            test_file_key = f"{test_name}_hdf5_file" if test_name else "test_hdf5_file"
            test_file = self.config.get(test_file_key)

            if test_file is None:
                raise ValueError(f"❌ Cannot find HDF5 path for test_name: '{test_name}' in config.")

            print(f"✅ Setting up test dataset: {test_file}")
            self.test_dataset = self.DatasetClass(
                test_file,
                augmentation_params=None,
                log_normalize_astf=self.log_normalize_astf,
                log_normalize_input=self.log_normalize_input,
            )
            self.test_name = test_name

    def train_dataloader(self) -> DataLoader:
        """Create and return the training DataLoader.

        Returns:
            DataLoader configured for training data.
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Create and return the validation DataLoader.

        Returns:
            DataLoader configured for validation data.
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        """Create and return the test DataLoader.

        Returns:
            DataLoader configured for test data.
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )
