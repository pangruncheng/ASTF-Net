from typing import Any, Dict, Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from astfnet.data_io.dataset import SeismicDatasetHDF5


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
        self.augmentation_params = {"data_augmentations": config.get("data_augmentations", [])}

    def setup(self, stage: Optional[str] = None) -> None:
        """Set up the datasets for different stages (fit, test).

        Args:
            stage: Either 'fit', 'test', or None. If None, sets up all stages.
        """
        if stage == "fit" or stage is None:
            self.train_dataset = SeismicDatasetHDF5(self.train_hdf5_file, self.augmentation_params)
            self.val_dataset = SeismicDatasetHDF5(self.val_hdf5_file)
        if stage == "test" or stage is None:
            self.test_dataset = SeismicDatasetHDF5(self.test_hdf5_file)

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
