import pytorch_lightning as pl
from torch.utils.data import DataLoader
from astfnet.data_io.dataset import SeismicDatasetHDF5


class SeismicDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.batch_size = config.get("batch_size", 32)
        self.num_workers = config.get("num_workers", 2)
        self.train_hdf5_file = config.get("train_hdf5_file")
        self.val_hdf5_file = config.get("val_hdf5_file")
        self.test_hdf5_file = config.get("test_hdf5_file")
        self.augmentation_params = {
            "data_augmentations": config.get("data_augmentations", [])
        }

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = SeismicDatasetHDF5(
                self.train_hdf5_file, self.augmentation_params
            )
            self.val_dataset = SeismicDatasetHDF5(self.val_hdf5_file)
        if stage == "test" or stage is None:
            self.test_dataset = SeismicDatasetHDF5(
                self.test_hdf5_file, self.augmentation_params
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
