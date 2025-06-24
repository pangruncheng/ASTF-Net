import pytest
import numpy as np
import torch
import h5py
from torch.utils.data import DataLoader
from astfnet.data.dataset import SeismicDatasetHDF5
from astfnet.data.datamodule import SeismicDataModule


@pytest.fixture
def dummy_config(tmp_path):
    # Create a dummy HDF5 file for testing
    h5_path = tmp_path / "dummy.h5"
    n, length = 10, 100
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("target_waveforms", data=np.random.randn(n, length))
        f.create_dataset("egfs", data=np.random.randn(n, length))
        f.create_dataset("astfs", data=np.random.randn(n, length))

    config = {
        "train_hdf5_file": str(h5_path),
        "val_hdf5_file": str(h5_path),
        "test_hdf5_file": str(h5_path),
        "batch_size": 2,
        "num_workers": 0,
        "data_augmentations": [
            {"AddNoise": {"snr_low": 10, "snr_high": 20}},
            {"DropChunk": {"drop_length_low": 5, "drop_length_high": 10}},
        ],
    }
    return config


def test_seismic_dataset_with_augmentation(dummy_config):
    dataset = SeismicDatasetHDF5(
        dummy_config["train_hdf5_file"],
        {"data_augmentations": dummy_config["data_augmentations"]},
    )
    sample = dataset[0]
    assert "target" in sample and "egf" in sample and "astf" in sample
    assert isinstance(sample["target"], torch.Tensor)
    assert isinstance(sample["egf"], torch.Tensor)
    assert isinstance(sample["astf"], torch.Tensor)
    # Check that augmentation does not change shape
    assert sample["target"].shape == torch.Size([100])
    assert sample["egf"].shape == torch.Size([100])
    assert sample["astf"].shape == torch.Size([100])


def test_seismic_dataset_without_augmentation(dummy_config):
    dataset = SeismicDatasetHDF5(dummy_config["val_hdf5_file"])
    sample = dataset[0]
    assert "target" in sample and "egf" in sample and "astf" in sample
    assert isinstance(sample["target"], torch.Tensor)
    assert isinstance(sample["egf"], torch.Tensor)
    assert isinstance(sample["astf"], torch.Tensor)
    assert sample["target"].shape == torch.Size([100])
    assert sample["egf"].shape == torch.Size([100])
    assert sample["astf"].shape == torch.Size([100])


def test_seismic_datamodule(dummy_config):
    dm = SeismicDataModule(dummy_config)
    dm.setup("fit")
    train_batch = next(iter(DataLoader(dm.train_dataset, batch_size=2)))
    val_batch = next(iter(DataLoader(dm.val_dataset, batch_size=2)))
    assert "target" in train_batch and "egf" in train_batch and "astf" in train_batch
    assert "target" in val_batch and "egf" in val_batch and "astf" in val_batch
    # Check that train and val batches have the same shape
    assert train_batch["target"].shape == torch.Size([2, 100])
    assert val_batch["target"].shape == torch.Size([2, 100])


def test_dataloader(dummy_config):
    dataset = SeismicDatasetHDF5(dummy_config["test_hdf5_file"])
    loader = DataLoader(dataset, batch_size=2)
    batch = next(iter(loader))
    assert "target" in batch and "egf" in batch and "astf" in batch
