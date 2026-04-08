"""Data preprocessing utilities for ASTF-net."""

import os
from typing import List, Tuple

import h5py
import numpy as np
import torch
from obspy import Trace, read

from astfnet.utils.seismic_utils import compute_M0, get_window_times, read_lst_file


def load_sac_file(sac_file: str) -> Trace:
    """Load SAC file and return waveform data.

    Args:
        sac_file: Path to SAC file.

    Returns:
        waveform: An ObsPy trace object.
    """
    waveform = read(sac_file)[0]
    return waveform


def pad_to_max_length(waveform_data: np.ndarray, max_length: int) -> np.ndarray:
    """Pad waveform data to maximum length.

    Args:
        waveform_data: Input waveform data
        max_length: Target maximum length

    Returns:
        Padded waveform data
    """
    if len(waveform_data) < max_length:
        padded_data = np.pad(waveform_data, (0, max_length - len(waveform_data)), mode="constant")
    else:
        padded_data = waveform_data
    return padded_data


def normalize_waveform(waveform: np.ndarray) -> Tuple[float, np.ndarray]:
    """Normalize waveform data.

    Args:
        waveform: Input waveform data

    Returns:
        Tuple of (normalization_coefficient, normalized_waveform)

    Raises:
        TypeError: If input is not a numpy array
    """
    if isinstance(waveform, np.ndarray):
        # Calculate maximum and minimum values
        max_value = np.max(waveform)
        min_value = np.min(waveform)

        # Calculate maximum amplitude, take absolute values of max and min
        normalization_coefficient = max(abs(max_value), abs(min_value))

        # Normalize
        norm_waveform = waveform / normalization_coefficient
        return normalization_coefficient, norm_waveform
    else:
        raise TypeError("Input waveform should be a numpy ndarray")


def load_data_pair(
    target_waveform_path: str, egf_path: str, astf_path: str
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load a pair of data, return Target, EGF, ASTF tensors.

    Args:
        target_waveform_path: Path to target waveform file
        egf_path: Path to EGF file
        astf_path: Path to ASTF file

    Returns:
        Tuple of (target_waveform_tensor, egf_tensor, astf_tensor)
    """
    # Load EGF and Target Waveform
    target_waveform = load_sac_file(target_waveform_path)
    egf = load_sac_file(egf_path)
    M0 = compute_M0(egf.stats.sac.mag)

    # Get time window (P-wave and S-wave arrival times)
    start_time, end_time = get_window_times(egf.stats.sac)

    # Extract data
    target_waveform_data = target_waveform.data
    egf_data = egf.data[start_time:end_time]

    # Get longer length
    max_length = max(len(target_waveform_data), len(egf_data))

    # Pad zeros
    egf_data = pad_to_max_length(egf_data, max_length)

    # Load ASTF
    astf_data = load_sac_file(astf_path).data  # Directly load ASTF data
    astf_data /= M0

    # Convert to PyTorch tensors
    target_waveform_tensor = torch.tensor(target_waveform_data, dtype=torch.float32)
    egf_tensor = torch.tensor(egf_data, dtype=torch.float32)
    astf_tensor = torch.tensor(astf_data, dtype=torch.float32)

    return target_waveform_tensor, egf_tensor, astf_tensor


def save_data_to_hdf5(lst_file: str, hdf5_filename: str, batch_size: int = 10000, compress: bool = True) -> None:
    """Read data paths from .lst file, load and save data as HDF5 format, optimized for batch writing.

    Args:
        lst_file: Path to LST file containing data paths
        hdf5_filename: Output HDF5 filename
        batch_size: Batch size for writing
        compress: Whether to compress the data
    """
    # Open HDF5 file and create datasets
    with h5py.File(hdf5_filename, "w") as hf:
        # Set compression options
        compression_opts = "gzip" if compress else None

        # Create dataset structure
        target_waveforms_ds = hf.create_dataset(
            "target_waveforms",
            shape=(0, 510),
            maxshape=(None, 510),
            dtype="float32",
            chunks=(batch_size, 510),
            compression=compression_opts,
        )
        egfs_ds = hf.create_dataset(
            "egfs",
            shape=(0, 510),
            maxshape=(None, 510),
            dtype="float32",
            chunks=(batch_size, 510),
            compression=compression_opts,
        )
        astfs_ds = hf.create_dataset(
            "astfs",
            shape=(0, 301),
            maxshape=(None, 301),
            dtype="float32",
            chunks=(batch_size, 301),
            compression=compression_opts,
        )

        # Read paths from .lst file
        data_pairs = read_lst_file(lst_file)

        # Cache data
        target_waveforms_batch = []
        egfs_batch = []
        astfs_batch = []

        for idx, (target_waveform_path, egf_path, astf_path) in enumerate(data_pairs):
            # Load data pair
            target_waveform_tensor, egf_tensor, astf_tensor = load_data_pair(target_waveform_path, egf_path, astf_path)

            # Convert data to NumPy arrays and add to batch
            target_waveforms_batch.append(target_waveform_tensor.numpy())
            egfs_batch.append(egf_tensor.numpy())
            astfs_batch.append(astf_tensor.numpy())

            # If current batch reaches specified size, write to HDF5 file
            if len(target_waveforms_batch) >= batch_size:
                target_waveforms_ds.resize(target_waveforms_ds.shape[0] + len(target_waveforms_batch), axis=0)
                egfs_ds.resize(egfs_ds.shape[0] + len(egfs_batch), axis=0)
                astfs_ds.resize(astfs_ds.shape[0] + len(astfs_batch), axis=0)

                target_waveforms_ds[-len(target_waveforms_batch) :] = np.array(target_waveforms_batch)
                egfs_ds[-len(egfs_batch) :] = np.array(egfs_batch)
                astfs_ds[-len(astfs_batch) :] = np.array(astfs_batch)

                # Clear batch cache
                target_waveforms_batch = []
                egfs_batch = []
                astfs_batch = []

            # Output progress every 50000 samples
            if (idx + 1) % 50000 == 0:
                print(f"Processed {idx + 1}/{len(data_pairs)} data pairs")

        # If there are remaining data (less than one batch), write to HDF5 file
        if target_waveforms_batch:
            target_waveforms_ds.resize(target_waveforms_ds.shape[0] + len(target_waveforms_batch), axis=0)
            egfs_ds.resize(egfs_ds.shape[0] + len(egfs_batch), axis=0)
            astfs_ds.resize(astfs_ds.shape[0] + len(astfs_batch), axis=0)

            target_waveforms_ds[-len(target_waveforms_batch) :] = np.array(target_waveforms_batch)
            egfs_ds[-len(egfs_batch) :] = np.array(egfs_batch)
            astfs_ds[-len(astfs_batch) :] = np.array(astfs_batch)

        print(f"Data saved to HDF5 file: {hdf5_filename}")


def preprocess_seismic_data(
    input_paths: List[str],
    output_path: str,
    batch_size: int = 10000,
    compress: bool = True,
) -> None:
    """Preprocess seismic data from multiple LST files and save to HDF5.

    Args:
        input_paths: List of LST file paths
        output_path: Output HDF5 file path
        batch_size: Batch size for processing
        compress: Whether to compress the output
    """
    # Create temporary combined LST file
    temp_lst_file = "temp_combined.lst"

    with open(temp_lst_file, "w") as f:
        for lst_file in input_paths:
            with open(lst_file, "r") as input_f:
                f.write(input_f.read())

    try:
        save_data_to_hdf5(temp_lst_file, output_path, batch_size, compress)
    finally:
        # Clean up temporary file
        if os.path.exists(temp_lst_file):
            os.remove(temp_lst_file)
