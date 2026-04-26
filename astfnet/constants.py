"""Common data constants and path resolution for ASTF-net."""

import os
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Canonical file names
# ---------------------------------------------------------------------------

TRAIN_FILENAME = "Train_new_3_pairs_Samezero256_normalize_vr_ratio2_min0_2.h5"
VAL_FILENAME = "Validation_new_3_pairs_Samezero256_normalize_vr_ratio2_min0_2.h5"

TEST_SUBDIR = "test_set"
TEST_FILENAMES: List[str] = [
    "Test_level1_new_3_pairs_Samezero256_normalize_vr_1.h5",
    "Test_level2_new_3_pairs_Samezero256_normalize_vr_1.h5",
    "Test_level3_new_3_pairs_Samezero256_normalize_vr_1.h5",
]


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


def resolve_data_paths(data_path: str) -> Dict[str, Any]:
    """Derive canonical HDF5 paths from a single data directory.

    Args:
        data_path: Root directory containing train/val files and a
            ``test_set/`` subdirectory for test files.

    Returns:
        Dictionary with keys ``train_hdf5_file``, ``val_hdf5_file``, and
        ``test_hdf5_files`` (a list of paths to all test HDF5 files).
    """
    test_dir = os.path.join(data_path, TEST_SUBDIR)
    return {
        "train_hdf5_file": os.path.join(data_path, TRAIN_FILENAME),
        "val_hdf5_file": os.path.join(data_path, VAL_FILENAME),
        "test_hdf5_files": [os.path.join(test_dir, fname) for fname in TEST_FILENAMES],
    }
