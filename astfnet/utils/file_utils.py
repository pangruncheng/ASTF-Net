"""File utility functions for ASTF-net."""

import os
import random
import shutil
from typing import List


def extract_and_copy_files(
    z_folder: str, t_folder: str, lst_file_paths: List[str], output_folders: List[str]
) -> None:
    """Extract and copy files from Z and T folders according to a 2:1 ratio.

    Ensures no duplicates and copies to target folders.

    Args:
        z_folder: Path to Z component folder
        t_folder: Path to T component folder
        lst_file_paths: List of LST file paths
        output_folders: List of output folder paths
    """
    # Create target folders
    for output_folder in output_folders:
        os.makedirs(output_folder, exist_ok=True)

    # Store all selected file paths to ensure no duplicates
    all_selected_files = set()

    # Process 5 LST files
    for lst_file_path, output_folder in zip(lst_file_paths, output_folders):
        print(f"Processing LST file: {lst_file_path}")

        with open(lst_file_path, "r") as f:
            lines = f.readlines()

        # Extract Z and T component paths
        z_files = [line.strip() for line in lines if "Z" in line]
        t_files = [line.strip() for line in lines if "T" in line]

        # Calculate required P and S wave file counts
        num_z = len(z_files)
        num_t = len(t_files)

        # Number of P and S wave files to extract (2x)
        num_p = 2 * num_z
        num_s = 2 * num_t

        # Select non-duplicate files from z_files and t_files
        p_files = random.sample(z_files, num_p)
        s_files = random.sample(t_files, num_s)

        # Check for duplicates
        while not len(p_files) == len(set(p_files)) or not len(s_files) == len(
            set(s_files)
        ):
            print(f"Found duplicates, re-selecting files for {lst_file_path}")
            p_files = random.sample(z_files, num_p)
            s_files = random.sample(t_files, num_s)

        # Copy files to target folder
        selected_files = []
        for p_file in p_files + s_files:
            file_name = os.path.basename(p_file)
            destination_path = os.path.join(output_folder, file_name)

            # Ensure no duplicate files
            if file_name not in all_selected_files:
                shutil.copy(p_file, destination_path)
                selected_files.append(destination_path)
                all_selected_files.add(file_name)
            else:
                print(f"File {file_name} already copied, skipping.")

        # Generate new LST file (write copied file paths to LST)
        new_lst_file = os.path.join(output_folder, os.path.basename(lst_file_path))
        with open(new_lst_file, "w") as f_out:
            for file_path in selected_files:
                f_out.write(file_path + "\n")

        print(f"Generated new LST file: {new_lst_file}")


if __name__ == "__main__":
    # Example usage with default paths
    z_folder = "/media/admin123/Datastorage/data_files/data/data_files/Kiskatinaw_Events_local/ASTF_P_data/ASTF_P_data_Even_mag3to4.5_M0_randperm_SAC"
    t_folder = "/media/admin123/Datastorage/data_files/data/data_files/Kiskatinaw_Events_local/ASTF_S_data/ASTF_S_data_Even_mag3to4.5_M0_randperm_SAC"

    lst_file_paths = [
        "/media/admin123/Datastorage/data_files/dataset_ASTF-net/EGFs/EGFs_test_level1.lst",
        "/media/admin123/Datastorage/data_files/dataset_ASTF-net/EGFs/EGFs_test_level2.lst",
        "/media/admin123/Datastorage/data_files/dataset_ASTF-net/EGFs/EGFs_test_level3.lst",
        "/media/admin123/Datastorage/data_files/dataset_ASTF-net/EGFs/EGFs_train.lst",
        "/media/admin123/Datastorage/data_files/dataset_ASTF-net/EGFs/EGFs_validation.lst",
    ]

    output_folders = [
        "/media/admin123/Datastorage/data_files/dataset_ASTF-net/ASTFs/ASTFs_test_level1.lst",
        "/media/admin123/Datastorage/data_files/dataset_ASTF-net/ASTFs/ASTFs_test_level2.lst",
        "/media/admin123/Datastorage/data_files/dataset_ASTF-net/ASTFs/ASTFs_test_level3.lst",
        "/media/admin123/Datastorage/data_files/dataset_ASTF-net/ASTFs/ASTFs_train.lst",
        "/media/admin123/Datastorage/data_files/dataset_ASTF-net/ASTFs/ASTFs_validation.lst",
    ]

    # Execute function
    extract_and_copy_files(z_folder, t_folder, lst_file_paths, output_folders)
