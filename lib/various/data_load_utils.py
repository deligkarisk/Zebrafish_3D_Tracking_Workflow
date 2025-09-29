import os
from typing import List, Dict, Tuple, Any

import h5py
import numpy as np


def load_h5_data(filepaths: List[str]) -> Tuple[List[np.ndarray], List[str]]:
    """
    Load tracking data from multiple H5 files.

    Args:
        filepaths: List of paths to H5 files containing tracking data

    Returns:
        Tuple containing:
            - List of location arrays from each file
            - List of node names (from the last file processed, assuming all files have the same node names)

    Note:
        This function assumes all H5 files have the same node names structure.
        Only the node names from the last processed file will be returned.
    """
    locations_all = []
    node_names = []

    for filepath in filepaths:
        with h5py.File(filepath, 'r') as f:
            locations = f["tracks"][:].T
            locations_all.append(locations)
            # Update node names (assumes all files have the same node names)
            node_names = [n.decode() for n in f["node_names"][:]]

    return locations_all, node_names


def load_h5_stack(folder: str, base_filename: str, num_of_stacks: int, 
                 start_frame: int, end_frame: int) -> Dict[str, np.ndarray]:
    """
    Load and process tracking data from multiple H5 files organized in stacks.

    Args:
        folder: Directory containing the H5 files
        base_filename: Base name of the H5 files
        num_of_stacks: Number of stack files to process
        start_frame: First frame to include
        end_frame: Last frame to include (exclusive)

    Returns:
        Dictionary of locations, with camera label as keys. 
        Values are ndarrays with shape (frames, nodes, coordinates, instances)
    """

    # Calculate frames per stack by reading the first file from the xy camera
    frames_per_stack = 0
    sample_file_path = os.path.join(folder, "xy", f"{base_filename}_1.slp.h5")
    with h5py.File(sample_file_path, 'r') as f:
        sample_locations = f["tracks"][:].T
        frames_per_stack = len(sample_locations) - start_frame

    num_frames_needed = end_frame - start_frame

    # Load data from all cameras
    camera_locations = {}
    cameras = ["xz", "xy", "yz"]

    for camera in cameras:
        camera_dir_path = os.path.join(folder, camera)
        camera_stack_locations = []

        for stack_idx in range(1, num_of_stacks + 1):
            print(f"Reading {camera} camera, stack {stack_idx}/{num_of_stacks}")

            stack_file_path = os.path.join(camera_dir_path, f"{base_filename}_{stack_idx}.slp.h5")
            with h5py.File(stack_file_path, 'r') as f:
                stack_locations = f["tracks"][:].T

                # Calculate valid frame indices for this stack
                start_valid_idx = (stack_idx - 1) * frames_per_stack + start_frame
                end_valid_idx = start_valid_idx + frames_per_stack
                valid_locations = stack_locations[start_valid_idx:end_valid_idx]

                # Extract node names (not used in this function, but kept for potential future use)
                # Note: Only the last processed file's node names would be available if needed
                node_names = [n.decode() for n in f["node_names"][:]]

                # Handle the case where only one track is identified
                # This adds a placeholder track with NaN values to maintain consistent dimensions
                if valid_locations.shape[3] == 1:
                    print(f"  Fixing shape for {camera} camera, stack {stack_idx} (adding placeholder track)")
                    nan_placeholder = np.ones(
                        (valid_locations.shape[0], valid_locations.shape[1], 
                         valid_locations.shape[2], 1)
                    ) * np.nan
                    valid_locations = np.concatenate((valid_locations, nan_placeholder), axis=3)

                camera_stack_locations.append(valid_locations)

        # Combine all stacks for this camera and trim to required frame count
        combined_camera_locations = np.concatenate(camera_stack_locations)
        trimmed_camera_locations = combined_camera_locations[:num_frames_needed]
        camera_locations[camera] = trimmed_camera_locations

    print("Successfully loaded all camera data")
    return camera_locations
