"""
SLEAP and idTracker Data Preprocessor

This script preprocesses tracking data from SLEAP and idTracker.ai for 3D zebrafish tracking.

Usage:
    python preprocess_sleap_idtracker_data.py <recording> <start_frame> <end_frame> <num_splits_inference> <idtracker_session_name> [--existing_inference_results PATH]

Arguments:
    recording: Label of the experiment
    start_frame: Start frame of the tracking
    end_frame: End frame of the tracking
    num_splits_inference: Number of splits for SLEAP results
    idtracker_session_name: Label of the idTracker session
    --existing_inference_results: Optional path to existing inference results

The script will:
    1. Load SLEAP tracking data from H5 files
    2. Load idTracker.ai data from NPY files
    3. Process and align both datasets
    4. Save the combined data to a new H5 file for further processing

Output:
    H5 file containing preprocessed SLEAP and idTracker data
"""

import argparse
import os
from pathlib import Path

import h5py
import numpy as np

from lib.idtracker.process_idtracker import load_and_process_idtracker_data
from lib.various.data_load_utils import load_h5_stack

# Constants for paths
EXPERIMENTS_BASE_PATH = "COMPLETED BY USER"
RESULTS_INFERENCE_PATH = "./results/inference/"
RESULTS_TRACKING_PATH = "./results/tracking"


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Preprocess SLEAP and idTracker data for 3D tracking")
    parser.add_argument("recording", help="Label of the experiment")
    parser.add_argument("start_frame", type=int, help="Start frame of the tracking")
    parser.add_argument("end_frame", type=int, help="End frame of the tracking")
    parser.add_argument("num_splits_inference", type=int, help="Number of splits for SLEAP results")
    parser.add_argument("idtracker_session_name", help="Label of the idTracker session")
    parser.add_argument("--existing_inference_results", default=None, 
                        help="Path to existing inference results (optional)")
    return parser.parse_args()


def setup_paths(args):
    """Set up file and directory paths."""
    # Set up predictions folder
    if args.existing_inference_results:
        main_predictions_folder = args.existing_inference_results
    else:
        main_predictions_folder = RESULTS_INFERENCE_PATH

    # Set up recordings and results folders
    main_recordings_folder = os.path.join(EXPERIMENTS_BASE_PATH, args.recording)
    results_path = RESULTS_TRACKING_PATH
    Path(results_path).mkdir(parents=True, exist_ok=True)

    # Set up idTracker data path
    idtracker_data_path = os.path.join(
        main_recordings_folder, 
        "idtrackerai",
        "session_base", 
        args.idtracker_session_name, 
        "trajectories", 
        "with_gaps.npy"
    )

    return main_predictions_folder, main_recordings_folder, results_path, idtracker_data_path


def load_sleap_data(predictions_folder, num_splits, start_frame, end_frame):
    """Load and process SLEAP tracking data."""
    # Load raw SLEAP data
    locations = load_h5_stack(
        predictions_folder, 
        "predictions", 
        num_splits, 
        start_frame, 
        end_frame
    )

    # Stack camera views
    locations = np.stack((locations["xz"], locations["xy"], locations["yz"]))

    # Rearrange data to match expected format
    # SLEAP data dimensions: (cameras, frames, bodyparts, individuals, coordinates)
    sleap_data = np.transpose(locations, (0, 1, 4, 2, 3))

    return sleap_data


def load_idtracker_data(idtracker_path, sleap_data_length, start_frame, accel_thresh, nan_window):
    """Load and process idTracker.ai data."""
    # Load raw idTracker data
    trajectories_dict = np.load(idtracker_path, allow_pickle=True).item()
    idtracker_data = trajectories_dict["trajectories"]

    # Extract relevant frames from idTracker data
    # idTracker data contains all video frames, so we need to slice it to match SLEAP data
    idtracker_data = idtracker_data[start_frame: start_frame + sleap_data_length, :, :]

    # Process idTracker data (filter, clean, etc.)
    idtracker_data = load_and_process_idtracker_data(
        idtracker_data, 
        accel_thresh, 
        nan_window
    )

    return idtracker_data


def save_combined_data(sleap_data, idtracker_data, save_path):
    """Save processed SLEAP and idTracker data to H5 file."""
    save_file_path = os.path.join(save_path, "sleap_and_idtracker_data.h5")

    with h5py.File(save_file_path, "w") as hf:
        hf.create_dataset("sleap_data", data=sleap_data)
        hf.create_dataset("idtracker_data", data=idtracker_data)

    print(f"Data saved to {save_file_path}")


def main(idtracker_nan_window=6, idtracker_acceleration_thresh=3.3 ):
    """Main function to preprocess SLEAP and idTracker data.

    --- args ---
    idtracker_nan_window: Window size for NaN interpolation in idTracker data
    idtracker_acceleration_thresh: Threshold for acceleration-based filtering

    """
    # Parse command line arguments
    args = parse_arguments()

    # Set up paths
    predictions_folder, recordings_folder, results_path, idtracker_path = setup_paths(args)

    # Load and process SLEAP data
    print("Loading SLEAP data...")
    sleap_data = load_sleap_data(
        predictions_folder, 
        args.num_splits_inference, 
        args.start_frame, 
        args.end_frame
    )

    # Load and process idTracker data
    print("Loading idTracker data...")
    idtracker_data = load_idtracker_data(
        idtracker_path, 
        sleap_data.shape[1],  # Use SLEAP data length to determine how much idTracker data to use
        args.start_frame, 
        idtracker_acceleration_thresh, 
        idtracker_nan_window
    )

    # Save combined data
    save_combined_data(sleap_data, idtracker_data, results_path)


if __name__ == "__main__":
    main()
