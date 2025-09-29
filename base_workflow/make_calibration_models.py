"""
Calibration Model Generator

This script creates calibration models from input datasets for 3D tracking.

Usage:
    1. Set the measured_coord_file path to your calibration board data
    2. Set the base_recordings_path to your experiments location
    3. Add your experiment IDs to the experiments_folders list
    4. Run the script

The script will:
    - Read input calibration data from the specified file
    - Process image and measured coordinates
    - Use polynomial fitting for calibration
    - Export calibration models and saved data to an output folder

For each experiment, the script creates four calibration models:
    1. xz_xy_to_yz.joblib: Predicts YZ coordinates from XZ and XY coordinates
    2. xy_yz_to_xz.joblib: Predicts XZ coordinates from XY and YZ coordinates
    3. xz_yz_to_xy.joblib: Predicts XY coordinates from XZ and YZ coordinates
    4. imCoords_to_XYZ.joblib: Predicts 3D coordinates from all camera views
"""

import os
import numpy as np
from lib.calibration.helper_functions import get_measured_coordinates, get_image_coordinates, \
    filter_data_with_balls_mask, prepare_image_data, print_shapes, get_mask_for_balls_position
from lib.calibration.model_utils import examine_residuals, fit_and_save_model, compute_residuals, \
    compute_residuals_im_to_real, get_model_parameters
from lib.various.filesystem import find_calibration_folder


measured_coord_file = '../physical_coord_newersetup.xlsx' # this sample file already exists. Replace with your specific calibration board
base_recordings_path = 'COMPLETED BY USER' # The main location/folder of the experiments e.g., "/home/experiments"
experiments_folders = ["EXPERIMENT_ID"] # a list of experiment_ids, e.g., ["Zebrafish20250421_1017-pc2"]

pos_names = ['pos1', 'pos2', 'pos3', 'pos4']
camera_names = ['xz', 'xy', 'yz']

for i in range(0, len(experiments_folders)):

    # Define respective locations for saving calibration models
    experiment_folder = os.path.join(base_recordings_path, experiments_folders[i])
    calibration_sub_folder_name = find_calibration_folder(experiment_folder)
    calibration_folder = os.path.join(experiment_folder, calibration_sub_folder_name, 'auto_calibration_results')
    save_model_data_folder = os.path.join(calibration_folder, 'python_calibration_models')
    os.makedirs(save_model_data_folder, exist_ok=True)


    # Load measured coordinates (real-world points)
    full_measured_coordinates = get_measured_coordinates(measured_coord_file=measured_coord_file,
                                                         position_names=pos_names)

    # Load image points (camera pixel coordinates)
    full_image_coordinates = get_image_coordinates(camera_names=camera_names, image_coord_folder=calibration_folder,
                                                   position_names=pos_names)

    print_shapes(full_measured_coordinates, full_image_coordinates)

    total_balls = full_measured_coordinates.shape[0]
    balls_mask = get_mask_for_balls_position(total_balls, full_measured_coordinates, full_image_coordinates)

    measured_coordinates, image_coordinates = filter_data_with_balls_mask(
        full_measured_coordinates=full_measured_coordinates,
        full_image_coordinates=full_image_coordinates,
        balls_mask=balls_mask)


    # Transformation of arrays for subsequent processing
    paired_coordinates, individual_coordinates, flat_image_coordinates = prepare_image_data(image_coordinates,
                                                                                            measured_coordinates)
    xz_xy_image_coordinates = paired_coordinates['xzxy']
    xy_yz_image_coordinates = paired_coordinates['xyyz']
    xz_yz_image_coordinates = paired_coordinates['xzyz']
    xz_image_coordinates = individual_coordinates['xz']
    xy_image_coordinates = individual_coordinates['xy']
    yz_image_coordinates = individual_coordinates['yz']

    # XZ and XY to YZ
    x_data = np.copy(xz_xy_image_coordinates)
    y_data = np.copy(yz_image_coordinates)
    getter_GS = get_model_parameters(x_data=x_data, y_data=y_data)
    alpha, polyDeg = examine_residuals(x_data, y_data, getter_GS, save_fig_folder=save_model_data_folder,
                                       fig_filename='residuals_xzxy_to_yz.png')
    getter = fit_and_save_model(alpha, polyDeg, x_data, y_data, save_model_data_folder,
                                getter_name='xz_xy_to_yz.joblib')
    compute_residuals(getter, x_data, y_data, save_fig_folder=save_model_data_folder,
                      fig_filename='computed_residuals_xzxy_to_yz.png')

    # XY and YZ to XZ
    x_data = np.copy(xy_yz_image_coordinates)
    y_data = np.copy(xz_image_coordinates)
    getter_GS = get_model_parameters(x_data=x_data, y_data=y_data)
    alpha, polyDeg = examine_residuals(x_data, y_data, getter_GS, save_fig_folder=save_model_data_folder,
                                       fig_filename='residuals_xyyz_to_xz.png')
    getter = fit_and_save_model(alpha, polyDeg, x_data, y_data, save_model_data_folder,
                                getter_name='xy_yz_to_xz.joblib')
    compute_residuals(getter, x_data, y_data, save_fig_folder=save_model_data_folder,
                      fig_filename='computed_residuals_xyyz_to_xz.png')

    # XZ and YZ to XY
    x_data = np.copy(xz_yz_image_coordinates)
    y_data = np.copy(xy_image_coordinates)
    getter_GS = get_model_parameters(x_data=x_data, y_data=y_data)
    alpha, polyDeg = examine_residuals(x_data, y_data, getter_GS, save_fig_folder=save_model_data_folder,
                                       fig_filename='residuals_xzyz_to_xy.png')
    getter = fit_and_save_model(alpha, polyDeg, x_data, y_data, save_model_data_folder,
                                getter_name='xz_yz_to_xy.joblib')
    compute_residuals(getter, x_data, y_data, save_fig_folder=save_model_data_folder,
                      fig_filename='computed_residuals_xzyz_to_xy.png')

    # XZ and XY and YZ to XYZ
    x_data = np.copy(flat_image_coordinates)
    y_data = np.copy(measured_coordinates)
    getter_GS = get_model_parameters(x_data=x_data, y_data=y_data)
    alpha, polyDeg = examine_residuals(x_data, y_data, getter_GS, save_fig_folder=save_model_data_folder,
                                       fig_filename='residuals_xz_xy_yz_to_xyz')
    getter = fit_and_save_model(alpha, polyDeg, x_data, y_data, save_model_data_folder,
                                getter_name='imCoords_to_XYZ.joblib')
    compute_residuals_im_to_real(getter, x_data, y_data, save_fig_folder=save_model_data_folder,
                                 fig_filename='computed_residuals_xzxyyz_to_xyz')
