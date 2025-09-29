import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt


def get_measured_coordinates(measured_coord_file, position_names):
    """
    Load measured coordinates (real-world points) from an Excel file.

    Args:
        measured_coord_file (str): Path to the Excel file containing measured coordinates
        position_names (list): List of sheet names in the Excel file to read

    Returns:
        numpy.ndarray: Array of measured coordinates with shape (numBalls, 3)
    """
    measured_coordinates = []
    for sheet_name in position_names:
        data = pd.read_excel(measured_coord_file, header=1, usecols=[0, 1, 2], sheet_name=sheet_name)
        data = data/10
        data_np = np.array(data)
        measured_coordinates.append(data_np)
    full_measured_coordinates = np.vstack(measured_coordinates)

    return full_measured_coordinates


def get_image_coordinates(camera_names, position_names, image_coord_folder):
    """
    Load image coordinates (camera pixel coordinates) from text files.

    Args:
        camera_names (list): List of camera names (e.g., ['xz', 'xy', 'yz'])
        position_names (list): List of position names (e.g., ['pos1', 'pos2', 'pos3', 'pos4'])
        image_coord_folder (str): Path to the folder containing image coordinate files

    Returns:
        numpy.ndarray: Array of image coordinates with shape (numCams, numBalls, 2)
    """
    coord_array = {}
    for camera in camera_names:
        df = pd.DataFrame()
        for position in position_names:
            position_folder = os.path.join(image_coord_folder, position)
            df_pos = pd.read_csv(os.path.join(position_folder, camera + '.txt'), sep=',', header=None)
            df = pd.concat([df, df_pos])
            coord_array[camera] = np.array(df)
    full_image_coordinates = np.stack([coord_array['xz'], coord_array['xy'], coord_array['yz']], axis=0)
    return full_image_coordinates


def print_shapes(full_measured_coordinates, full_image_coordinates):
    """
    Print the shapes of the measured and image coordinate arrays for debugging.

    Args:
        full_measured_coordinates (numpy.ndarray): Array of measured coordinates with shape (numBalls, 3)
        full_image_coordinates (numpy.ndarray): Array of image coordinates with shape (numCams, numBalls, 2)
    """
    print('-- Full Measured Coordinates --')
    print('(numBalls, 3D_coord)')
    print(full_measured_coordinates.shape)
    print()
    print('-- Full Image coordinates --')
    print('(numCams, numBalls, 2D_coord)')
    print(full_image_coordinates.shape)

    total_numBalls = full_measured_coordinates.shape[0]
    print('total number of calibration balls: ' + str(total_numBalls))


def get_mask_for_balls_position(total_numBalls, full_measured_coordinates, full_image_coordinates):
    """
    Create a boolean mask indicating which balls are inside the interior cage.

    Args:
        total_numBalls (int): Total number of calibration balls
        full_measured_coordinates (numpy.ndarray): Array of measured coordinates with shape (numBalls, 3)
        full_image_coordinates (numpy.ndarray): Array of image coordinates with shape (numCams, numBalls, 2)

    Returns:
        numpy.ndarray: Boolean mask with shape (numBalls,), True if the ball is inside the cage, False if outside
    """
    new_meas_list = []
    new_im_list = []
    within_mask = np.zeros((total_numBalls,), dtype=bool)

    for ballIdx in range(total_numBalls):
        if is_within_interior_cage(full_measured_coordinates[ballIdx]):
            new_meas_list.append(full_measured_coordinates[ballIdx])
            new_im_list.append(full_image_coordinates[:, ballIdx])
            within_mask[ballIdx] = True
    return within_mask

def is_within_interior_cage(xyz_coordinates, x_range=(5, 35), y_range=(5, 35), z_range=(0, 32)):
    """
    Check if 3D coordinates are within the interior cage boundaries.

    Args:
        xyz_coordinates (numpy.ndarray): 3D coordinates with shape (3,)
        x_range (tuple): Min and max values for X dimension (default: (5, 35))
        y_range (tuple): Min and max values for Y dimension (default: (5, 35))
        z_range (tuple): Min and max values for Z dimension (default: (0, 32))

    Returns:
        bool: True if coordinates are within the cage, False otherwise
    """
    # Define the X, Y and Z ranges of the interior cage
    interior_cage_lims = np.array([x_range, y_range, z_range])

    # Check each dimension, does component lie within the cage component limits?
    dim_successes = [
        dim_range[0] <= xyz_coordinates[i] <= dim_range[1]
        for i, dim_range in enumerate(interior_cage_lims)
    ]

    # If all 3 dimensions are within limits, return True
    return all(dim_successes)


def filter_data_with_balls_mask(full_measured_coordinates, full_image_coordinates, balls_mask):
    """
    Filter measured and image coordinates to only include points inside the interior cage.

    Args:
        full_measured_coordinates (numpy.ndarray): Array of measured coordinates with shape (numBalls, 3)
        full_image_coordinates (numpy.ndarray): Array of image coordinates with shape (numCams, numBalls, 2)
        balls_mask (numpy.ndarray): Boolean mask with shape (numBalls,), True if the ball is inside the cage

    Returns:
        tuple: (measured_coordinates, image_coordinates)
            - measured_coordinates (numpy.ndarray): Filtered measured coordinates
            - image_coordinates (numpy.ndarray): Filtered image coordinates
    """
    # Now define measured and image coordinates to only use interior points
    measured_coordinates = full_measured_coordinates[balls_mask]
    image_coordinates = full_image_coordinates[:, balls_mask]

    print(measured_coordinates.shape)
    print(image_coordinates.shape)

    numBalls = measured_coordinates.shape[0]
    print(numBalls)
    return measured_coordinates, image_coordinates


def prepare_image_data(image_coordinates, measured_coordinates):
    """
    Transform image coordinates into various formats needed for calibration.

    This function creates paired coordinates (e.g., xz+xy, xy+yz, xz+yz),
    individual camera coordinates, and flattened coordinates for model fitting.

    Args:
        image_coordinates (numpy.ndarray): Array of image coordinates with shape (numCams, numBalls, 2)
        measured_coordinates (numpy.ndarray): Array of measured coordinates with shape (numBalls, 3)

    Returns:
        tuple: (pair_coordinates, individual_coordinates, flat_image_coordinates)
            - pair_coordinates (dict): Dictionary of paired camera coordinates
            - individual_coordinates (dict): Dictionary of individual camera coordinates
            - flat_image_coordinates (numpy.ndarray): Flattened image coordinates with shape (numBalls, 6)
    """
    numBalls = image_coordinates.shape[1]
    xz_xy_image_coordinates = np.copy(image_coordinates[:2, :, :].swapaxes(0,1).reshape(numBalls, -1))
    xy_yz_image_coordinates = np.copy(image_coordinates[1:, :, :].swapaxes(0,1).reshape(numBalls, -1))
    xz_yz_image_coordinates = np.copy(image_coordinates[::2, :, :].swapaxes(0,1).reshape(numBalls, -1))

    xz_image_coordinates = np.copy(image_coordinates[0, :, :])
    xy_image_coordinates = np.copy(image_coordinates[1, :, :])
    yz_image_coordinates = np.copy(image_coordinates[2, :, :])

    # Transform from (numCams,numBalls,2) to (numBalls,6)
    flat_image_coordinates = np.copy(image_coordinates.swapaxes(0, 1).reshape(-1, 6))

    print('xzxy shape: ' + str(xz_xy_image_coordinates.shape))
    print('xyyz shape: ' + str(xy_yz_image_coordinates.shape))
    print('xzyz shape: ' + str(xz_yz_image_coordinates.shape))
    print()
    print('xz shape: ' + str(xz_image_coordinates.shape))
    print('zy shape: ' + str(xy_image_coordinates.shape))
    print('yz shape: ' + str(yz_image_coordinates.shape))
    print()
    print('measured coords shape: ' + str(measured_coordinates.shape))
    print()
    print('flat image coords shape: ' + str(flat_image_coordinates.shape))

    pair_coordinates = {'xzxy': xz_xy_image_coordinates, 'xyyz': xy_yz_image_coordinates, 'xzyz': xz_yz_image_coordinates}
    individual_coordinates = {'xz': xz_image_coordinates, 'xy': xy_image_coordinates, 'yz': yz_image_coordinates}

    return pair_coordinates, individual_coordinates, flat_image_coordinates
