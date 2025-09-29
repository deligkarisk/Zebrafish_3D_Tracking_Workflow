import numpy as np


def prepend_nan_to_results_array(results_array, num_frames):
    total_frames = num_frames + len(results_array)
    new_array = np.ones((total_frames, 2, 3, 3)) * np.nan
    new_array[num_frames:, :] = results_array
    return new_array


def prepend_nan_to_im_coords_array(im_coords_array, num_frames):
    total_frames = num_frames + im_coords_array.shape[1]
    new_array = np.ones((3, total_frames, 2, 3, 2)) * np.nan
    new_array[:, num_frames:, :, :, :] = im_coords_array
    return new_array


def prepend_nan_to_idtracker_data(idtracker_data, num_frames):
    total_frames = num_frames + len(idtracker_data)
    new_array = np.ones((total_frames, 2, 2)) * np.nan
    new_array[num_frames:, :] = idtracker_data
    return new_array

def prepend_nan_to_speed_smooth_data(speed_smooth_data, num_frames):
    total_frames = num_frames + len(speed_smooth_data)
    new_array = np.ones((total_frames, 2, 3)) * np.nan
    new_array[num_frames:, :] = speed_smooth_data
    return new_array