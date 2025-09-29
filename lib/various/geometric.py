import numpy
import numpy as np
import scipy


def get_distances_from_locations_array(locations, bp_index):
    """
    This function calculates the fish distances in pixels from the locations array
    :param array: array in the shape of (cameras, frames, fish, bodyparts, coordinates)
    :param bp_index: body part index to use to calculate the distance
    """""

    num_cameras, num_frames, num_fish, num_bodyparts, _ = locations.shape
    distances = numpy.empty((num_cameras, num_frames))
    for i in range(num_cameras):
        for j in range(num_frames):
            fish_one_coordinates = locations[i, j, 0, bp_index]
            fish_two_coordinates = locations[i, j, 1, bp_index]
            if not (np.any(np.isnan(fish_one_coordinates)) or np.any(np.isnan(fish_two_coordinates))):
                distances[i, j] = scipy.spatial.distance.euclidean(fish_one_coordinates, fish_two_coordinates)
            else:
                distances[i,j] = np.nan
    return distances





