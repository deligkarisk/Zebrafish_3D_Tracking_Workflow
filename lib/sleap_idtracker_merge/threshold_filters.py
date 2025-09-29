import time

import numpy as np

from lib.registration.skeleton_registration_costs import get_mean_body_point_cost


def apply_reg_threshold_to_positions(registration_costs, positions_3d, positions_image_coordinates_processed,
                                     positions_3d_processed,
                                     reg_threshold):
    # Apply the reg_thresh to positions

    # Any registrations made by method2 already pass the theshold,
    # But we might have method1 results that do not pass.
    # We remove these skeletons

    t0 = time.time()
    num_fish = 2

    # find the registration costs averaged along bodypoints
    registration_costs_meanBp = get_mean_body_point_cost(registration_costs, num_fish)

    # find a mask, which shows 'True' frame-fish pairs that don't pass the threshold
    bad_registration_mask = registration_costs_meanBp > reg_threshold

    # use the mask to NaN positions which don't pass the threshold
    # (reg_thresh_removal_info will keep track of which fish in which frame we discard)
    reg_thresh_removal_info = []
    for fIdx in range(positions_3d.shape[0]):
        for fishIdx in range(positions_3d.shape[1]):
            if bad_registration_mask[fIdx, fishIdx]:
                reg_thresh_removal_info.append([fIdx, fishIdx])
                positions_image_coordinates_processed[:, fIdx, fishIdx, :, :] = np.NaN
                positions_3d_processed[fIdx, fishIdx, :, :] = np.NaN
    reg_thresh_removal_info = np.array(reg_thresh_removal_info)

    tE = time.time()
    print('finished: ', tE - t0, 's')
    return positions_3d_processed, positions_image_coordinates_processed, reg_thresh_removal_info



def apply_size_thresholds_to_positions(positions_3D, positions_imageCoordinates_processed,
                                     positions_3D_processed, head_pec_thresh, pec_tail_thresh):
    t0 = time.time()
    numFish = 2
    numFrames = positions_3D.shape[0]

    # Calculate the sizes
    head_pec_dists = np.ones((numFrames, numFish)) * np.NaN
    pec_tail_dists = np.ones((numFrames, numFish)) * np.NaN
    for fishIdx in range(numFish):
        for fIdx in range(numFrames):
            fishData = np.copy(positions_3D_processed[fIdx, fishIdx])
            head_pec_dists[fIdx, fishIdx] = np.linalg.norm(fishData[0] - fishData[1])
            pec_tail_dists[fIdx, fishIdx] = np.linalg.norm(fishData[1] - fishData[2])

    # remove any fish that are too big
    # TODO: should be also remove skeletons with zero distances
    size_thresh_removal_info = []
    for fIdx in range(positions_3D.shape[0]):
        for fishIdx in range(positions_3D.shape[1]):

            if head_pec_dists[fIdx, fishIdx] > head_pec_thresh:
                size_thresh_removal_info.append([fIdx, fishIdx, 0])
                positions_imageCoordinates_processed[:, fIdx, fishIdx, :, :] = np.NaN
                positions_3D_processed[fIdx, fishIdx, :, :] = np.NaN

            elif pec_tail_dists[fIdx, fishIdx] > pec_tail_thresh:
                size_thresh_removal_info.append([fIdx, fishIdx, 1])
                positions_imageCoordinates_processed[:, fIdx, fishIdx, :, :] = np.NaN
                positions_3D_processed[fIdx, fishIdx, :, :] = np.NaN

            else:
                continue
    size_thresh_removal_info = np.array(size_thresh_removal_info)

    tE = time.time()
    print('finished: ', tE - t0, 's')
    return positions_3D_processed, positions_imageCoordinates_processed, size_thresh_removal_info
