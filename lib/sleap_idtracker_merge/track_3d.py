import numpy as np
from scipy.optimize import linear_sum_assignment

from lib.tracking.track_utils import compute_distance_between_3D_fish


def track_segment_in_3D_if_possible(existing_tracks_3D_for_segment, existing_imCoords_3D_for_segment,
                                    positions_3D_processed_for_segment,
                                    positions_imageCoordinates_processed_for_segment,
                                    last_known_positions, final_known_positions,
                                    tracks_available_post_pass2_id_assignments_for_segment):
    ''' Try to track the passed segment of frames, progagating identity
        through the 3D skeletons.


    --- Idea ---
    Using tracks_3D, we have the location of both individuals before and after this segment.
    We try to propagate identity through this collection of frames by tracking the observed
    3D skeletons. We know we did this correctly, if at the end of our tracking, our identities
    match the identities for after this segment.

    --- schematic ----
    * = last_known_position
    & = final_known_position

    |___before__|_*__|<-----segment--|-&-->|_____after_____|
    time ->

    Segment runs from the first frame without locations for both, up to and including
    the first frame where we have idxs for both again.
    Using the last known position to initialize the tracking during the segment,
    we track all the way until final_known_positions (the next time we have identities for
    both fish). If final_known_positions is equal to where we ended up by tracking,
    we declare success.

    --- args ---
    existing_tracks_3D_for_segment:
    existing_imCoords_3D_for_segment:
    positions_3D_processed_for_segment:
    positions_imageCoordinates_processed_for_segment:
    last_known_positions:
    final_known_positions:
    tracks_available_post_pass2_id_assignments_for_segment:

    --- returns ---
    ouput = [
            was_successful: bool, True if we successfully tracked
            segment_tracks_3D: ((refFE-regF0)+1, numFish, numBodypoints, 3)
            segment_tracks_imCoords: (3, (refFE-regF0)+1, numFish, numBodypoints, 2)
            seg_track_method: ((refFE-regF0)+1,), entries={0,1,2,3,NaN}
            seg_data_available_array: ((refFE-regF0)+1,numFish), 0=no_data, 1=data
            ]
    '''
    # parse input shapes
    reg_nfs, numFish, numBodyPoints, _ = existing_tracks_3D_for_segment.shape
    numCams = 3

    # preallocate the outputs
    segment_tracks_3D = np.zeros_like(existing_tracks_3D_for_segment) * np.NaN
    segment_tracks_imCoords = np.zeros((numCams, reg_nfs, numFish, numBodyPoints, 2)) * np.NaN

    # a debugging array to keep track of which method we used for the frame
    seg_track_method = np.zeros((reg_nfs,)) * np.NaN

    for ii in range(reg_nfs):

        # ---- case 1 ---- #
        # If we already entered a value in tracks_3D for either fish,
        # then both fish are already spoken for,
        # because if we found one in pass2 using idtracker comparison,
        # then we would have assigned the other detected skeleton and id if we have another
        # detected skeleton.
        # So test if we can resolve this frame by simply importing from tracks_3D,
        # unless this is the last frame, in which case we want to naturally land on
        # the right matching.

        # if we are at the last frame, skip this step
        if ii == reg_nfs - 1:
            pass
        # otherwise see if we have already fill-in the tracks for this frame
        else:
            # if some fish already identified
            if np.sum(tracks_available_post_pass2_id_assignments_for_segment[ii]) > 0:
                # assign the identities
                segment_tracks_3D[ii] = np.copy(existing_tracks_3D_for_segment[ii])
                segment_tracks_imCoords[:, ii] = np.copy(existing_imCoords_3D_for_segment[:, ii])
                # record the method
                seg_track_method[ii] = 0
                # update the last known positions
                for fishIdx in range(numFish):
                    if tracks_available_post_pass2_id_assignments_for_segment[ii, fishIdx] == 1:
                        last_known_positions[fishIdx] = segment_tracks_3D[ii, fishIdx]
                continue

        # --- case 2 ------ #
        # We are not referencing tracks_3D at all,
        # and will simply try to match observations to last known positions

        # get the observed 3D skeletons for this frame
        frame_positions = np.copy(positions_3D_processed_for_segment[ii])

        # count the number of obsercations available
        # Do we have 0,1,2 skeletons available
        fish_have_some_bps = np.zeros((numFish,), dtype=bool)
        for fishIdx in range(numFish):
            fish_have_some_bps[fishIdx] = ~np.all(np.isnan(frame_positions[fishIdx]))
        numFishFound = np.sum(fish_have_some_bps)

        # deal with 2 observation case
        if numFishFound == numFish:
            frame_cost_mat = np.zeros((numFish, numFish))  # the cost array
            current_fish = []
            last_fish = []
            for fishIdx in range(numFish):
                current_fish.append(frame_positions[fishIdx])
                last_fish.append(last_known_positions[fishIdx])
            # fill-in the cost matrix
            for curr_idx in range(numFish):
                for prev_idx in range(numFish):
                    frame_cost_mat[curr_idx, prev_idx] = compute_distance_between_3D_fish(current_fish[curr_idx],
                                                                                          last_fish[prev_idx])
            # solve the optimal assigning of identities between frames
            row_ind, col_ind = linear_sum_assignment(frame_cost_mat)
            # update the tracks arrays
            for fishIdx in range(numFish):
                segment_tracks_3D[ii, fishIdx] = np.copy(frame_positions[col_ind[fishIdx]])
                segment_tracks_imCoords[:, ii, fishIdx] = np.copy(
                    positions_imageCoordinates_processed_for_segment[:, ii, col_ind[fishIdx]])
            # update the last known positions
            last_known_positions = np.copy(segment_tracks_3D[ii])
            # record what we did
            seg_track_method[ii] = 1

        # deal with 1 obsercation case
        elif numFishFound == (numFish - 1):
            # find the last observation that this detection is closest to, and assign that identity
            # which fish is not missing
            detectionIdx = int(np.where(fish_have_some_bps)[0][0])
            detection_skeleton = frame_positions[detectionIdx]
            detection_imCoords = np.copy(positions_imageCoordinates_processed_for_segment[:, ii, detectionIdx])
            # find the distance to last known positions
            reg_costs = []
            for fishIdx in range(numFish):
                reg_costs.append(compute_distance_between_3D_fish(detection_skeleton, last_known_positions[fishIdx]))
            reg_costs = np.array(reg_costs)
            # find the correct identity for this observation
            fishIdx_for_observation = np.argmin(reg_costs)
            # record
            segment_tracks_3D[ii, fishIdx_for_observation] = detection_skeleton
            segment_tracks_imCoords[:, ii, fishIdx_for_observation] = detection_imCoords
            last_known_positions[fishIdx_for_observation] = detection_skeleton
            # record what we did
            seg_track_method[ii] = 2

        # deal with 0 observation case
        elif numFishFound == 0:
            # Do nothing, and move to next frame
            # record what we did
            seg_track_method[ii] = 3
            continue

        # deal with 0 observation case
        else:
            raise TypeError('frame {0}: number of fish detected is not 0,1 or 2.'.format(ii))

    # ----- count frames where we have skeleton info, -----#
    seg_data_available_array = np.zeros((reg_nfs, numFish)) * np.NaN
    for fidx in range(segment_tracks_3D.shape[0]):
        for fishIdx in range(numFish):
            if ~np.all(np.isnan(segment_tracks_3D[fidx, fishIdx])):
                seg_data_available_array[fidx, fishIdx] = 1
            else:
                seg_data_available_array[fidx, fishIdx] = 0

    # ---- if the end of the segment_tracks_3D matches the next known positions --- #
    # declare success
    if np.all(segment_tracks_3D[-1][~np.isnan(segment_tracks_3D[-1])] == final_known_positions[
        ~np.isnan(final_known_positions)]):
        was_successful = True
    else:
        was_successful = False

    # finish up
    return [was_successful, segment_tracks_3D, segment_tracks_imCoords, seg_track_method, seg_data_available_array]
