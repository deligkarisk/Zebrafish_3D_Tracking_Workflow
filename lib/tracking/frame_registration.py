import numpy as np

from lib.registration.compute_position import compute_3D_positions_from_registered_frame_instances
from lib.registration.fill_camera_views import fill_in_bad_camera_view_image_coordinates, \
    fill_in_bad_camera_view_image_coordinates_single_fish
from lib.registration.registration_methods_utils import find_mean_bodypoint_costs_for_more_than_two_bodypoints_found




def find_mean_bodypoint_registration_costs_array(registration_costs_array):
    ''' Return a version of the registration_costs_array where the bodypoint dimension has
        been collapsed down to the mean of the bodypooint costs, but only for skeleton
        registration costs with 2 or more nonNan values for bodypoints. If only 1 bodypoint is
        registered, the skeleton registration cost is set to NaN.

    --- examples ---
    <<Skeleton registration costs>>  ---------------- > <<summed values>>

    [  0.4673873 , 0.45785861,   0.79911474] ---------> 0.57478688
    [       nan,   6.54993568,   4.82579788] ---------> 5.68786678
    [2.56936649,          nan,   2.36329278] ---------> 2.46632964
    [         nan,        nan,  53.06485773] ---------> nan

     --- args ---
    registration_costs_array: shape (numFrames, numFish, numBodyPoints).
                              Comes from the cross-camera registration step.

    --- returns ---
    registration_costs_array_meanBp: shape (numFrames, numFish)
    '''

    numFrames, numFish, numBodyPoints = registration_costs_array.shape

    registration_costs_meanBp = []
    for fishIdx in range(numFish):

        fish_reg_data = registration_costs_array[:,fishIdx]

        # count the number of nonNaN bodypoint registrations in each frame
        numBps_registered_each_frame = np.count_nonzero(~np.isnan(fish_reg_data), axis=(1))

        # take the mean along bodypoints
        registration_costs_fish_meanBp = np.nanmean(fish_reg_data, axis=1)

        # now naN any frame results if we detected less than 2 bodypoints
        registration_costs_fish_meanBp[numBps_registered_each_frame < 2] = np.NaN

        # record
        registration_costs_meanBp.append(registration_costs_fish_meanBp)

    # finish up
    registration_costs_array_meanBp = np.stack(registration_costs_meanBp, axis=1)
    return registration_costs_array_meanBp



def register_frame_instances_method1(frame_permutations, permutation_costs, calOb):
    ''' Find the cam pairing with the lowest cost for registering both fish.
        This is method 1 of registering frame_permutations.

    --- args ---
    frame_permutations: a list containing all permutations of the frame instances
    permutation_costs: shape (numPerms, numGoodCamPairings, numFish, numBodyPoints).
                             Array containing the costs of all permutation registrations
    calOb: an instantiated calibration object

    --- returns ---
    positions_imageCoordinates:       (numCams, numFish, numBodyPoints, 2)
                                      The registered image coordinates for this frame.
    positions_3DBps:                  (numFish, numBodyPoints, 3)
                                      The associated 3D position of the registered
                                      imageCoordinates.
    positions_registration_costs:     (numFish, numBodyPoints)
                                      The registration costs of all bodypoints of all fish.
    usedCamIdxs:                      List of two ints. Contains the idxs of the cam pair used.
                                      i.e. [0,1], [0,2], or [1,2]
    permutation_costs_meanBP:         The permutation_costs array collapsed along bodypoints,
                                      so shape=(numPerms, numGoodCamPairings, numFish)
    twofish_permutation_costs_meanBP: The permutation_costs array collapsed along Fish,
                                      so shape=(numPerms, numGoodCamPairings)
    options_info_and_costs:           An array of shape (numPerms*numGoodCamPairings, 3),
                                      where the elements of the second dimension represent
                                      [permIdx, camPairingIdx, totalRegistrationCost],
                                      with totalRegistrationCost
    bestOptInfoIdx:                   The row index of the best option from the
                                      options_info_and_costs array.


    --- see also ---
    check_method1() -> a function for checking if we can use this function to solve the frame

    '''
    # define some vars on camera views
    good_cam_pairings = [[0, 1], [0, 2], [1, 2]]
    numGoodCamPairings = len(good_cam_pairings)
    numPerms = len(frame_permutations)

    # sum costs for bodypoints
    permutation_costs_meanBP = find_mean_bodypoint_costs_for_more_than_two_bodypoints_found(permutation_costs)

    # sum the costs for both fish
    # shape (numPerms, numCamPairings)
    twofish_permutation_costs_meanBP = np.sum(permutation_costs_meanBP, axis=-1)

    # check if we can proceed:
    # if not all option costs are zero, then we have at least 1 option where we can register both fish
    if np.all(np.isnan(twofish_permutation_costs_meanBP)):
        raise TypeError('frame_permutations does not contain enough info to use used with this method')

    # make an array for (permIdx, campairIdx, cost) for each option from all perms
    numOptionRows = numPerms * numGoodCamPairings
    options_info_and_costs = []
    for permIdx in range(numPerms):
        for camPairIdx in range(numGoodCamPairings):
            options_info_and_costs.append(
                np.array([permIdx, camPairIdx, twofish_permutation_costs_meanBP[permIdx, camPairIdx]]))
    options_info_and_costs = np.array(options_info_and_costs)

    # find the option with the lowest nonNan cost for registering both fish
    # Note: entries will come in pairs, we pick the first one, the order doesnt matter
    bestOptInfoIdx = np.argsort(options_info_and_costs[:, -1])[0]
    bestOpt_permIdx = int(options_info_and_costs[bestOptInfoIdx, 0])
    bestOpt_camPairIdx = int(options_info_and_costs[bestOptInfoIdx, 1])
    bestOpt_cost = options_info_and_costs[bestOptInfoIdx, 2]  #

    # now derivate the missing imagecoordinates, and get the 3D positions
    chosen_frame_coordinates = np.copy(frame_permutations[bestOpt_permIdx])
    chosen_camPairing_idx = good_cam_pairings[bestOpt_camPairIdx]
    positions_imageCoordinates = fill_in_bad_camera_view_image_coordinates(chosen_frame_coordinates,
                                                                           chosen_camPairing_idx,
                                                                           calOb)

    # now get the 3D positions
    positions_3DBps = compute_3D_positions_from_registered_frame_instances(positions_imageCoordinates, calOb)

    # get the registration costs for the option we used
    positions_registration_costs = permutation_costs[bestOpt_permIdx, bestOpt_camPairIdx, :, :]

    # for debugging purposes
    usedCamIdxs = good_cam_pairings[bestOpt_camPairIdx]

    # finish up
    outputs = [positions_imageCoordinates,
               positions_3DBps,
               positions_registration_costs,
               usedCamIdxs,
               permutation_costs_meanBP,
               twofish_permutation_costs_meanBP,
               options_info_and_costs,
               bestOptInfoIdx]
    return outputs


def  register_frame_instances_method2(frame_permutations, permutation_costs, cal, numCams, numFish, numBodyPoints, mean_reg_skel_thresh=2):
    ''' Find the cam pairing with the lowest cost for registering both fish.
        This is method 2 of registering frame_permutations.

        --- args ---
        frame_permutations: a list containing all permutations of the frame instances
        permutation_costs: shape (numPerms, numGoodCamPairings, numFish, numBodyPointss).
                                 Array containing the costs of all permutation registrations
        calOb: an instantiated calibration object

        --- kwargs ---
        mean_reg_skel_thresh=10 : a threshold on registration costs.
                                  This is a threshold on the bodypoint-mean cost of registering
                                  a sleap skeleton from one camera view with a sleap skeleton from
                                  a different camera view, with both skeletons having n>=2
                                  bodypoints in common.
                                  This parameter can be found by applying method1 to all
                                  approriate frames, and hence estimating the distribution of costs
                                  for correctly matched skeletons.

        --- returns ---
        positions_imageCoordinates:       (numCams, numFish, numBodyPoints, 2)
                                          The registered image coordinates for this frame.
        positions_3DBps:                  (numFish, numBodyPoints, 3)
                                          The associated 3D position of the registered
                                          imageCoordinates.
        positions_registration_costs:     (numFish, numBodyPoints)
                                          The registration costs of all bodypoints of all fish.

        --- see also ---
        check_method1() -> a function for checking if we can use this function to solve the frame

    '''
    # define some vars on camera views
    good_cam_pairings = [[0, 1], [0, 2], [1, 2]]
    numGoodCamPairings = len(good_cam_pairings)
    numPerms = len(frame_permutations)

    # sum costs for bodypoints
    permutation_costs_meanBP = find_mean_bodypoint_costs_for_more_than_two_bodypoints_found(permutation_costs)

    # ------- find the best permutation to use ----- #
    # these are list over permutations, with each element being an array of length 2,
    # with elements of the cost of registering a fish and the cam pairing used.
    # The threshed version just applies the mean_reg_skel_thresh
    perm_best_options_for_making_fish = []
    perm_best_options_for_making_fish_threshed = []

    for permIdx in range(numPerms):

        reg_costs = permutation_costs_meanBP[permIdx]

        best_pairings_for_fish = []
        best_pairings_for_fish_threshed = []

        for fishIdx in range(numFish):
            # find the info for this fish
            if np.any(~np.isnan(reg_costs[:, fishIdx])):
                fish_min_cost = np.nanmin(reg_costs[:, fishIdx])
                fish_min_camPairIdx = np.argsort(reg_costs[:, fishIdx])[0]
                best_pairings_for_fish.append(np.array([fish_min_cost, fish_min_camPairIdx]))
            else:
                fish_min_cost = np.NaN
                fish_min_camPairIdx = np.NaN
                best_pairings_for_fish.append(np.array([fish_min_cost, fish_min_camPairIdx]))

            # does it pass threshold?
            if ~np.isnan(fish_min_cost):
                if fish_min_cost < mean_reg_skel_thresh:
                    best_pairings_for_fish_threshed.append(np.array([fish_min_cost, fish_min_camPairIdx]))
                else:
                    best_pairings_for_fish_threshed.append(np.array([np.NaN, np.NaN]))
            else:
                best_pairings_for_fish_threshed.append(np.array([np.NaN, np.NaN]))

        perm_best_options_for_making_fish.append(best_pairings_for_fish)
        perm_best_options_for_making_fish_threshed.append(best_pairings_for_fish_threshed)

    # --- count the number of found fish in each permutation ----#
    # an array containing the number of fish we can register in each permutation
    perm_numFish_found = []
    for permIdx, permInfo in enumerate(perm_best_options_for_making_fish_threshed):
        permFishFound = 0
        for fishIdx in range(numFish):
            fishRegCost = permInfo[fishIdx][0]
            if ~np.isnan(fishRegCost):
                permFishFound += 1
        perm_numFish_found.append(permFishFound)
    perm_numFish_found = np.array(perm_numFish_found)

    # ---- the best permutation ---------#
    # The only with the most fish, past that, we don't care (since all costs passed threshold)
    best_permIdx = np.argmax(perm_numFish_found)
    best_options_for_making_fish = perm_best_options_for_making_fish_threshed[best_permIdx]
    permuted_frame_instances = np.copy(frame_permutations[best_permIdx])

    # --- find the final positions_registration_costs ---#
    # positions_registration_costs array of shape (numFish, numBodyPoints)
    camIdxs_used_for_fish = []
    for fishIdx in range(numFish):
        cam_used_for_fish = best_options_for_making_fish[fishIdx][1]
        camIdxs_used_for_fish.append(cam_used_for_fish)
    camIdxs_used_for_fish = np.array(camIdxs_used_for_fish)

    positions_registration_costs = []
    for fishIdx in range(numFish):
        camPairIdx_used = camIdxs_used_for_fish[fishIdx]
        if ~np.isnan(camPairIdx_used):
            camPairIdx_used = int(camPairIdx_used)
            fish_reg_cost = permutation_costs[best_permIdx, camPairIdx_used, fishIdx]
        else:
            fish_reg_cost = np.ones((numBodyPoints,)) * np.NaN
        positions_registration_costs.append(fish_reg_cost)
    positions_registration_costs = np.array(positions_registration_costs)

    # --- fill-in the coordinates of both fish using calibration ------ #
    # preallocate the output
    positions_imageCoordinates = np.ones((numCams, numFish, numBodyPoints, 2)) * np.NaN
    for fishIdx in range(numFish):
        # parse the values of interest for this fish
        fish_bestOpt_cost, fish_bestOpt_camPairIdx = best_options_for_making_fish[fishIdx]
        fish_instances = permuted_frame_instances[:, fishIdx, :, :]

        # if we can register the fish, fill-in its 3rd cam image coordinates
        if ~np.isnan(fish_bestOpt_cost):
            can_make_fish = True
            fill_in_fish_instances = fill_in_bad_camera_view_image_coordinates_single_fish(fish_instances,
                                                                                           good_cam_pairings[
                                                                                               int(fish_bestOpt_camPairIdx)],
                                                                                           cal)
        # if we cant register the fish, return NaN image coordinates so we don't try to make this fish
        else:
            can_make_fish = False
            fill_in_fish_instances = np.ones_like(fish_instances) * np.NaN

        # record
        positions_imageCoordinates[:, fishIdx] = fill_in_fish_instances

    # ----- now get the 3D positions ---- #
    positions_3DBps = compute_3D_positions_from_registered_frame_instances(positions_imageCoordinates, cal)

    # --------- finish up ------------------#
    outputs = [positions_imageCoordinates,
               positions_3DBps,
               positions_registration_costs,
               perm_best_options_for_making_fish,
               perm_best_options_for_making_fish_threshed,
               perm_numFish_found,
               best_permIdx]
    return outputs
