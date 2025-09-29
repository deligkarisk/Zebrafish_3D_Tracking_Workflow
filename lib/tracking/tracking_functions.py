from lib.registration.registration_methods_utils import make_all_permutations, make_permutation_costs_matrix, \
    check_method1_on_permutation_costs_matrix
from lib.tracking.frame_registration import register_frame_instances_method1, register_frame_instances_method2


def get_3D_positions_from_sleap_imcoords(frame_instances, cal, mean_reg_skel_thresh=2, debug=False):
    ''' Given the sLEAP results from all 3 camera views for a frame,
        return the 3D positions for fish bodypoints.

    -- inputs --
    frame_instances: (numCams=3, numFish=2, numBodyPoints=3, numCoords=2)
                     The results from the 3 sLEAP networks for a single frame
    calOb: an instantiated calibration object

    --- kwargs ---
    mean_reg_skel_thresh=2  : a threshold on registration costs.
                              This is a threshold on the bodypoint-mean cost of registering
                              a sleap skeleton from one camera view with a sleap skeleton from
                              a different camera view, with both skeletons having n>=2
                              bodypoints in common.
                              This parameter can be found by applying method1 to all
                              approriate frames, and hence estimating the distribution of costs
                              for correctly matched skeletons.
    debug=False             : if true, return dbug info

    -- returns ---
    methodIdx: the index of the method use to register across cameras.
    positions_imageCoordinates:   (numCams, numFish, numBodyPoints, 2)
                                  The registered image coordinates for this frame.
    positions_3DBps:              (numFish, numBodyPoints, 3)
                                  The associated 3D position of the registered imageCoordinates,
    positions_registration_costs: (numFish, numBodyPoints)
                                  The registration costs of all bodypoints of all fish.
    debug_vals:                   A list of internal variables that can be used for debugging.
    '''
    # parse some shapes
    numCams, numFish, numBodyPoints, _ = frame_instances.shape

    # define some vars on camera views
    good_cam_pairings = [[0, 1], [0, 2], [1, 2]]
    numGoodCamPairings = len(good_cam_pairings)

    #  get the permutations
    frame_permutations = make_all_permutations(frame_instances)
    numPerms = frame_permutations.shape[0]

    # compute the cost of each permutation
    permutation_costs = make_permutation_costs_matrix(frame_permutations, cal)

    # find the methodIdx
    do_use_method1 = check_method1_on_permutation_costs_matrix(permutation_costs)
    if do_use_method1:
        methodIdx = 0
    else:
        methodIdx = 1

    # solve the frame using the correct method
    if methodIdx == 0:
        positions_imageCoordinates, \
            positions_3DBps, \
            positions_registration_costs, \
            usedCamIdxs, \
            permutation_costs_meanBP, \
            twofish_permutation_costs_meanBP, \
            options_info_and_costs, \
            bestOptInfoIdx = register_frame_instances_method1(frame_permutations,
                                                              permutation_costs,
                                                              cal)
    elif methodIdx == 1:
        positions_imageCoordinates, \
            positions_3DBps, \
            positions_registration_costs, \
            perm_best_options_for_making_fish, \
            perm_best_options_for_making_fish_threshed, \
            perm_numFish_found, \
            best_permIdx = register_frame_instances_method2(frame_permutations, permutation_costs, cal,numCams=numCams,
                                                            numFish=numFish, numBodyPoints=numBodyPoints,
                                                            mean_reg_skel_thresh=mean_reg_skel_thresh)
    else:
        raise TypeError('methodIdx is not 0 or 1')

    # ---- finish up ---- #
    if debug:
        if methodIdx == 0:
            debug_vals = [usedCamIdxs,
                          permutation_costs,
                          permutation_costs_meanBP,
                          twofish_permutation_costs_meanBP,
                          options_info_and_costs,
                          bestOptInfoIdx]
        elif methodIdx == 1:
            debug_vals = [permutation_costs,
                          perm_best_options_for_making_fish,
                          perm_best_options_for_making_fish_threshed,
                          perm_numFish_found,
                          best_permIdx]

        return methodIdx, positions_imageCoordinates, positions_3DBps, positions_registration_costs, debug_vals

    else:
        return methodIdx, positions_imageCoordinates, positions_3DBps, positions_registration_costs, []
