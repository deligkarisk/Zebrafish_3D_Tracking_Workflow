import numpy as np




def make_all_permutations(frame_instances):
    ''' Given the instances (or tracks) array, along with a frame idx and cam idx,
        return all possible permutations of the results for this frame
    '''
    # perm1 is as we got it
    perm1 = np.copy(frame_instances)

    # perm2: xz swap
    perm2 = np.copy(frame_instances)
    perm2[0, 0] = np.copy(frame_instances[0 ,1])
    perm2[0, 1] = np.copy(frame_instances[0 ,0])

    # perm3: xy swap
    perm3 = np.copy(frame_instances)
    perm3[1, 0] = np.copy(frame_instances[1 ,1])
    perm3[1, 1] = np.copy(frame_instances[1 ,0])

    # perm4: yz swap
    perm4 = np.copy(frame_instances)
    perm4[2, 0] = np.copy(frame_instances[2 ,1])
    perm4[2, 1] = np.copy(frame_instances[2 ,0])

    permutations = [perm1, perm2, perm3, perm4]
    permutations = np.stack(permutations, axis=0)
    return permutations


def make_permutation_costs_matrix(frame_permutations, calOb):
    ''' Give a list containing all permutations of the frame instances,
        return an array containing the registration costs of all possible options.

    --- args --
    frame_permutations: a list containing all permutations of the frame instances
    calOb:              an instantiated calibration object

    --- returns ---
    permutation_costs: shape (numPerms, numGoodCamPairings, numFish, numBodyPoints)
                       The cost of registering detected skeletons across camera views
                       for all permutations and all camera pairings.

    --- see also ---
    make_all_permutations()
    '''
    # parse input shapes
    numPerms = len(frame_permutations)
    _, numFish, numBodyPoints, _ = frame_permutations[0].shape

    # define some vars on camera views
    good_cam_pairings = [ [0 ,1], [0 ,2], [1 ,2] ]
    numGoodCamPairings = len(good_cam_pairings)

    # ----------------------------------------------------#
    # preallocate output array
    permutation_costs = np.zeros((numPerms, numGoodCamPairings, numFish, numBodyPoints))

    # loop over permutations
    for permIdx in range(numPerms):
        perm_instances = frame_permutations[permIdx]

        # loop over each good_cam_pairing
        for good_cams_idx, good_cams in enumerate(good_cam_pairings):

            # compute the error for each bp of each fish for this cam pairing in this permutation
            for fishIdx in range(numFish):
                for bpIdx in range(numBodyPoints):
                    cam1_bp = perm_instances[good_cams[0], fishIdx, bpIdx]
                    cam2_bp = perm_instances[good_cams[1], fishIdx, bpIdx]
                    error = calOb.compute_point_correspondence_error(good_cams, cam1_bp, cam2_bp)
                    permutation_costs[permIdx, good_cams_idx, fishIdx, bpIdx] = error
    return permutation_costs


def find_mean_bodypoint_costs_for_more_than_two_bodypoints_found(permutation_costs):
    ''' Return a version of the permutation_costs where the bodypoint dimension has
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
    permutation_costs: shape (numPerms, numGoodCamPairings, numFish, numBodyPoints)

    --- returns ---
    permutation_costs_meanBp: shape (numPerms, numGoodCamPairings, numFish)
    '''
    numPerms, numGoodCamPairings, numFish, numBodyPoints = permutation_costs.shape

    # preallocate
    permutation_costs_meanBp = np.ones((numPerms ,numGoodCamPairings ,numFish) ) *np.NaN
    for permIdx in range(numPerms):
        for gcIdx in range(numGoodCamPairings):
            for fishIdx in range(numFish):
                bp_costs = permutation_costs[permIdx, gcIdx, fishIdx]
                numBps_used = np.count_nonzero(~np.isnan(bp_costs))
                if numBps_used >= 2:
                    permutation_costs_meanBp[permIdx, gcIdx, fishIdx] = np.nanmean(bp_costs)

    return permutation_costs_meanBp




def check_method1_on_permutation_costs_matrix(permutation_costs):
    ''' Check if register_frame_instances_method1() will work to register the
        camera view for this permutation costs matrix.

        Big Idea: Can this frame be solved without a threshold on registration costs?

        Idea: We want to find a permutation of the data, together with a pairing
              of two camera views, where we have a non-Nan cost for registering
              3D positions for at least 2 bodypoints on both fish.

        Process: First sum along bodypoint costs, casting to NaN if less than
                 two bodypoints are available.
                 Then sum along fish, to get the cost of registering for each
                 cam pairing in each permutation.

    --- see also ---
    register_frame_instances_method1()

    --- args ---
    permutation_costs: shape (numPerms, numGoodCamPairings, numFish, numBodyPoints)

    --- returns ---
    True of False: True to use method1, False for don't use method1
    '''
    numPerms, numGoodCamPairings, numFish, numBodyPoints = permutation_costs.shape

    # sum costs for bodypoints
    permutation_costs_meanBP = find_mean_bodypoint_costs_for_more_than_two_bodypoints_found(permutation_costs)

    # sum the costs for both fish
    # shape (numPerms, numCamPairings)
    twofish_permutation_costs_meanBp = np.sum(permutation_costs_meanBP, axis=-1)

    # if not all option costs are zero, then we have at least 1 option where we can register both fish
    if ~np.all(np.isnan(twofish_permutation_costs_meanBp)):
        return True
    else:
        return False
