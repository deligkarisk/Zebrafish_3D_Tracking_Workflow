import numpy as np


def compute_3D_positions_from_registered_frame_instances(registered_frame_instances, calOb):
    ''' Given registered_frame_instances obtain the corresponding 3D bodypoint positions,
        namely the positions_3D for this frame.

    --- args ---
    registered_frame_instances: (numCams=3, numFish=2, numBodyPoints=3, numImCoords=2)
                                These are the 3 camera instances this frame that have been
                                registered together. See functions on registering frame instances.
    calOb: an instantiated calibration object

    --- returns ---
    fish_3D_positions: shape (numFish, numBodyPoints, 3),
                       the 3D positions of the bodypoints of the fish for this frame
    '''
    _, numFish, numBodyPoints, _ = registered_frame_instances.shape

    # compute the 3D positions
    fish_3D_positions = np.zeros((numFish, numBodyPoints, 3))
    for fishIdx in range(numFish):
        for bpIdx in range(numBodyPoints):
            if np.any(np.isnan(registered_frame_instances[:, fishIdx, bpIdx])):
                XYZ = np.NaN
            else:
                XYZ = calOb.compute_XYZ_from_imageCoord_triplet(registered_frame_instances[:, fishIdx, bpIdx])
            fish_3D_positions[fishIdx, bpIdx, :] = XYZ
    return fish_3D_positions