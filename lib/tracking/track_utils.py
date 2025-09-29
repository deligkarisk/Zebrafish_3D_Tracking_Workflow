import os

import numpy as np


def save_tracks_3d(tracks_3D, save_data_folder):

    # get the head positions of both fish
    # head is bpIdx=0
    bpIdx = 0
    h1_data = np.copy(tracks_3D[:, 0, bpIdx, :])
    h2_data = np.copy(tracks_3D[:, 1, bpIdx, :])

    np.savetxt(os.path.join(save_data_folder, 'head1.txt'), h1_data, delimiter=',', fmt='%s')
    np.savetxt(os.path.join(save_data_folder, 'head2.txt'), h2_data, delimiter=',', fmt='%s')

    bpIdx = 1
    fin1_data = np.copy(tracks_3D[:, 0, bpIdx, :])
    fin2_data = np.copy(tracks_3D[:, 1, bpIdx, :])

    np.savetxt(os.path.join(save_data_folder, 'fin1.txt'), fin1_data, delimiter=',', fmt='%s')
    np.savetxt(os.path.join(save_data_folder, 'fin2.txt'), fin2_data, delimiter=',', fmt='%s')

    bpIdx = 2
    tail1_data = np.copy(tracks_3D[:, 0, bpIdx, :])
    tail2_data = np.copy(tracks_3D[:, 1, bpIdx, :])

    np.savetxt(os.path.join(save_data_folder, 'tail1.txt'), tail1_data, delimiter=',', fmt='%s')
    np.savetxt(os.path.join(save_data_folder, 'tail2.txt'), tail2_data, delimiter=',', fmt='%s')


def create_array_of_start_stop_frames_for_parallelization(numFrames, step=1000):
    ''' Return an array that we can index easily to divide the frames into chunks
        for processing in parallel.

    --- args ---
    numFrames: the number of frames we want to process
    step=1000: the size (in frames) of each chunk

    --- returns ---
    start_stop_frms: array of shape (numChunks, 2), where
                     each row contains the start frame and stop frame
                     for that chunk.

    --- example ---
    >> nfs = 501943
    >> start_stop_frms = create_array_of_start_stop_frames_for_parallelization(nfs, step=1000)
    >> start_stop_frms
    array([[     0,   1000],
           [  1000,   2000],
           [  2000,   3000],
           ...,
           [499000, 500000],
           [500000, 501000],
           [501000, 501943]])
    '''
    start_frms = np.arange(0, numFrames, step)
    stop_frms = np.append(np.arange(step, numFrames, step), numFrames)
    start_stop_frms = np.stack([start_frms, stop_frms], axis=1)
    return start_stop_frms


def compute_distance_between_3D_fish(fish1_3D_bps, fish2_3D_bps):
    ''' Return the distance between the two fish using our chosen metric
    '''
    numBodyPoints = fish1_3D_bps.shape[0]
    bp_distances = np.ones((numBodyPoints))
    for bpIdx in range(numBodyPoints):
        bp_distances[bpIdx] = np.linalg.norm(fish1_3D_bps[bpIdx] - fish2_3D_bps[bpIdx])
    distance = np.nanmean(bp_distances)
    return distance