import numpy as np


def load_and_process_idtracker_data(idtraj, acceleration_thresh, NaN_window):
    ''' A function to load idTracker results and parse them, any potential swaps
        after collision events by results with very high accelerations

    --- args ---
    idTracker_filepath:  the filepath to the trajectories.npy file for an experiment
    acceleration_thresh: the threshold on the absolute value of the acceleration
    NaN_window:          the number of frames either side of problematic frames to remove

    --- returns ---
    idtraj: the processed idtracker timeseries.

    -- Thanks to --
    Simon Goorney
    '''
    # load the centroid timeseries
  #  trajectories_dict = np.load(idTracker_filepath, allow_pickle=True).item()
  #  idtraj = trajectories_dict['trajectories']

    # compute the relevant derivatives
    speed = np.gradient(idtraj, axis=0)
    normspeed = np.linalg.norm(speed, axis=2)
    normaccel = np.gradient(normspeed, axis=0)
    absaccel = np.abs(normaccel)

    # compute the number of frames in the experiment
    nfs = speed.shape[0]

    def compare_nan_array(func, a, thresh):
        # Thanks: https://stackoverflow.com/a/47340067
        out = ~np.isnan(a)
        out[out] = func(a[out], thresh)
        return out

    # find problematic frames
    super_threshold_indices = np.where(compare_nan_array(np.greater, absaccel, acceleration_thresh))
    indices = super_threshold_indices[0]

    # loop through problem frames, Nan'ing either side in a window
    for i in indices:
        if i < NaN_window:
            idtraj[:i + int(NaN_window / 2)] = np.NaN
        elif nfs - i < NaN_window:
            idtraj[i - int(NaN_window / 2):] = np.NaN
        else:
            idtraj[i - int(NaN_window / 2):i + int(NaN_window / 2)] = np.NaN

    # finish up
    return idtraj