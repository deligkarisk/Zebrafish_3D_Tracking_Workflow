import numpy as np
import pandas as pd
from scipy.signal import savgol_filter as sgf


def interpolate_over_small_gaps(tracks_3D_raw, limit=5, polyord=1):
    ''' Return an interpolated version of the raw trajectories.
        These will still contain NaNs in big gaps

    -- args --
    tracks_3D_raw: array, shape (numFrames,numFish,numBodyPoints,3),
                   the trajectories before post-processing
    limit: int, the maximum size of gaps to interpolate over
    polyord: the order of the polynomial to use for the interpolation.
             (normally set to 1, as we will later smooth with a higher order poly)

    -- Returns --
    tracks_3D_interpd: array, shape (numFrames,numFish,numBodyPoints,3),
                        the trajectories after interpolation over small gaps
    '''
    # parse shapes
    print(tracks_3D_raw.shape)
    numFrames, numFish, numBodyPoints, _ = tracks_3D_raw.shape

    tracks_3D_interpd = np.copy(tracks_3D_raw)
    # loop to grab a 1D timeseries
    for fishIdx in range(numFish):
        for bpIdx in range(numBodyPoints):
            for dimIdx in range(3):
                # interpolate the data in the most basic way
                data_1D = tracks_3D_raw[:, fishIdx, bpIdx, dimIdx]
                df = pd.DataFrame(data_1D)
                interpd_df = df.interpolate(method='polynomial', order=polyord, limit_direction='both', limit=limit,
                                            inplace=False)
                # record

                tracks_3D_interpd[:, fishIdx, bpIdx, dimIdx] = interpd_df.values[:, 0]

    return tracks_3D_interpd


def get_smooth_timeseries_and_derivatives_using_savgol(tracks_3D_interpd, win_len=9, polyOrd=2, dt=0.01):
    ''' Given trajectories, which should be, not not necessarily are,
        interpolated, return smoothes versions of the data and derivative info

    --- Method ---
    Apply a savitzky-golay filter to each component of the trajectory
    vector separately

    --- args ---
    tracks_3D_interpd: array, shape (numFrames,numFish,numBodyPoints,3),
                       the trajectories after interpolation over small gaps

    -- kwargs --
    win_len=9: the window size in frames for the savitzky-golay filter
    polyOrd=2: the polynomial order for the savitzky-golay filter
    dt=0.01: the time interval between frames (1/fps). Used for computing derivatives.

    --- Returns ---
    tracks_3D_smooth:      array, shape (numFrames,numFish,numBodyPoints,3),
                           the trajectories after smoothing
    tracks_3D_vel_smooth:  array, shape (numFrames,numFish,numBodyPoints,3),
                           the first derivatrive of the trajectories after smoothing
    tracks_3D_speed_smooth: array, shape (numFrames,numFish,numBodyPoints),
                            the norm of the vel_smooth
    tracks_3D_accvec_smooth: array, shape (numFrames,numFish,numBodyPoints,3),
                             the second derivatrive of the trajectories after smoothing
    tracks_3D_accmag_smooth: array, shape (numFrames,numFish,numBodyPoints),
                             the norm of the accvec_smooth
    '''
    # parse shapes
    numFrames, numFish, numBodyPoints, _ = tracks_3D_interpd.shape

    # preallocate
    tracks_3D_smooth = np.copy(tracks_3D_interpd)
    tracks_3D_vel_smooth = np.copy(tracks_3D_interpd)
    tracks_3D_speed_smooth = np.ones((numFrames, numFish, numBodyPoints)) * np.NaN
    tracks_3D_accvec_smooth = np.copy(tracks_3D_interpd)
    tracks_3D_accmag_smooth = np.ones((numFrames, numFish, numBodyPoints)) * np.NaN

    # smoothed position
    for fishIdx in range(numFish):
        for bpIdx in range(numBodyPoints):
            for dimIdx in range(3):
                raw_data_1D = np.copy(tracks_3D_interpd[:, fishIdx, bpIdx, dimIdx])
                tracks_3D_smooth[:, fishIdx, bpIdx, dimIdx] = sgf(raw_data_1D, window_length=win_len, polyorder=polyOrd)

    # smooth velocity and speed
    for fishIdx in range(numFish):
        for bpIdx in range(numBodyPoints):
            for dimIdx in range(3):
                raw_data_1D = np.copy(tracks_3D_interpd[:, fishIdx, bpIdx, dimIdx])
                tracks_3D_vel_smooth[:, fishIdx, bpIdx, dimIdx] = sgf(raw_data_1D, window_length=win_len, deriv=1,
                                                                      polyorder=polyOrd) / dt
    for fishIdx in range(numFish):
        for bpIdx in range(numBodyPoints):
            for dimIdx in range(3):
                tracks_3D_speed_smooth[:, fishIdx, bpIdx] = np.linalg.norm(tracks_3D_vel_smooth[:, fishIdx, bpIdx, :],
                                                                           axis=1)

    # smooth acceleration vector and magnitude
    for fishIdx in range(numFish):
        for bpIdx in range(numBodyPoints):
            for dimIdx in range(3):
                raw_data_1D = np.copy(tracks_3D_interpd[:, fishIdx, bpIdx, dimIdx])
                tracks_3D_accvec_smooth[:, fishIdx, bpIdx, dimIdx] = sgf(raw_data_1D, window_length=win_len, deriv=2,
                                                                         polyorder=polyOrd) / dt
    for fishIdx in range(numFish):
        for bpIdx in range(numBodyPoints):
            for dimIdx in range(3):
                tracks_3D_accmag_smooth[:, fishIdx, bpIdx] = np.linalg.norm(
                    tracks_3D_accvec_smooth[:, fishIdx, bpIdx, :], axis=1)

    outs = [tracks_3D_smooth,
            tracks_3D_vel_smooth,
            tracks_3D_speed_smooth,
            tracks_3D_accvec_smooth,
            tracks_3D_accmag_smooth]
    return outs