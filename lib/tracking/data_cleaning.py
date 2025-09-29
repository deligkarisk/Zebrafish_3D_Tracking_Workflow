import numpy as np

from lib.tracking.fix_swaps import (
    pair_swap_fix, 
    single_fish_jump_fix, 
    get_order_velocity,
    fix_single_fish_jumps_in_positions
)


def clean_data_from_jumps(tracks, tracks_im_coords, dt, threshold=20, max_frame_window=20):
    """
    Clean tracking data by removing jumps and outliers.

    This function applies several cleaning steps to the tracking data:
    1. Removes pair switches (when two fish IDs are swapped)
    2. Removes single fish jumps (when a single fish has a sudden position change)
    3. Removes velocity outliers across multiple time scales

    Parameters
    ----------
    tracks : numpy.ndarray
        3D tracking data with shape (frames, fish, bodypoints, coordinates)
    tracks_im_coords : numpy.ndarray
        Image coordinates tracking data
    dt : float
        Time step between frames
    threshold : float, optional
        Threshold for detecting jumps, default is 20
    max_frame_window : int, optional
        Maximum number of frames to consider for fixing jumps, default is 20

    Returns
    -------
    tuple
        (cleaned_tracks, pair_switch_removed_frames, velocity_outlier_removed_frames)
    """
    # Initialize counters for removed frames
    pair_switch_removed_frames = np.zeros((len(tracks), 1))
    velocity_outlier_removed_frames = np.zeros((len(tracks), 1))

    # Step 1: Remove pair switches (first order)
    tracks, pair_switch_removed_frames = pair_swap_fix(
        tracks, 
        tracks_im_coords, 
        pair_switch_removed_frames,
        threshold=threshold, 
        max_frame_window=max_frame_window
    )

    # Step 2: Remove pair switches (second order)
    tracks, pair_switch_removed_frames = pair_swap_fix(
        tracks, 
        tracks_im_coords, 
        pair_switch_removed_frames,
        threshold=threshold, 
        max_frame_window=max_frame_window, 
        order=2
    )

    # Increase threshold for single fish jump detection
    single_fish_threshold = threshold * 2

    # Step 3: Remove single fish jumps (first order)
    tracks, pair_switch_removed_frames = single_fish_jump_fix(
        tracks, 
        tracks_im_coords, 
        pair_switch_removed_frames, 
        threshold=single_fish_threshold, 
        max_frame_window=max_frame_window
    )

    # Step 4: Remove single fish jumps (second order)
    tracks, pair_switch_removed_frames = single_fish_jump_fix(
        tracks, 
        tracks_im_coords, 
        pair_switch_removed_frames, 
        threshold=single_fish_threshold, 
        max_frame_window=max_frame_window,
        order=2
    )

    # Step 5: Remove velocity outliers at multiple time scales
    # This iterates through different time scales (orders) to catch outliers
    # that might only be visible at certain temporal resolutions
    for order in range(1, 25):
        # Calculate velocity at current time scale
        velocity = get_order_velocity(tracks, order, dt)

        # Set threshold as high percentile of velocity distribution
        velocity_threshold = np.nanpercentile(velocity, 99.995)

        # Remove frames with velocities above threshold
        tracks, velocity_outlier_removed_frames = fix_single_fish_jumps_in_positions(
            tracks, 
            velocity_outlier_removed_frames,
            velocity, 
            threshold=velocity_threshold,
            order=order
        )

    return tracks, pair_switch_removed_frames, velocity_outlier_removed_frames
