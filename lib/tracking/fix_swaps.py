import numpy as np


def single_fish_jump_fix(tracks, tracks_imCoords, removed_frames, threshold, max_frame_window=14, order=1):
    # max_frame_window: The pair swapping issue appears as follows:
    # fish ids are swapped and the new assignment remains for a certain number of frames
    # then, after e.g. 10-15 frames the ids revert back to their original ones.
    # this scripts finds not only the frames with swaps but also removed the in-between frames where the ids have been swapped.
    # the max_frame_window parameter is a threshold on the maximum size of the number of in-between frames.

    new_tracks = np.copy(tracks)

    diff = np.ones(tracks_imCoords.shape) * np.NaN

    for i_frame in range(order, tracks.shape[0]):
        diff[:, i_frame, :, :, :] = tracks_imCoords[:, i_frame, :, :, :] - tracks_imCoords[:, i_frame - order, :, :, :]
    vel = np.linalg.norm(diff, axis=4)

    vel_mask = vel[:, :, :, :] > threshold

    # both fishes jump, for any bodypoint, in any of the cameras

    # any body point jumps
    bd_jump = np.any(vel_mask, axis=(3))
    # for any of the fishes
    fish_jump = np.any(bd_jump, axis=(2))
    # in any of the cameras
    fish_any_camera_jump = np.any(fish_jump, axis=(0))
    # find the indexes
    both_fishes_jump_index = np.where(fish_any_camera_jump)[0]

    total_removed_frames = []

    for i in range(1, len(both_fishes_jump_index)):
        frames_diff = both_fishes_jump_index[i] - both_fishes_jump_index[i - 1] + 1

        if frames_diff <= max_frame_window:
            # removed_frames_diff_all.append(frames_diff)
            begin_frame = both_fishes_jump_index[i - 1]
            end_frame = both_fishes_jump_index[i] + 1
            total_removed_frames.append(frames_diff)
            new_tracks[begin_frame:end_frame] = np.NaN
            removed_frames[begin_frame:end_frame] = 1

    total_removed_frames = sum(total_removed_frames)
    print("Removed {0} frames with single fish jump in image coordinates".format(total_removed_frames))

    return new_tracks, removed_frames


def pair_swap_fix(tracks, tracks_imCoords, removed_frames, threshold, max_frame_window=14, order=1):
    # max_frame_window: The pair swapping issue appears as follows:
    # fish ids are swapped and the new assignment remains for a certain number of frames
    # then, after e.g. 10-15 frames the ids revert back to their original ones.
    # this scripts finds not only the frames with swaps but also removed the in-between frames where the ids have been swapped.
    # the max_frame_window parameter is a threshold on the maximum size of the number of in-between frames.

    new_tracks_3D = np.copy(tracks)
    diff = np.ones(tracks_imCoords.shape) * np.NaN

    for i_frame in range(order, tracks.shape[0]):
        diff[:, i_frame, :, :, :] = tracks_imCoords[:, i_frame, :, :, :] - tracks_imCoords[:, i_frame - order, :, :, :]
    vel = np.linalg.norm(diff, axis=4)

    vel_mask = vel[:, :, :, :] > threshold

    # both fishes jump, for any bodypoint, in any of the cameras

    # any body point jumps
    bd_jump = np.any(vel_mask, axis=(3))
    # for both fishes
    fish_jump = np.all(bd_jump, axis=(2))
    # in any of the cameras
    fish_any_camera_jump = np.any(fish_jump, axis=(0))
    # find the indexes
    both_fishes_jump_index = np.where(fish_any_camera_jump)[0]


    frames_diff_all = []

    for i in range(1, len(both_fishes_jump_index)):
        frames_diff = both_fishes_jump_index[i] - both_fishes_jump_index[i - 1] + 1

        if frames_diff <= max_frame_window:
            begin_frame = both_fishes_jump_index[i - 1]
            end_frame = both_fishes_jump_index[i] + 1
            frames_diff_all.append(frames_diff)
            new_tracks_3D[begin_frame:end_frame] = np.NaN
            removed_frames[begin_frame:end_frame] = 1

    total_removed_frames = sum(frames_diff_all)
    print("Removed {0} frames with pair swap in image coordinates".format(total_removed_frames))

    return new_tracks_3D, removed_frames


def fix_sudden_jumps(tracks_3D, tracks_imCoords, threshold):
    diff = np.ones(tracks_imCoords.shape) * np.NaN
    frames_removed = np.zeros((len(tracks_3D), 1))

    for i_frame in range(1, tracks_3D.shape[0]):
        diff[:, i_frame, :, :, :] = tracks_imCoords[:, i_frame, :, :, :] - tracks_imCoords[:, i_frame - 1, :, :, :]
    vel = np.linalg.norm(diff, axis=4)

    vel_mask = vel[:, :, :, :] > threshold

    # any fish jump, for any bodypoints, in any of the cameras
    any_fish_jump_index = np.where(np.any(np.any(vel_mask, axis=(2, 3)), axis=0))[0]

    frames_diff_all = []
    removed_frames_diff_all = []

    for i in range(0, len(any_fish_jump_index)):
        frame = any_fish_jump_index[i]

        tracks_3D[frame] = np.NaN
        frames_removed[frame] = 1

    total_removed_frames = len(any_fish_jump_index)
    print("Removed {0} frames with sudden jumps in image coordinates".format(total_removed_frames))

    return tracks_3D, total_removed_frames, frames_removed


def fix_sudden_jumps_in_positions(tracks_3D, threshold, dt, order=1):
    frames_removed = np.zeros((len(tracks_3D), 1))
    vel = get_order_velocity(tracks_3D, order, dt)
    vel_mask = vel[:, :, :] > threshold

    # both fish jump, for any bodypoints
    both_fish_jump_index = np.where(np.all(np.any(vel_mask, axis=(2)), axis=(1)))[0]

    for i in range(0, len(both_fish_jump_index)):
        frame = both_fish_jump_index[i]

        tracks_3D[frame] = np.NaN
        frames_removed[frame] = 1

    total_removed_frames = len(both_fish_jump_index)
    print("Removed {0} frames with sudden jumps in position coordinates ({1} order)".format(total_removed_frames, order))

    return tracks_3D, vel, total_removed_frames, frames_removed




def fix_single_fish_jumps_in_positions(tracks, removed_frames, vel, threshold, order=1):
    vel_mask = vel[:, :, :] > threshold

    # any fish jump, for any bodypoints
    any_fish_jump_index = np.where(np.any(np.any(vel_mask, axis=(2)), axis=(1)))[0]

    for i in range(0, len(any_fish_jump_index)):
        frame = any_fish_jump_index[i]

        tracks[frame] = np.NaN
        removed_frames[frame] = 1

    total_removed_frames = len(any_fish_jump_index)
    print("Removed {0} frames with sudden jumps in position coordinates ({1} order)".format(total_removed_frames, order))

    return tracks, removed_frames





def get_order_velocity(tracks_3D, order, dt):
    diff = np.ones(tracks_3D.shape) * np.NaN
    frames_removed = np.zeros((len(tracks_3D), 1))
    print("ok")
    for i_frame in range(order, tracks_3D.shape[0]):
        diff[i_frame, :, :, :] = tracks_3D[i_frame, :, :, :] - tracks_3D[i_frame - order, :, :, :]
    #vel = np.linalg.norm(diff, axis=3)
    vel = diff/(order*dt)
    return vel
