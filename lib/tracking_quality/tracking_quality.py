import numpy as np


def get_percentage_frames_full_info(tracks):

    num_frames = len(tracks)

    fish_one_full_coordinates = np.isnan(tracks[:, 0, :, :])
    fish_one_full_coordinates = np.any(fish_one_full_coordinates, axis=(1, 2))
    fish_one_full_coordinates = num_frames - sum(fish_one_full_coordinates)
    fish_one_full_coordinates = round((fish_one_full_coordinates / num_frames) * 100, 2)

    fish_two_full_coordinates = np.isnan(tracks[:, 1, :, :])
    fish_two_full_coordinates = np.any(fish_two_full_coordinates, axis=(1, 2))
    fish_two_full_coordinates = num_frames - sum(fish_two_full_coordinates)
    fish_two_full_coordinates = round((fish_two_full_coordinates / num_frames) * 100, 2)

    return (fish_one_full_coordinates, fish_two_full_coordinates)


def get_percentage_frames_at_least_one_bp(tracks):
    num_frames = len(tracks)

    temp = ~np.isnan(tracks)
    temp = np.any(temp, axis=(2, 3))

    fish_one_at_least_one_bp = sum(temp[:, 0])
    fish_two_at_least_one_bp = sum(temp[:, 1])

    fish_one_at_least_one_bp = round((fish_one_at_least_one_bp / num_frames) * 100, 2)
    fish_two_at_least_one_bp = round((fish_two_at_least_one_bp / num_frames) * 100, 2)

    return fish_one_at_least_one_bp, fish_two_at_least_one_bp



def get_percentage_frames_both_fish_full_info(tracks):
   frames = np.sum(np.any(np.isnan(tracks), axis=(1,2,3)))
   frames = round(((len(tracks) - frames)/len(tracks))*100, 2)
   return frames

def get_percentage_frames_no_info(tracks):
    frames = np.sum(np.all(np.isnan(tracks), axis=(1, 2, 3)))
    return round((frames/len(tracks))*100, 2)



def get_percentage_frames_no_idtracker_info(idtracker_data):
    frames = np.sum(np.all(np.isnan(idtracker_data), axis=(1, 2)))
    return round((frames/len(idtracker_data))*100, 2)


