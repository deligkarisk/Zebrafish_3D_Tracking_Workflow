import numpy as np


def fill_in_bad_camera_view_image_coordinates(frame_instances, good_cams, calOb):
    ''' Given frame instances, and two chosen camera views, return a set of frame instances where
        the values of the image coordinates in the unchosen third camera are replaced by values
        calculated using the calibration object and the image coordinates from the chosen views.

    --- args ---
    frame_instances: (numCams=3, numFish=2, numBodyPoints=3, numImCoords=2)
    calOb: an instantiated calibration object
    good_cams: list of two idxs showing which cams to use.
               good_cam_pairings = [ [0,1], [0,2], [1,2] ]
               So, good_cams[0] = [0,1], meaning XZ and XY cameras

    --- returns ---
    filled_in_frame_instances: (numCams=3, numFish=2, numBodyPoints=3, numImCoords=2)
                               Frame instances with imagecoordinates for the unsed camera
                               view set by the calibration to be consistent with chosen views.

    '''
    # parse shapes
    _, numFish, numBodyPoints, _ = frame_instances.shape

    # preallocate an output array
    filled_in_frame_instances = np.copy(frame_instances)

    # find the bad cam idx
    if good_cams == [0, 1]:
        bad_cam_idx = 2
    elif good_cams == [1, 2]:
        bad_cam_idx = 0
    elif good_cams == [0, 2]:
        bad_cam_idx = 1

    if bad_cam_idx == 0:
        for fishIdx in range(numFish):
            for bpIdx in range(numBodyPoints):
                xy_imcoord = frame_instances[1, fishIdx, bpIdx]
                yz_imcoord = frame_instances[2, fishIdx, bpIdx]
                if ~np.all(np.isnan(xy_imcoord)) and ~np.all(np.isnan(yz_imcoord)):
                    xz_imcoord = calOb.compute_XZ_imcoords_from_XY_YZ(xy_imcoord, yz_imcoord)
                else:
                    xz_imcoord = np.array([np.NaN, np.NaN])
                filled_in_frame_instances[bad_cam_idx, fishIdx, bpIdx] = xz_imcoord

    elif bad_cam_idx == 1:
        for fishIdx in range(numFish):
            for bpIdx in range(numBodyPoints):
                xz_imcoord = frame_instances[0, fishIdx, bpIdx]
                yz_imcoord = frame_instances[2, fishIdx, bpIdx]
                if ~np.all(np.isnan(xz_imcoord)) and ~np.all(np.isnan(yz_imcoord)):
                    xy_imcoord = calOb.compute_XY_imcoords_from_XZ_YZ(xz_imcoord, yz_imcoord)
                else:
                    xy_imcoord = np.array([np.NaN, np.NaN])
                filled_in_frame_instances[bad_cam_idx, fishIdx, bpIdx] = xy_imcoord

    elif bad_cam_idx == 2:
        for fishIdx in range(numFish):
            for bpIdx in range(numBodyPoints):
                xz_imcoord = frame_instances[0, fishIdx, bpIdx]
                xy_imcoord = frame_instances[1, fishIdx, bpIdx]
                if ~np.all(np.isnan(xz_imcoord)) and ~np.all(np.isnan(xy_imcoord)):
                    yz_imcoord = calOb.compute_YZ_imcoords_from_XZ_XY(xz_imcoord, xy_imcoord)
                else:
                    yz_imcoord = np.array([np.NaN, np.NaN])
                filled_in_frame_instances[bad_cam_idx, fishIdx, bpIdx] = yz_imcoord

    return filled_in_frame_instances


def fill_in_bad_camera_view_image_coordinates_single_fish(fish_instances, good_cams, calOb):
    ''' Given frame instances for a single fish, and two chosen camera views, return a set of
        fish instances where the values of the image coordinates in the unchosen third camera
        are replaced by values calculated using the calibration object and the image coordinates
        from the chosen views.

    --- args ---
    fish_instances: (numCams=3, numBodyPoints=3, numImCoords=2)
    calOb: an instantiated calibration object
    good_cams: list of two idxs showing which cams to use.
               good_cam_pairings = [ [0,1], [0,2], [1,2] ]
               So, good_cams[0] = [0,1], meaning XZ and XY cameras

    --- returns ---
    filled_in_fish_instances: (numCams=3, numBodyPoints=3, numImCoords=2)
                               Frame instances for single fish with imagecoordinates for the
                               unsed camera view set by the calibration to be consistent with
                               chosen views.

    --- see also ---
    fill_in_bad_camera_view_image_coordinates() -> similar to this function, but it works on
                                                   the results for all fish from a particular
                                                   frame instead of just a single fish.
    '''
    # parse shapes
    _, numBodyPoints, _ = fish_instances.shape

    # preallocate an output array
    filled_in_fish_instances = np.copy(fish_instances)

    # find the bad cam idx
    if good_cams == [0, 1]:
        bad_cam_idx = 2
    elif good_cams == [1, 2]:
        bad_cam_idx = 0
    elif good_cams == [0, 2]:
        bad_cam_idx = 1

    if bad_cam_idx == 0:
        for bpIdx in range(numBodyPoints):
            xy_imcoord = fish_instances[1, bpIdx]
            yz_imcoord = fish_instances[2, bpIdx]
            if ~np.all(np.isnan(xy_imcoord)) and ~np.all(np.isnan(yz_imcoord)):
                xz_imcoord = calOb.compute_XZ_imcoords_from_XY_YZ(xy_imcoord, yz_imcoord)
            else:
                xz_imcoord = np.array([np.NaN, np.NaN])
            filled_in_fish_instances[bad_cam_idx, bpIdx] = xz_imcoord

    elif bad_cam_idx == 1:
        for bpIdx in range(numBodyPoints):
            xz_imcoord = fish_instances[0, bpIdx]
            yz_imcoord = fish_instances[2, bpIdx]
            if ~np.all(np.isnan(xz_imcoord)) and ~np.all(np.isnan(yz_imcoord)):
                xy_imcoord = calOb.compute_XY_imcoords_from_XZ_YZ(xz_imcoord, yz_imcoord)
            else:
                xy_imcoord = np.array([np.NaN, np.NaN])
            filled_in_fish_instances[bad_cam_idx, bpIdx] = xy_imcoord

    elif bad_cam_idx == 2:
        for bpIdx in range(numBodyPoints):
            xz_imcoord = fish_instances[0, bpIdx]
            xy_imcoord = fish_instances[1, bpIdx]
            if ~np.all(np.isnan(xz_imcoord)) and ~np.all(np.isnan(xy_imcoord)):
                yz_imcoord = calOb.compute_YZ_imcoords_from_XZ_XY(xz_imcoord, xy_imcoord)
            else:
                yz_imcoord = np.array([np.NaN, np.NaN])
            filled_in_fish_instances[bad_cam_idx, bpIdx] = yz_imcoord

    return filled_in_fish_instances
