"""
3D Fish Tracking Experiment

This script processes SLEAP and idTracker.ai data to create 3D tracks of fish movement.

The workflow includes:
1. Cross-camera registration of SLEAP detections
2. Applying registration and skeleton size thresholds
3. Matching idTracker.ai identities with 3D skeletons
4. Tracking segments in 3D
5. Post-processing trajectories (cleaning, interpolation, smoothing)
6. Saving results to H5 and CSV files

Usage:
    python track_experiment.py <expName> <mean_reg_skel_thresh> <idtracks_to_sleap_thresh> <start_frame_in_video> [options]

Arguments:
    expName: Name of the experiment
    mean_reg_skel_thresh: Threshold for skeleton registration
    idtracks_to_sleap_thresh: Threshold for matching idTracker.ai with SLEAP
    start_frame_in_video: Start frame in the video
"""

import argparse
import os
import time
from multiprocessing import Pool, RawArray

import h5py
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.cbook import contiguous_regions
from scipy.optimize import linear_sum_assignment

from lib.calibration.calibration import Calibration
from lib.registration.skeleton_registration_costs import plot_registration_costs
from lib.sleap_idtracker_merge.plot_utils import plot_idtracker_sleap_assignment_costs, plot_bodypoints_distances
from lib.sleap_idtracker_merge.post_processing import interpolate_over_small_gaps, \
    get_smooth_timeseries_and_derivatives_using_savgol
from lib.sleap_idtracker_merge.track_3d import track_segment_in_3D_if_possible
from lib.sleap_idtracker_merge.various import prepend_nan_to_idtracker_data, prepend_nan_to_results_array, \
    prepend_nan_to_im_coords_array
from lib.tracking.data_cleaning import clean_data_from_jumps
from lib.tracking.frame_registration import find_mean_bodypoint_registration_costs_array
from lib.tracking.io import save_tracks_3D_to_csv_and_return_dataFrame
from lib.tracking.track_utils import create_array_of_start_stop_frames_for_parallelization
from lib.tracking.tracking_functions import get_3D_positions_from_sleap_imcoords
from lib.tracking_quality.tracking_quality import get_percentage_frames_both_fish_full_info, \
    get_percentage_frames_no_info, get_percentage_frames_no_idtracker_info
from lib.various.filesystem import find_calibration_folder
from lib.various.geometric import get_distances_from_locations_array

# Constants for paths
BUCKET_BASE_PATH = "/bucket/StephensU/fish3D"
RESULTS_TRACKING_PATH = "./results/tracking"
RESULTS_INFERENCE_PATH = "./results/inference"

# -------------------- Parse Arguments -------------------- #
parser = argparse.ArgumentParser()
parser.add_argument("expName", type=str)
parser.add_argument("mean_reg_skel_thresh", type=float)
parser.add_argument("idtracks_to_sleap_thresh", type=float)
parser.add_argument("start_frame_in_video", type=int)
parser.add_argument("--head_pec_thresh", type=float, default=1.4)
parser.add_argument("--pec_tail_thresh", type=float, default=2.5)
parser.add_argument("--pair_swap_velocity_threshold", type=float, default=30)
parser.add_argument("--parStep", type=int, default=1000)
parser.add_argument("--numProcessors", type=int, default=20)
parser.add_argument("--interp_polyOrd", type=int, default=1)
parser.add_argument("--interp_limit", type=int, default=7)
parser.add_argument("--savgol_win", type=int, default=9)
parser.add_argument("--savgol_ord", type=int, default=2)
parser.add_argument("--fps", type=int, default=140)
args = parser.parse_args()

print()
print('------- Tracking {0} -------'.format(args.expName))
print()

print('-- inputs --')
print('expName: {0}'.format(args.expName))
print('parStep: {0}'.format(args.parStep))
print('numProcessors: {0}'.format(args.numProcessors))
print('mean_reg_skel_thresh: {0}'.format(args.mean_reg_skel_thresh))
print('head_pec_thresh: {0}'.format(args.head_pec_thresh))
print('pec_tail_thresh: {0}'.format(args.pec_tail_thresh))
print('idtracks_to_sleap_thresh: {0}'.format(args.idtracks_to_sleap_thresh))
print('interp_polyOrd: {0}'.format(args.interp_polyOrd))
print('interp_limit: {0}'.format(args.interp_limit))
print('savgol_win: {0}'.format(args.savgol_win))
print('savgol_ord: {0}'.format(args.savgol_ord))
print('fps: {0}'.format(args.fps))
print('start frame in video: {0}'.format(args.start_frame_in_video))
print('pair swap velocity threshold: {0}'.format(args.pair_swap_velocity_threshold))
print()

# -------------------- Setup Paths and Parameters -------------------- #
input_path = os.path.join(RESULTS_TRACKING_PATH, "sleap_and_idtracker_data.h5")
saveFile_h5_path = os.path.join(RESULTS_TRACKING_PATH, "full_output.h5")
saveFile_csv_path = os.path.join(RESULTS_TRACKING_PATH, "tracks.csv")
saveFile_aligned_h5 = os.path.join(RESULTS_TRACKING_PATH, "tracks.h5")
prepend_results_path = RESULTS_TRACKING_PATH
dt = 1 / args.fps

main_recordings_folder = os.path.join(BUCKET_BASE_PATH, args.expName)
calibration_sub_folder_path = find_calibration_folder(main_recordings_folder)
calibration_folder_path = os.path.join(calibration_sub_folder_path, "auto_calibration_results")

movie_filepath_xy = os.path.join(main_recordings_folder, "xy", "fishfight.mp4")
movie_filepath_xz = os.path.join(main_recordings_folder, "xz", "fishfight.mp4")
movie_filepath_yz = os.path.join(main_recordings_folder, "yz", "fishfight.mp4")
movie_filepaths = [movie_filepath_xz, movie_filepath_xy, movie_filepath_yz]

# -------------------- Load Tracking Data -------------------- #

with h5py.File(input_path, 'r') as hf:
    sleap_data = hf['sleap_data'][:]
    idtracker_data = hf['idtracker_data'][:]

# sleap_data = sleap_data[:, 0:260000]

# parse some shapes
numCams, numFrames, numFish, numBodyPoints, coordinates = sleap_data.shape
print("The shape of SLEAP data is: {0}".format(sleap_data.shape))
print("The shape of idtracker data is: {0}".format(idtracker_data.shape))

# -------------------- Initialize Calibration -------------------- #
cal = Calibration(calibration_folder_path)

# -------------------- Setup Parallelization -------------------- #

# parse frames up into chunks
parallelization_start_stop_frms = create_array_of_start_stop_frames_for_parallelization(numFrames,
                                                                                        step=args.parStep)

# the list of jobIdxs to map over
job_idxs = [i for i in range(parallelization_start_stop_frms.shape[0])]


##############################

# We are going to use a trick with a global dictionary
# to allow multiple processes to access the same array in parallel.
# This does require some uglyness/boilerplate,
# which is contained below


# First, define an instrically parallel function for mapping over,
# to cross calibrate different chunks of frames in parallel.
# This function will want access to the global dictionary

def get_imcoord_positions_and_3D_positions_parallel(i):
    ''' Perform the cross camera registration for a section of frames.

        NB: THIS FUNCTION IS INTRINSICALLY PARALLEL
            It needs to be run with the below parallel boilerplate code.

        See: register_sleap_instances() in the lib file,
             for a non-parallel version of this function, which can be used
             without the global var_dict

    --- args ---
    i: int, an index used for parsing lists and global arrays to get the data
       for this particular process

    --- returns ---
    frame_methodIdxs                 : array, shape (numFrames,),
                                       the methodIdx of the method used to solve the frame
    frame_positions_imageCoordinates : array, shape (numCams, numFrames, numFish, numBodyPoints, 2),
                                       the image coordinates of the skeletons registered across cams
    frame_positions_3DBps            : array, shape (numFrames, numFish, numBodyPoints, 3),
                                       the 3D coordinates of the skeletons registered across cams
    frame_registration_costs         : array, shape (numFrames, numFish, numBodyPoints),
                                       the cross-camera registration costs for the frame
    '''
    # make the calibration object
    calOb = Calibration(var_dict['calibrationFolderPath'])

    # get numpy versions of the idTracker data
    complete_sleap_data = np.frombuffer(var_dict['sleap_data']).reshape(var_dict['sleap_data_shape'])

    # parse the inputs
    numCams, numFrames, numFish, numBodyPoints, _ = complete_sleap_data.shape

    # make the parsing array
    parStep = var_dict['parStep']
    start_stop_frms = create_array_of_start_stop_frames_for_parallelization(numFrames, step=parStep)

    # get the data for this jobID
    jobF0, jobFE = start_stop_frms[i]
    jobNumFrames = jobFE - jobF0
    job_sleap_data = complete_sleap_data[:, jobF0:jobFE]

    #  ----- preallocate  -------#
    frame_methodIdxs = np.ones((jobNumFrames,)) * np.NaN
    frame_positions_imageCoordinates = np.ones((numCams, jobNumFrames, numFish, numBodyPoints, 2)) * np.NaN
    frame_positions_3DBps = np.ones((jobNumFrames, numFish, numBodyPoints, 3)) * np.NaN
    frame_registration_costs = np.ones((jobNumFrames, numFish, numBodyPoints)) * np.NaN

    for fIdx in range(jobNumFrames):
        # get the frame loaded_imcoords
        frame_instances = np.copy(job_sleap_data[:, fIdx, :, :, :])

        # register the skeletons across camera views
        methodIdx, \
            positions_imageCoordinates, \
            positions_3DBps, \
            positions_registration_costs, \
            debug_vals = get_3D_positions_from_sleap_imcoords(frame_instances, calOb, debug=False)

        # record
        frame_methodIdxs[fIdx] = methodIdx
        frame_positions_imageCoordinates[:, fIdx] = positions_imageCoordinates
        frame_positions_3DBps[fIdx] = positions_3DBps
        frame_registration_costs[fIdx] = positions_registration_costs

    # finish up
    return frame_methodIdxs, frame_positions_imageCoordinates, frame_positions_3DBps, frame_registration_costs


# Now make the global dictionary

# This is a formality to allow multiple processes to share access to arrays
sleap_data_RARR = RawArray('d', int(np.prod(sleap_data.shape)))
sleap_data_np = np.frombuffer(sleap_data_RARR).reshape(sleap_data.shape)
np.copyto(sleap_data_np, sleap_data)

# A global dictionary storing the variables passed from the initializer
# This dictionary holds variables that each process will share access to, instead of making copies
var_dict = {}


# This function initializes the shared data in each job process
def init_worker(sleap_data, sleap_data_shape, calibrationFolderPath, parStep, mean_reg_skel_thresh):
    var_dict['sleap_data'] = sleap_data
    var_dict['sleap_data_shape'] = sleap_data_shape
    var_dict['calibrationFolderPath'] = calibrationFolderPath
    var_dict['parStep'] = parStep
    var_dict['mean_reg_skel_thresh'] = mean_reg_skel_thresh


##############################


# -------------------- Perform Cross-Camera Registration -------------------- #

t0 = time.time()

print()
print('Launching cross-camera registration...')

# map the function
with Pool(processes=args.numProcessors, initializer=init_worker,
          initargs=(sleap_data_RARR,
                    sleap_data.shape,
                    calibration_folder_path,
                    args.parStep,
                    args.mean_reg_skel_thresh)) as pool:
    outputs = pool.map(get_imcoord_positions_and_3D_positions_parallel, job_idxs)

#  ---- parse the output ---- #
methodIdxs = []
positions_imageCoordinates = []
positions_3D = []
registration_costs = []

for job_results in outputs:
    job_frame_methodIdxs, job_frame_positions_imageCoordinates, job_frame_positions_3DBps, job_frame_registration_costs = job_results
    methodIdxs.append(job_frame_methodIdxs)
    positions_imageCoordinates.append(job_frame_positions_imageCoordinates)
    positions_3D.append(job_frame_positions_3DBps)
    registration_costs.append(job_frame_registration_costs)

methodIdxs = np.concatenate(methodIdxs)
positions_imageCoordinates = np.concatenate(positions_imageCoordinates, axis=1)
positions_3D = np.concatenate(positions_3D, axis=0)
registration_costs = np.concatenate(registration_costs)

# ----- finish up -----#
print()
print(f'Cross-camera registration completed in {time.time() - t0:.2f} seconds')

# -- preallocate main outputs -- #

positions_imageCoordinates_processed = np.copy(positions_imageCoordinates)
positions_3D_processed = np.copy(positions_3D)

tracks_3D = np.ones_like(positions_3D_processed) * np.NaN
tracks_imCoords = np.ones_like(positions_imageCoordinates_processed) * np.NaN

# -------- Step 1 ------------#
# Apply the reg_thresh to positions

print('Applying the registration threshold...')

# find the registration costs averaged along bodypoints
registration_costs_mean_bp = find_mean_bodypoint_registration_costs_array(registration_costs)

# find a mask, which shows 'True' frame-fish pairs that don't pass the threshold
bad_registration_mask = registration_costs_mean_bp > args.mean_reg_skel_thresh

# use the mask to NaN positions which don't pass the threshold
# (reg_thresh_removal_info will keep track of which fish in which frame we discard)
reg_thresh_removal_info = []
for fIdx in range(positions_3D.shape[0]):
    for fishIdx in range(positions_3D.shape[1]):
        if bad_registration_mask[fIdx, fishIdx]:
            reg_thresh_removal_info.append([fIdx, fishIdx])
            positions_imageCoordinates_processed[:, fIdx, fishIdx, :, :] = np.NaN
            positions_3D_processed[fIdx, fishIdx, :, :] = np.NaN
reg_thresh_removal_info = np.array(reg_thresh_removal_info)

tE = time.time()
print(f'Registration threshold applied in {tE - t0:.2f} seconds')

# save cross-registration costs as a figure (only for frames where fish are distant)
distances = get_distances_from_locations_array(sleap_data, 0)
index_xz = distances[0, :] > 100
index_xy = distances[1, :] > 100
index_yz = distances[2, :] > 100

long_distances_index = np.logical_and(np.logical_and(index_xz, index_xy), index_yz)
all_registration_costs_mean_distant = registration_costs_mean_bp[long_distances_index]
fig = plot_registration_costs(all_registration_costs_mean_distant, max_x=40,
                              mean_reg_skel_thresh=args.mean_reg_skel_thresh)
plt.savefig(os.path.join(prepend_results_path, 'all_registration_costs_100_distant.jpg'), dpi=300)
plt.show()

# -------- Step 2 ------------#
# apply the size thresholds to positions

print('Applying the skeleton 3D size threshold...')

# Calculate the sizes
head_pec_dists = np.ones((numFrames, numFish)) * np.NaN
pec_tail_dists = np.ones((numFrames, numFish)) * np.NaN
for fishIdx in range(numFish):
    for fIdx in range(numFrames):
        fishData = np.copy(positions_3D_processed[fIdx, fishIdx])
        head_pec_dists[fIdx, fishIdx] = np.linalg.norm(fishData[0] - fishData[1])
        pec_tail_dists[fIdx, fishIdx] = np.linalg.norm(fishData[1] - fishData[2])

# remove any fish that are too big
size_thresh_removal_info = []
for fIdx in range(positions_3D.shape[0]):
    for fishIdx in range(positions_3D.shape[1]):
        if head_pec_dists[fIdx, fishIdx] > args.head_pec_thresh:
            size_thresh_removal_info.append([fIdx, fishIdx, 0])
            positions_imageCoordinates_processed[:, fIdx, fishIdx, :, :] = np.NaN
            positions_3D_processed[fIdx, fishIdx, :, :] = np.NaN
        elif pec_tail_dists[fIdx, fishIdx] > args.pec_tail_thresh:
            size_thresh_removal_info.append([fIdx, fishIdx, 1])
            positions_imageCoordinates_processed[:, fIdx, fishIdx, :, :] = np.NaN
            positions_3D_processed[fIdx, fishIdx, :, :] = np.NaN
        else:
            continue
size_thresh_removal_info = np.array(size_thresh_removal_info)
frames_removed_size_threshold = len([element[0] for element in size_thresh_removal_info])
print(f"Total frames removed due to fish size threshold: {frames_removed_size_threshold}")

# plot stats for head-pec and pec-tail distances
fig = plot_bodypoints_distances(head_pec_dists, args.head_pec_thresh, title="Head-pec distances")
plt.savefig(os.path.join(prepend_results_path, 'head_pec_distances.jpg'), dpi=300)
fig = plot_bodypoints_distances(pec_tail_dists, args.pec_tail_thresh, title="Pec-tail distances")
plt.savefig(os.path.join(prepend_results_path, 'pec_tail_distances.jpg'), dpi=300)

tE = time.time()
print(f'Size threshold applied in {tE - t0:.2f} seconds')

# -------- Step 3 ------------#
# Match idTracker info with 3D skeletons to start the trajectories array
# In this step, we only work with frames where we have pec points and idtracks for all fish


print('Performing Hungarian matching of 3D skeletons with idTracker.ai identities...')

# make an array to hold information on if data has been assigned after this step
tracks_available_post_initial_id_assignments = np.zeros((numFrames, numFish), dtype=int)

# find frames with both idTracker.ai centroids for all fish, and xy cam pec image coordinates for both fish
fIdxs_with_idtracker_info_for_all_fish = np.where(~np.any(np.isnan(idtracker_data), axis=(1, 2)))[0]

xyCamIdx = 1
pecBpIdx = 1
fIdxs_with_xyPec_info_for_all_fish = \
np.where(~np.any(np.isnan(positions_imageCoordinates_processed[xyCamIdx, :, :, pecBpIdx, :]),
                 axis=(1, 2))
         )[0]

fIdxs_with_idtracker_and_xyPec_info_for_all_fish = np.intersect1d(fIdxs_with_idtracker_info_for_all_fish,
                                                                  fIdxs_with_xyPec_info_for_all_fish)

# do the idtracker.ai <-> sLEAP matching to assign identities for these frame
# Note: registrations must pass the threshold

# this array will keep track of costs of assignments that we make
idtracker_sleap_assignment_costs = np.zeros(
    (fIdxs_with_idtracker_and_xyPec_info_for_all_fish.shape[0], numFish)) * np.NaN
idtracker_sleap_assignment_nothresh_costs = np.zeros(
    (fIdxs_with_idtracker_and_xyPec_info_for_all_fish.shape[0], numFish)) * np.NaN

# test all frames with idtracks and xypecs
for ii, fIdx in enumerate(fIdxs_with_idtracker_and_xyPec_info_for_all_fish):

    # get the image coordinates of interest
    frame_pec_points = positions_imageCoordinates_processed[xyCamIdx, fIdx, :, pecBpIdx, :]
    frame_idTracker_centroids = idtracker_data[fIdx, :]

    # create the cost matrix for the hungarian sorting
    frame_cost_mat = np.zeros((numFish, numFish))
    for idtracker_idx in range(numFish):
        for sleap_idx in range(numFish):
            frame_cost_mat[idtracker_idx, sleap_idx] = np.linalg.norm(frame_idTracker_centroids[idtracker_idx] -
                                                                      frame_pec_points[sleap_idx])

    # sort the cost matrix record the identity assignment for this frame
    row_ind, col_ind = linear_sum_assignment(frame_cost_mat)

    # record the identity assignment, and cost, for this frame if they past the threshold
    for fishIdx in range(numFish):

        id_assignment_cost = frame_cost_mat[row_ind, col_ind][fishIdx]
        # record this, for debugging purposes
        idtracker_sleap_assignment_nothresh_costs[ii, fishIdx] = id_assignment_cost

        if id_assignment_cost < args.idtracks_to_sleap_thresh:

            # update the tracks arrays
            tracks_3D[fIdx, fishIdx] = np.copy(positions_3D_processed[fIdx, col_ind[fishIdx]])
            tracks_imCoords[:, fIdx, fishIdx] = np.copy(positions_imageCoordinates_processed[:, fIdx, col_ind[fishIdx]])
            # record some other data
            tracks_available_post_initial_id_assignments[fIdx, fishIdx] = 1
            idtracker_sleap_assignment_costs[ii, fishIdx] = id_assignment_cost

        else:
            continue

fig = plot_idtracker_sleap_assignment_costs(idtracker_sleap_assignment_nothresh_costs,
                                            threshold=args.idtracks_to_sleap_thresh)
plt.savefig(os.path.join(prepend_results_path, 'idtracker_sleap_assignment_costs.jpg'), dpi=300)

tE = time.time()
print(f'Hungarian matching completed in {tE - t0:.2f} seconds')

# -------- Step 4 ------------#
# Match idTracker info with 3D skeletons in all other frames where possible.
# In these frames, we cant do a hungarian sorting, because we don't 2 xypec points and 2 idtracks
# (by definition of these frame, they are the complement of the above frames).
# So in these frames, we assign identities to any fish we can assignment passes the threshold


# t0 = time.time()


print('Matching individual 3D skeletons with idTracker.ai identities...')

# find the indexs of frames we havent looked at yet
# (all frame numbers less (set less) the frame numbers we looked at already)
fIdxs_withOUT_idtracker_and_xyPec_info_for_all_fish = np.setdiff1d(np.arange(0, numFrames, 1),
                                                                   fIdxs_with_idtracker_and_xyPec_info_for_all_fish)

# make an array to hold information on if data has been assigned after this step
# start with a copy of what we already have, and we will add to this copy here
tracks_available_post_pass2_id_assignments = np.copy(tracks_available_post_initial_id_assignments)

# -- test all frames with idtracks and xypecs -- #

# this list will hold [fIdx, fishIdx, costs] values for
# where we add data to the tracks arrays in this pass
pass2_added_data_info_from_ids = []
pass2_added_data_info_from_assigning_remaining_detections = []

for ii, fIdx in enumerate(fIdxs_withOUT_idtracker_and_xyPec_info_for_all_fish):

    # get the image coordinates of interest
    frame_pec_points = positions_imageCoordinates_processed[xyCamIdx, fIdx, :, pecBpIdx, :]
    frame_idTracker_centroids = idtracker_data[fIdx, :]

    # test idtracks to see if we can make matches with pec points
    for fishIdx in range(numFish):

        idtrack_centroid = frame_idTracker_centroids[fishIdx]

        # count the number of xyPec points that pass the threshold with this idtrack coord
        candidate_match_idxs = []
        candidate_match_costs = []
        for dummyFishIdx in range(numFish):
            pec_point = frame_pec_points[dummyFishIdx]
            assign_cost = np.linalg.norm(idtrack_centroid - pec_point)
            if assign_cost < args.idtracks_to_sleap_thresh:
                candidate_match_idxs.append(dummyFishIdx)
                candidate_match_costs.append(assign_cost)
            else:
                continue

        # if we have exactly one match, use it.
        # update the tracks, then try to assign ids to image_cooridinate positions left over
        if len(candidate_match_idxs) == 1:
            # update the tracks arrays
            tracks_3D[fIdx, fishIdx] = np.copy(positions_3D_processed[fIdx, candidate_match_idxs[0]])
            tracks_imCoords[:, fIdx, fishIdx] = np.copy(
                positions_imageCoordinates_processed[:, fIdx, candidate_match_idxs[0]])
            # record some other data
            tracks_available_post_pass2_id_assignments[fIdx, fishIdx] = 1
            pass2_added_data_info_from_ids.append([fIdx, fishIdx, candidate_match_costs[0]])

            # Is there a 3D skeleton for this frame that doesn't have an identity from idtracker.ai?
            # If there is, since we just assigned an identity to one individual,
            # then this 3D skeleton can get the identity of the remaining fish.
            index_used_to_assign_id = candidate_match_idxs[0]
            other_index = np.mod(candidate_match_idxs[0] + 1, 2)  # NB: only works for numFish=2
            skeleton_just_used = np.copy(positions_3D_processed[fIdx, index_used_to_assign_id])
            other_skeleton = np.copy(positions_3D_processed[fIdx, other_index])
            # get the index of the other individual we are trying to assign data to now
            other_individual_fishIdx = np.mod(fishIdx + 1, 2)  # NB: only works for numFish=2

            # check that we don't already have something in tracks_3D for this individual
            if np.all(np.isnan(tracks_3D[fIdx, other_individual_fishIdx])):
                # check that the unidentified (remaining) 3D skeleton is not entirely empty
                if ~np.all(np.isnan(other_skeleton)):
                    tracks_3D[fIdx, other_individual_fishIdx] = np.copy(other_skeleton)
                    tracks_imCoords[:, fIdx, other_individual_fishIdx] = np.copy(
                        positions_imageCoordinates_processed[:, fIdx, other_index])
                    # record some other data (the cost is NaN since we have no idtracker identity)
                    tracks_available_post_pass2_id_assignments[fIdx, other_individual_fishIdx] = 1
                    pass2_added_data_info_from_assigning_remaining_detections.append(
                        [fIdx, other_individual_fishIdx, np.NaN])

        else:
            continue

pass2_added_data_info_from_ids = np.array(pass2_added_data_info_from_ids)
pass2_added_data_info_from_assigning_remaining_detections = np.array(
    pass2_added_data_info_from_assigning_remaining_detections)

tE = time.time()
print(f'Individual matching completed in {tE - t0:.2f} seconds')

# -------- Step 5 ------------#

# Fill-in gaps in trajectories by propagating idx in 3D,
# only accepting cases where we get it right

print('Tracking segments in time with 3D skeletons but without idtracker.ai matches...')

# find contiguous regions that don't have tracks available for both fish
frames_without_numFish_3D_tracks_yet = np.zeros((numFrames,))
frames_without_numFish_3D_tracks_yet[np.sum(tracks_available_post_pass2_id_assignments, axis=1) != numFish] = 1
regions_without_numFish_3D_tracks_yet = contiguous_regions(frames_without_numFish_3D_tracks_yet)

# ---- track in 3D ------- #
# prepare containers to grab results
successful_tracking = []

for regIdx in range(len(regions_without_numFish_3D_tracks_yet)):

    # prepare the args
    regF0, regFE = regions_without_numFish_3D_tracks_yet[regIdx]

    # Test:
    # If regF0=0, that means the first region starts at the first frame,
    # i.e. we have no idtracker.ai results to start us off.
    # We do not have a concept of 'start from known positions and going towards known positions'
    # for this region, so we cannot track it, hence we skip it
    if regF0 == 0:
        was_successfull = False
        successful_tracking.append(was_successfull)
        continue
    # Test End:

    # Test:
    # If regFE=tracks_3D.shape[0], that means we don't have last known positions contained within the experiment data
    # So we will skip this region
    if regFE == tracks_3D.shape[0]:
        was_successfull = False
        successful_tracking.append(was_successfull)
        continue
    # Test End:

    existing_tracks_3D_for_segment = np.copy(tracks_3D[regF0:regFE + 1])
    existing_imCoords_3D_for_segment = np.copy(tracks_imCoords[:, regF0:regFE + 1])

    positions_3D_processed_for_segment = np.copy(positions_3D_processed[regF0:regFE + 1])
    positions_imageCoordinates_processed_for_segment = np.copy(positions_imageCoordinates_processed[:, regF0:regFE + 1])

    last_known_positions = np.copy(tracks_3D[regF0 - 1])
    final_known_positions = np.copy(tracks_3D[regFE])

    tracks_available_post_pass2_id_assignments_for_segment = np.copy(
        tracks_available_post_pass2_id_assignments[regF0:regFE + 1])

    # Track segment
    track_outs = track_segment_in_3D_if_possible(existing_tracks_3D_for_segment,
                                                 existing_imCoords_3D_for_segment,
                                                 positions_3D_processed_for_segment,
                                                 positions_imageCoordinates_processed_for_segment,
                                                 last_known_positions,
                                                 final_known_positions,
                                                 tracks_available_post_pass2_id_assignments_for_segment)
    # parse the output
    [was_successfull, segment_tracks_3D, segment_tracks_imCoords, seg_track_method,
     seg_data_available_array] = track_outs

    # update tracks 3D if we can
    if was_successfull:
        # the :-1 is to remove the final frame from the segment_tracks, which is one
        # longer than the segment itself, since we included the frame with the
        # final_known_position information
        tracks_3D[regF0:regFE] = np.copy(segment_tracks_3D[:-1])
        tracks_imCoords[:, regF0:regFE] = np.copy(segment_tracks_imCoords[:, :-1])

    # record outputs
    successful_tracking.append(was_successfull)

successful_tracking = np.array(successful_tracking)

# -------- Step 6 ------------#
#  post process the trajectories, and save
# post processing
print()
print('Post-processing trajectories...')

tracks_3D_tracked = np.copy(tracks_3D)

# Clean up pair swaps, sudden jumps etc...
tracks_3D, pair_switch_removed_frames, velocity_outlier_removed_frames = clean_data_from_jumps(tracks_3D,
                                                                                               tracks_imCoords, dt,
                                                                                               threshold=args.pair_swap_velocity_threshold)

# interpolate
tracks_3D_interpd = interpolate_over_small_gaps(tracks_3D,
                                                limit=args.interp_limit,
                                                polyord=args.interp_polyOrd)
# use sav-gol filter
outs = get_smooth_timeseries_and_derivatives_using_savgol(tracks_3D_interpd,
                                                          win_len=args.savgol_win,
                                                          polyOrd=args.savgol_ord,
                                                          dt=dt)
tracks_3D_smooth, tracks_3D_vel_smooth, tracks_3D_speed_smooth, tracks_3D_accvec_smooth, tracks_3D_accmag_smooth = outs
print('Post-processing completed successfully')

# save the results, as they are, unsynchronized with video
print('Saving results to H5 file...')
with h5py.File(saveFile_h5_path, 'w') as hf:
    hf.create_dataset('tracks_3D_tracked', data=tracks_3D_tracked)
    hf.create_dataset('tracks_3D_raw', data=tracks_3D)

    hf.create_dataset('tracks_imCoords_raw', data=tracks_imCoords)
    hf.create_dataset('tracks_3D_smooth', data=tracks_3D_smooth)

    # save debugging info
    hf.create_dataset('methodIdxs', data=methodIdxs)
    hf.create_dataset('positions_imageCoordinates', data=positions_imageCoordinates)
    hf.create_dataset('registration_costs', data=registration_costs)
    hf.create_dataset('reg_thresh_removal_info', data=reg_thresh_removal_info)
    hf.create_dataset('size_thresh_removal_info', data=size_thresh_removal_info)
    hf.create_dataset('tracks_available_post_initial_id_assignments', data=tracks_available_post_initial_id_assignments)
    hf.create_dataset('pass2_added_data_info_from_ids', data=pass2_added_data_info_from_ids)
    hf.create_dataset('pass2_added_data_info_from_assigning_remaining_detections',
                      data=pass2_added_data_info_from_assigning_remaining_detections)
    hf.create_dataset('successful_tracking', data=successful_tracking)

    hf.create_dataset('pair_swap_removed_frames', data=pair_switch_removed_frames)
    hf.create_dataset('single_fish_jump_removed_frames', data=velocity_outlier_removed_frames)

    # extra stuff to save, if you like
    hf.create_dataset('idtracker_sleap_assignment_nothresh_costs', data=idtracker_sleap_assignment_nothresh_costs)
    hf.create_dataset('idtracker_data', data=idtracker_data)
    hf.create_dataset("registration_costs_mean_distant", data=all_registration_costs_mean_distant)
    hf.create_dataset("head_pec_dist", data=head_pec_dists)
    hf.create_dataset("pec_tail_dist", data=pec_tail_dists)

    # -- strings -- #
    string_type = h5py.special_dtype(vlen=str)

    tracks_dim_names = np.array(['numFrames', 'numFish', 'numBodyPoints', 'XYZ'], dtype=string_type)
    hf.create_dataset('tracks_dim_names', data=tracks_dim_names)

    savgol_info = np.array(['win_len={0}'.format(args.savgol_win),
                            'polyOrd={0}'.format(args.savgol_ord)], dtype=string_type)
    hf.create_dataset('savgol_info', data=savgol_info)

    ## NB: HARDCODED
    interp_info = np.array(['method=polynomial', 'order={0}'.format(args.interp_polyOrd),
                            'limit_direction=both', 'limit={0}'.format(args.interp_limit),
                            'inplace=False'], dtype=string_type)
    hf.create_dataset('interp_info', data=interp_info)
print('H5 file saved successfully')

# Log stats on tracking quality

print("Stats on tracking quality in smooth data:")
print("Frames with full SLEAP detections in camera 0: {0}%".format(
    get_percentage_frames_both_fish_full_info(sleap_data[0])))
print("Frames with full SLEAP detections in camera 1: {0}%".format(
    get_percentage_frames_both_fish_full_info(sleap_data[1])))
print("Frames with full SLEAP detections in camera 2: {0}%".format(
    get_percentage_frames_both_fish_full_info(sleap_data[2])))
print("Frames with no SLEAP detections in camera 0: {0}%".format(get_percentage_frames_no_info(sleap_data[0])))
print("Frames with no SLEAP detections in camera 1: {0}%".format(get_percentage_frames_no_info(sleap_data[1])))
print("Frames with no SLEAP detections in camera 2: {0}%".format(get_percentage_frames_no_info(sleap_data[2])))
print("Frames with no idtrackerid information: {0}%".format(get_percentage_frames_no_idtracker_info(idtracker_data)))
print("Tracked data: Frames with full information: {0}%".format(
    get_percentage_frames_both_fish_full_info(tracks_3D_tracked)))
print("Tracked data: Frames with no information: {0}%".format(get_percentage_frames_no_info(tracks_3D_tracked)))
print("Raw data: Frames with full information: {0}%".format(get_percentage_frames_both_fish_full_info(tracks_3D)))
print("Raw data: Frames with no information: {0}%".format(get_percentage_frames_no_info(tracks_3D)))
print("Smooth data: Frames with full information: {0}%".format(
    get_percentage_frames_both_fish_full_info(tracks_3D_smooth)))
print("Smooth data: Frames with no information: {0}%".format(get_percentage_frames_no_info(tracks_3D_smooth)))

# before saving the track data, append NaN to the data until the start_frame value.
# This has the benefit that it "synchronizes" the data generated with the full video, even
# if we did the tracking for only a part of the video (i.e. started tracking after X frames).

tracks_3D = prepend_nan_to_results_array(tracks_3D, args.start_frame_in_video)
tracks_3D_tracked = prepend_nan_to_results_array(tracks_3D_tracked, args.start_frame_in_video)
tracks_imCoords = prepend_nan_to_im_coords_array(tracks_imCoords, args.start_frame_in_video)
tracks_3D_smooth = prepend_nan_to_results_array(tracks_3D_smooth, args.start_frame_in_video)
idtracker_data = prepend_nan_to_idtracker_data(idtracker_data, args.start_frame_in_video)

# save the tracks only, this time they are aligned with the video
print('saving ... ')
with h5py.File(saveFile_aligned_h5, 'w') as hf:
    hf.create_dataset('tracks_3D_raw', data=tracks_3D)
    hf.create_dataset('tracks_imCoords_raw', data=tracks_imCoords)
    hf.create_dataset('tracks_3D_smooth', data=tracks_3D_smooth)
    hf.create_dataset('tracks_3D_tracked', data=tracks_3D_tracked)
    hf.create_dataset('idtracker_data', data=idtracker_data)

print('saved')
print()

tE = time.time()
print()
print('Finished post processing and saving h5 file')
print()

# -------------------- Finish Up and Save Results -------------------- #
tE = time.time()
print('saving csv file ...')
_ = save_tracks_3D_to_csv_and_return_dataFrame(tracks_3D_smooth, saveFile_csv_path)
print()
print('Finished!')
print("Exporting a sample video...")
print()
print('------------')
print('Completely finished!')
print('total time: t =  ', (tE - t0) / 60, ' mins')
