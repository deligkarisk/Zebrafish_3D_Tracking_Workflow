import numpy as np
from matplotlib import pyplot as plt


def plot_registration_costs(frame_registration_costs_bp_mean, mean_reg_skel_thresh=None, max_x=100):
    # ---- collapse to a single array of costs ---- #
    reg_costs = frame_registration_costs_bp_mean.reshape(-1)

    fig, axs = plt.subplots()
    ax = axs

    histbins = np.linspace(0, max_x, num=100)
    ax.hist(reg_costs, label='head', alpha=0.5, density=True, bins=histbins)

    ax.set_title('Histogram of skeleton registration costs for correctly matched skeletons')
    ax.set_xlim(0, max_x)
    ax.set_xlabel('registration cost')
    ax.set_ylabel('density')
    if mean_reg_skel_thresh is not None:
        ax.axvline(mean_reg_skel_thresh, label='mean_reg_skel_thresh', color='black')
    return fig



def get_mean_body_point_cost(frame_registration_costs, numFish):
    frame_registration_costs_bp_mean = np.ones((frame_registration_costs.shape[0], numFish)) * np.NaN

    for fIdx in range(frame_registration_costs.shape[0]):
        for fishIdx in range(numFish):
            bp_costs = frame_registration_costs[fIdx, fishIdx]
            numBps_used = np.count_nonzero(~np.isnan(bp_costs))
            if numBps_used >= 2:
                frame_registration_costs_bp_mean[fIdx, fishIdx] = np.nanmean(bp_costs)

    return frame_registration_costs_bp_mean
