import numpy as np
from matplotlib import pyplot as plt


def plot_idtracker_sleap_assignment_costs(assignment_costs, threshold=None):

    fig, ax = plt.subplots()
    max_x = 30
    histbins = np.linspace(0, max_x, num=40)
    ax.hist(assignment_costs, label=['fish 1', 'fish 2'], alpha=0.5, density=True, bins=histbins)

    ax.set_title('Histogram of idtracker/sleap assignment costs')
    ax.set_xlim(0, max_x)
    ax.set_xlabel('idtracker/sleap assignment cost [pixels]')
    ax.set_ylabel('density')
    if threshold is not None:
        ax.axvline(threshold)
    plt.legend()
    return fig



def plot_bodypoints_distances(distances, threshold=None, max_x=4, title=None):

    # distances: ndarray of size (frames, num_fish), indicating the distance of specific
    # body points (e.g. head to pec) for each fish


    fig, ax = plt.subplots()
    histbins = np.linspace(0, max_x, num=200)
    ax.hist(distances, label=['fish 1', 'fish 2'], alpha=0.5, density=True, bins=histbins)
    ax.set_xlim(0, max_x)
    ax.set_xlabel("Distance [cm]")
    ax.set_ylabel("Density")
    if title is not None:
        ax.set_title(title)
    if threshold is not None:
        ax.axvline(threshold)
    plt.legend()
    return fig

def plot_pair_swap_velocity(velocities, threshold=None, max_x=30):


    velocities = velocities.reshape(-1)
    fig, ax = plt.subplots()
    histbins = np.linspace(0, max_x, num=100)
    ax.hist(velocities, alpha=0.5, density=True, bins=histbins)

    ax.set_title('Histogram of velocities')
    ax.set_xlim(0, max_x)
    ax.set_xlabel('Velocity')
    ax.set_ylabel('density')
    if threshold is not None:
        ax.axvline(threshold)
    plt.legend()
    return fig