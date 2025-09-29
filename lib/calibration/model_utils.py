import os
import time

import numpy as np
from joblib import dump
from matplotlib import pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures


def get_model_parameters(x_data, y_data):
    """
    estimates a model based on x and y data
    x_data is of shape (:, 4), i.e. contains the two pairs of input coordinates
    y_data is of shape (:, 2) and contains the output coordinates
    """

    # ---- STEP1: determine alpha parameter and polydegree ---- #

    # set the parameters for the grid search
    alpha = [1e-10, 1e-7, 1e-6, 1e-4, 1e-5, 1e-3, 0.001, 0.01, 0.1, 1]
    polyDegList = [1, 2, 3, 4]
    parameters = {'estimator__alpha': alpha,
                  'preprocessor__degree': polyDegList}

    # set the pipeline for building the model
    pipe = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('preprocessor', PolynomialFeatures(include_bias=True)),
        ('estimator', Ridge())
    ])

    # set the gridsearch score metric
    score = 'neg_mean_squared_error'

    # perform a gridsearch
    getter_GS = GridSearchCV(pipe, parameters, cv=3, scoring=score)
    getter_GS.fit(x_data, y_data)

    print('best params: ' + str(getter_GS.best_params_))
    print('best_score: ' + str(getter_GS.best_score_))
    return getter_GS


def examine_residuals(x_data, y_data, getter_GS, save_fig_folder, fig_filename):

    # the number of training test splits
    numPermutations = 100
    # the proportion of the dataset to include in the train split
    trainFac = 0.9
    # the binning for plotting the residuals
    bins = np.arange(0, 10, 0.1)

    numBalls = x_data.shape[0]

    # --------------------------#

    t0 = time.time()

    # use the GS results to set the params
    alpha = getter_GS.best_params_['estimator__alpha']
    polyDeg = getter_GS.best_params_['preprocessor__degree']

    # based on the trainFac, determine the number of balls to use for training
    # we do this just for preallocating the results array below
    dummy_X_train, _, _, _, = train_test_split(x_data, y_data, train_size=trainFac)
    numTrainingBalls = dummy_X_train.shape[0]

    # preallocate arrays to hold the results
    train_errors = np.zeros((numPermutations, numTrainingBalls))
    test_errors = np.zeros((numPermutations, numBalls - numTrainingBalls))

    for permIdx in range(numPermutations):
        # train test split
        X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, train_size=trainFac)

        getter = Pipeline(steps=[('scaler', StandardScaler()),
                                 ('preprocessor', PolynomialFeatures(degree=polyDeg, include_bias=True)),
                                 ('estimator', Ridge(alpha=alpha))
                                 ])

        getter.fit(X_train, y_train)

        train_calculated_coordinates = getter.predict(X_train)
        errs = np.linalg.norm(y_train - train_calculated_coordinates, axis=1)
        train_errors[permIdx, :] = errs

        test_calculated_coordinates = getter.predict(X_test)
        errs = np.linalg.norm(y_test - test_calculated_coordinates, axis=1)
        test_errors[permIdx, :] = errs

    te = time.time()
    print('t = {0} s'.format(te - t0))
    print()

    # flatten the error arrays for histogramming the distributions
    train_errors = train_errors.reshape(-1)
    test_errors = test_errors.reshape(-1)

    plt.figure()
    plt.title('Residuals Test: PolyOrd = {0}, alpha = {1}'.format(polyDeg, alpha))
    plt.hist(train_errors, label='train errors', bins=bins, alpha=0.2, density=True)
    plt.hist(test_errors, label='test errors', bins=bins, alpha=0.2, density=True)
    plt.legend()
    plt.savefig(os.path.join(save_fig_folder, fig_filename), dpi=300)
    plt.clf()


    return alpha, polyDeg



def fit_and_save_model(alpha, polyDeg, x_data, y_data, save_model_data_folder, getter_name):
    # --------------- STEP3: Fit and save the final model ----------------#


    # make the final model
    getter = Pipeline(steps=[('scaler', StandardScaler()),
                             ('preprocessor', PolynomialFeatures(degree=polyDeg, include_bias=True)),
                             ('estimator', Ridge(alpha=alpha))
                             ])

    python_calibration_folderPath = save_model_data_folder
    os.makedirs(python_calibration_folderPath, exist_ok=True)

    getter.fit(x_data, y_data)

    dump(getter, os.path.join(python_calibration_folderPath, getter_name))
    return getter

def _plot_residuals(residuals, bins, xlabel, xlim, xticks, save_fig_folder, fig_filename):
    """
    Helper function to plot residuals histogram.

    Args:
        residuals (numpy.ndarray): Array of residual values
        bins (numpy.ndarray): Bins for histogram
        xlabel (str): Label for x-axis
        xlim (tuple): Limits for x-axis (min, max)
        xticks (list): Tick positions for x-axis
        save_fig_folder (str): Folder to save the figure
        fig_filename (str): Filename for the saved figure
    """
    print('residuals shape: ' + str(residuals.shape))
    fig, axs = plt.subplots()
    ax = axs

    ax.hist(residuals, bins=bins)

    ax.set_ylabel('counts')
    ax.set_xlabel(xlabel)

    ax.set_xlim(xlim)
    ax.set_ylim(0, 35)

    ax.set_xticks(xticks)
    ax.set_yticks([0, 35])
    plt.savefig(os.path.join(save_fig_folder, fig_filename), dpi=300)
    plt.clf()


def compute_residuals(getter, x_data, y_data, save_fig_folder, fig_filename):
    """
    Compute and plot residuals for image coordinate predictions.

    Args:
        getter (Pipeline): Fitted model pipeline
        x_data (numpy.ndarray): Input data
        y_data (numpy.ndarray): Target data
        save_fig_folder (str): Folder to save the figure
        fig_filename (str): Filename for the saved figure
    """
    calculated_im_coordinates = getter.predict(x_data)
    residuals = np.linalg.norm(y_data - calculated_im_coordinates, axis=1)

    # binning for plotting the residuals
    bins = np.arange(0, 10, 0.1)
    xlabel = 'residuals [pixels]'
    xlim = (0, 10)
    xticks = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    _plot_residuals(residuals, bins, xlabel, xlim, xticks, save_fig_folder, fig_filename)


def compute_residuals_im_to_real(getter, x_data, y_data, save_fig_folder, fig_filename):
    """
    Compute and plot residuals for real-world coordinate predictions.

    Args:
        getter (Pipeline): Fitted model pipeline
        x_data (numpy.ndarray): Input data
        y_data (numpy.ndarray): Target data
        save_fig_folder (str): Folder to save the figure
        fig_filename (str): Filename for the saved figure
    """
    calculated_im_coordinates = getter.predict(x_data)
    residuals = np.linalg.norm(y_data - calculated_im_coordinates, axis=1)

    # binning for plotting the residuals
    bins = np.arange(0, 5, 0.025)
    xlabel = 'residuals [cm]'
    xlim = (0, 5)
    xticks = [0, 0.5, 1, 2, 3, 4, 5]

    _plot_residuals(residuals, bins, xlabel, xlim, xticks, save_fig_folder, fig_filename)
