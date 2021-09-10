import json
import os

# from threading import current_thread
import numpy as np
from scipy.spatial import ConvexHull
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error
import csv
from glob import glob
from tqdm import tqdm

# import cuml


def read_comp_and_energy_points(datafile):
    """read_comp_and_energy_points(datafile)
    Generates points in composition and energy space for use in convex hull algorithms.
    Args:
        datafile(str): Path to the json data file that contains composition and formation energy data. (generate with "casm query -k comp formation_energy")

    Returns:
        points(ndarray): Numpy mxn matrix. m = # of configurations in the casm project, n = # of composition axes + 1 for the energy axis.
    """
    with open(datafile) as f:
        data = json.load(f)
    points = [
        [x[0] for x in entry["comp"]] + [entry["formation_energy"]] for entry in data
    ]
    points = np.array(points)
    return points


def checkhull(hull_vertex, test_coords):
    """Find if specified coordinates are above, on or below the specified lower convex hull.
    Args:
        hull_vertex(ndarray): 2D array, shape nxm where n = # of points, m = # of composition dimensions + 1 energy as the last column.
        test_coords(ndarray): 2D array, shape lxm where l = # of points to test, m = # of composition dimensions + 1 energy as the last column.
    Returns:
        tuple(
            above_hull(ndarray): 2D array, shape p x m where m = # of composition dimensions + 1 energy as the last column.
            on_hull(ndarray): 2D array, shape q x m where m = # of composition dimensions + 1 energy as the last column.
            below_hull(ndarray): 2D array, shape r x m where m = # of composition dimensions + 1 energy as the last column.
        )
    """

    # TODO: function that ony returns projected energy
    # Split data into composition and energy
    hull_comps = hull_vertex[:, 0:-1]
    hull_energies = hull_vertex[:, -1]
    x_test = test_coords[:, 0:-1]
    y_test = test_coords[:, -1]

    # Fit linear grid
    interp_hull = griddata(hull_comps, hull_energies, x_test, method="linear")

    # Check if the y_test points are above or below the hull
    hull_dist = y_test - interp_hull
    above_hull = []
    below_hull = []
    on_hull = []
    for i in range(x_test.shape[0]):
        if np.isclose(hull_dist[i], 0, 1e-14):
            on_hull.append([x_test[i], y_test[i]])
        elif hull_dist[i] > 0:
            above_hull.append([x_test[i], y_test[i]])
        elif hull_dist[i] < 0:
            below_hull.append([x_test[i], y_test[i]])
    above_hull = np.array(above_hull)
    below_hull = np.array(below_hull)
    on_hull = np.array(on_hull)
    # Just return hull distance
    return np.array(hull_dist)


def plot_clex_hull_data_1_x(
    fit_dir,
    hall_of_fame_index,
    full_formation_energy_file="full_formation_energies.txt",
):
    """plot_clex_hull_data_1_x(fit_dir, hall_of_fame_index, full_formation_energy_file='full_formation_energies.txt')

    Function to plot DFT energies, cluster expansion energies, DFT convex hull and cluster expansion convex hull.

    Args:
        fit_dir (str): absolute path to a casm cluster expansion fit.
        hall_of_fame_index (int or str): Integer index. "hall of fame" index for a specific fit (corresponding to a set of "Effective Cluster Interactions" or ECI).
        full_formation_energy_file (str): filename that contains the formation energy of all configurations of interest. Generated using a casm command

    Returns:
        fig: a python figure object.
    """
    # TODO: Definitely want to re-implement this with json input
    # Pre-define values to pull from data files
    # title is intended to be in the form of "casm_root_name_name_of_specific_fit_directory".
    title = fit_dir.split("/")[-3] + "_" + fit_dir.split("/")[-1]
    dft_scel_names = []
    clex_scel_names = []
    dft_hull_data = []
    clex_hull_data = []
    cv = None
    rms = None
    wrms = None
    below_hull_exists = False
    hall_of_fame_index = str(hall_of_fame_index)

    # Read necessary files
    os.chdir(fit_dir)
    files = glob("*")
    for f in files:
        if "_%s_dft_gs" % hall_of_fame_index in f:
            dft_hull_path = os.path.join(fit_dir, f)
            dft_hull_data = np.genfromtxt(
                dft_hull_path, skip_header=1, usecols=list(range(1, 10))
            ).astype(float)
            with open(dft_hull_path, "r") as dft_dat_file:
                dft_scel_names = [
                    row[0] for row in csv.reader(dft_dat_file, delimiter=" ")
                ]
                dft_scel_names = dft_scel_names[1:]

        if "_%s_clex_gs" % hall_of_fame_index in f:
            clex_hull_path = os.path.join(fit_dir, f)
            clex_hull_data = np.genfromtxt(
                clex_hull_path, skip_header=1, usecols=list(range(1, 10))
            ).astype(float)
            with open(clex_hull_path, "r") as clex_dat_file:
                clex_scel_names = [
                    row[0] for row in csv.reader(clex_dat_file, delimiter=" ")
                ]
                clex_scel_names = clex_scel_names[1:]

        if "_%s_below_hull" % hall_of_fame_index in f:
            below_hull_exists = True
            below_hull_path = os.path.join(fit_dir, f)
            below_hull_data = np.reshape(
                np.genfromtxt(
                    below_hull_path, skip_header=1, usecols=list(range(1, 10))
                ).astype(float),
                ((-1, 9)),
            )
            with open(below_hull_path, "r") as below_hull_file:
                below_hull_scel_names = [
                    row[0] for row in csv.reader(below_hull_file, delimiter=" ")
                ]
                below_hull_scel_names = below_hull_scel_names[1:]

        if "check.%s" % hall_of_fame_index in f:
            checkfile_path = os.path.join(fit_dir, f)
            with open(checkfile_path, "r") as checkfile:
                linecount = 0
                cv_rms_wrms_info_line = int
                for line in checkfile.readlines():
                    if (
                        line.strip() == "-- Check: individual 0  --"
                    ):  # % hall_of_fame_index:
                        cv_rms_wrms_info_line = linecount + 3

                    if linecount == cv_rms_wrms_info_line:
                        cv = float(line.split()[3])
                        rms = float(line.split()[4])
                        wrms = float(line.split()[5])
                    linecount += 1

    # Generate the plot
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.text(
        0.80,
        0.80 * min(dft_hull_data[:, 4]),
        "CV:      %.10f\nRMS:    %.10f\nWRMS: %.10f" % (cv, rms, wrms),
        fontsize=15,
    )
    labels = []
    plt.title(title, fontsize=30)
    plt.xlabel(r"Composition", fontsize=20)
    plt.ylabel(r"Energy $\frac{eV}{prim}$", fontsize=20)
    plt.plot(dft_hull_data[:, 1], dft_hull_data[:, 5], marker="o", color="xkcd:crimson")
    labels.append("DFT Hull")
    plt.plot(
        clex_hull_data[:, 1],
        clex_hull_data[:, 8],
        marker="o",
        linestyle="dashed",
        color="b",
    )
    labels.append("ClEx Hull")
    plt.scatter(dft_hull_data[:, 1], dft_hull_data[:, 8], color="k")
    labels.append("Clex Prediction of DFT Hull")

    if full_formation_energy_file:
        # format:
        # run casm query -k comp formation_energy hull_dist clex clex_hull_dist -o full_formation_energies.txt
        #            configname    selected           comp(a)    formation_energy    hull_dist(MASTER,atom_frac)        clex()    clex_hull_dist(MASTER,atom_frac)
        datafile = full_formation_energy_file
        data = np.genfromtxt(datafile, skip_header=1, usecols=list(range(2, 7))).astype(
            float
        )
        composition = data[:, 0]
        dft_formation_energy = data[:, 1]
        clex_formation_energy = data[:, 3]
        plt.scatter(composition, dft_formation_energy, color="salmon")
        labels.append("DFT energies")
        plt.scatter(composition, clex_formation_energy, marker="x", color="skyblue")
        labels.append("ClEx energies")

    # TODO: This implementation is wrong. This is the distance below the hull (energy difference) not the actual energy.
    if below_hull_exists:
        plt.scatter(below_hull_data[:, 1], below_hull_data[:, 7], marker="+", color="k")
        labels.append("Clex Below Clex Prediction of DFT Hull Configs")
    else:
        print("'_%s_below_hull' file doesn't exist" % hall_of_fame_index)

    plt.legend(labels, loc="lower left", fontsize=10)

    fig = plt.gcf()
    return fig


def read_corr_and_formation_energy(datafile):
    """
    read_corr_and_formation_energy(datafile)

    Reads and returns data from json containing correlation functions and formation energies.
    Args:
        datafile(str): Path to the json file containing the correlation functions and formation energies.
    Returns:
        tuple(
            corr,                   (ndarray): Correlation functions: nxm matrix of correlation funcitons: each row corresponds to a configuration.
            formation_energy,       (ndarray): Formation energies: vecrtor of n elements: one for each configuration.
            scel_names              (ndarray): The name for a given configuration. Vector of n elements.
        )
    """
    with open(datafile) as f:
        data = json.load(f)

    corr = []
    formation_energy = []
    scel_names = []
    for entry in data:
        corr.append(np.array(entry["corr"]).flatten())
        formation_energy.append(entry["formation_energy"])
        scel_names.append(entry["name"])

    corr = np.array(corr)
    formation_energy = np.array(formation_energy)
    scel_names = np.array(scel_names)
    return (corr, formation_energy, scel_names)


def run_lassocv(corr, formation_energy):
    reg = LassoCV(fit_intercept=False, n_jobs=4).fit(corr, formation_energy)
    eci = reg.coef_
    return eci


def generate_rand_eci_vec(num_eci, stdev, normalization):
    eci_vec = np.random.normal(scale=stdev, size=num_eci)
    eci_vec = (eci_vec / np.linalg.norm(eci_vec)) * normalization
    return eci_vec


def metropolis_hastings_ratio(current_eci, proposed_eci, corr, formation_energy):

    left_term = (
        np.linalg.norm(proposed_eci, ord=1) / np.linalg.norm(current_eci, ord=1)
    ) ** (-1 * current_eci.shape[0])

    current_energy = np.matmul(corr, current_eci)
    proposed_energy = np.matmul(corr, proposed_eci)
    right_term_numerator = np.linalg.norm(formation_energy - proposed_energy)
    right_term_denom = np.linalg.norm(formation_energy - current_energy)

    right_term = (right_term_numerator / right_term_denom) ** (
        -1 * formation_energy.shape[0]
    )

    mh_ratio = left_term * right_term
    if mh_ratio > 1:
        mh_ratio = 1
    return mh_ratio


def run_eci_monte_carlo(
    corr_and_energy_file, eci_walk_step_size, iterations, output_dir
):

    corr, formation_energy, names = read_corr_and_formation_energy(corr_and_energy_file)

    accept = None
    # Run lassoCV to get expected eci values
    lasso_eci = run_lassocv(corr, formation_energy)

    acceptance = []
    sampled_eci = []
    proposed_ground_states = []

    current_eci = lasso_eci
    sampled_eci.append(current_eci)
    for i in tqdm(range(iterations), desc="Monte Carlo Progress"):
        # for i in range(iterations):
        eci_random_vec = generate_rand_eci_vec(
            num_eci=lasso_eci.shape[0], stdev=1, normalization=eci_walk_step_size
        )
        proposed_eci = current_eci + eci_random_vec

        mh_ratio = metropolis_hastings_ratio(
            current_eci, proposed_eci, corr, formation_energy
        )

        acceptance_comparison = np.random.uniform()
        if mh_ratio >= acceptance_comparison:
            acceptance.append(True)
            current_eci = proposed_eci
            sampled_eci.append(proposed_eci)
        else:
            acceptance.append(False)
            sampled_eci.append(current_eci)

    acceptance = np.array(acceptance)
    sampled_eci = np.array(sampled_eci)
    acceptance_prob = np.count_nonzero(acceptance) / acceptance.shape[0]
    print("Acceptance Probability is: %f" % acceptance_prob)

    results = {
        "sampled_eci": sampled_eci,
        "acceptance": acceptance,
        "acceptance_prob": acceptance_prob,
    }
    # savefile = os.path.join(output_dir, "eci_mc_results.json")
    # print("Saving results to %s" % savefile)
    # with open(savefile, "w") as f:
    #    json.dump(results, f, indent="")
    return results


def plot_eci_hist(eci_data):
    plt.hist(x=eci_data, bins="auto", color="xkcd:crimson", alpha=0.7, rwidth=0.85)

    plt.xlabel("ECI value (eV)", fontsize=18)
    plt.ylabel("Count", fontsize=18)
    # plt.show()
    fig = plt.gcf()
    return fig


def plot_eci_covariance(eci_data_1, eci_data_2):
    plt.scatter(eci_data_1, eci_data_2, color="xkcd:crimson")
    plt.xlabel("ECI 1 (eV)", fontsize=18)
    plt.ylabel("ECI 2 (eV)", fontsize=18)
    fig = plt.gcf()
    return fig
