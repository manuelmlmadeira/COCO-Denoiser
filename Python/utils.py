import math
import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.datasets import load_svmlight_file
from scipy.optimize import fmin_l_bfgs_b


def get_data(dataname):
    data = load_svmlight_file(dataname)
    return data[0], data[1]


def find_precise_minimum(A, b, lbda, loss, grad):
    _, d = A.shape
    x_init = np.zeros(d)
    x_min, f_min, _ = fmin_l_bfgs_b(loss, x_init, grad,
                                    args=(A, b, lbda), pgtol=1e-30, factr=1e-30)
    return x_min, f_min


def run_method(dataset_str, method, method_plus_COCO, method_str, n_reps, x_init, store_every, n_steps, K_list, L, tol,
               method_hyperparams, n, loss, grad_i, A, b, lbda, x_min, f_min, use_cvx):
    d = x_init.shape[0]
    visited_points = np.empty((len(K_list), n_reps, n_steps + 1, d))
    solvers = []
    for K_idx, K in enumerate(K_list):
        COCO_hyperparams = (K, L, tol)
        print(f"K = {K}")
        # If K==1 we run vanilla SGD, otherwise we plug-in COCO
        if K == 1:
            solvers.append(method_str)
        else:
            solvers.append(method_str + f"+COCO$_{K}$")

        for rep in range(n_reps):
            if K == 1:
                rep_x_final, rep_visited_points = method(x_init, grad_i, store_every, n, n_steps, method_hyperparams,
                                                         args=(A, b, lbda))

            else:
                rep_x_final, rep_visited_points = method_plus_COCO(x_init, grad_i, store_every, n, n_steps,
                                                                   COCO_hyperparams, use_cvx, method_hyperparams,
                                                                   args=(A, b, lbda))
            visited_points[K_idx, rep, :, :] = rep_visited_points
            if (rep + 1) % 25 == 0:
                print(f"rep = {rep + 1}")

    # Save visited_points
    with open('fourclass_visited_points_' + method_str, 'wb') as f:
        pickle.dump([visited_points, solvers], f)

    # To read use:
    # with open('fourclass_visited_points...', 'rb') as f:
    #     visited_points, solvers = pickle.load(f)

    distance_visited_points, f_visited_points = compute_distances_and_function_values(visited_points, A, b, lbda, x_min,
                                                                                      f_min, loss)
    # Plot & save results
    metric_str = "$E[f(x_k) - f(x^*)]$"
    plot_epochs(f_visited_points, solvers, dataset_str, method_str, metric_str)
    metric_str = "$E[||x_k - x^*||]$"
    plot_epochs(distance_visited_points, solvers, dataset_str, method_str, metric_str)


def compute_distances_and_function_values(visited_points, A, b, lbda, x_min, f_min, loss):

    distance_visited_points = np.empty(visited_points.shape[:-1])
    f_visited_points = np.empty(visited_points.shape[:-1])

    for K_idx in range(visited_points.shape[0]):
        for rep_idx in range(visited_points.shape[1]):
            for step_idx in range(visited_points.shape[2]):
                distance_visited_points[K_idx, rep_idx, step_idx] = np.linalg.norm(visited_points[K_idx, rep_idx, step_idx, :] - x_min)
                f_visited_points[K_idx, rep_idx, step_idx] = loss(visited_points[K_idx, rep_idx, step_idx, :], A, b,
                                                                  lbda) - f_min

    return distance_visited_points, f_visited_points


def plot_epochs(metric_mtx,  solvers, dataset_str, method_str, metric_str):
    """Function used to plot results
    visited_points are of shape (number of K values, number of reps, number of steps, dimension)
    solvers are used to legend each plot
    """

    # Figure specifications
    fig, ax = plt.subplots(figsize=(20, 12))
    linestyles = ['-', '--', '--', '-.', ':', '--', '-']

    # Initialization
    ls = 0
    x = np.arange(metric_mtx.shape[2])
    n_reps = metric_mtx.shape[1]

    # Plot curve for each K
    for mtx in metric_mtx:
        # Compute mean and std for given K
        mean = np.mean(mtx, axis=0)
        sem = np.std(mtx, axis=0) / math.sqrt(n_reps)
        # Plot
        ax.errorbar(x, mean, xerr=0, yerr=sem, linestyle=linestyles[ls])
        ax.set_yscale('log')
        plt.xlabel("Oracle Consultations", fontsize=40)
        plt.ylabel(metric_str, fontsize=40)
        ls += 1
    plt.xticks(fontsize=40)
    plt.yticks(fontsize=40)
    plt.legend(solvers, fontsize=40)
    plt.xlim([0, metric_mtx.shape[2]])
    # Save figure
    fig_output_name = dataset_str + "_" + method_str + "_" + metric_str + ".pdf"
    plt.savefig(fig_output_name, bbox_inches='tight')
    print(f"Figure saved as " + fig_output_name)