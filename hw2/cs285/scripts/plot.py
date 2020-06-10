import glob
import math
import os
import pickle
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


DEFAULT_LOG_DIR = "./cs285/data"
DEFAULT_IMAGE_DIR = "./cs285/images"
COLORS = ('b', 'g', 'r', 'k')


plt.style.use('ggplot')


def average_smoothing(s, window_size):
    """
    Smooths a series of scalars to its moving window average.
    Inspired by: https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html
    Args:
        s: a series of scalars
        window_size: the size of the moving window
    Returns:
        a np.array of the same length of series containing the averages
    """
    if window_size < 2:
        return s
    s = np.r_[s[window_size-1:0:-1], s, s[-2:-window_size-1:-1]]
    window = np.ones(window_size) / window_size
    smooth_s = np.convolve(s, window, mode='valid')
    if window_size % 2 == 0:
        smooth_s = smooth_s[(window_size//2-1):-(window_size//2)]
    else:
        smooth_s = smooth_s[(window_size//2):-(window_size//2)]
    return smooth_s


def plot_learning_curves(logdir, imagedir, run_prefix, window_size):
    # get list of matching experiments
    prefix = os.path.join(logdir, "pg_{}_*".format(run_prefix))
    logdirs = glob.glob(prefix)
    if len(logdirs) == 0:
        raise ValueError("Log directory with prefix {} not found."
                         .format(prefix))

    # gather data from logs
    data = defaultdict(list)
    for logdir in logdirs:
        # load logs
        with open(os.path.join(logdir, "log.pkl"), "rb") as f:
            kvs = pickle.load(f)

        # store data
        exp_name = kvs['ExperimentName'][0]
        data[exp_name].append(kvs)

    # aggregate data from the same experiments
    avg_returns = {}
    all_avg_returns = {}
    for exp in data:
        print("{} run(s) found for experiment {}."
              .format(len(data[exp]), exp))
        all_avg_returns[exp] = []
        for kvs in data[exp]:
            rets = average_smoothing(
                np.array(kvs['Eval_AverageReturn']), window_size)
            all_avg_returns[exp].append(rets)
        avg_returns[exp] = np.array(all_avg_returns[exp]).mean(axis=0)

    # plot
    fig, ax = plt.subplots()

    for exp in avg_returns:
        color = next(ax._get_lines.prop_cycler)['color']
        iters = np.arange(len(avg_returns[exp]))
        for rets in all_avg_returns[exp]:
            ax.plot(iters, rets, color=color, alpha=0.4)
        ax.plot(iters, avg_returns[exp], color=color, label=exp)

    ax.set_ylabel('normalized return')
    ax.set_title('Normalized return of {} experiments'.format(run_prefix))
    ax.legend()
    fig.tight_layout()

    plt.savefig(os.path.join(imagedir, "{}.png".format(run_prefix)))


def plot_p3(logdir, imagedir, window_size):
    plot_learning_curves(logdir, imagedir, "sb", window_size)
    plot_learning_curves(logdir, imagedir, "lb", window_size)


def plot_p4(logdir, imagedir, window_size):
    plot_learning_curves(logdir, imagedir, "ip", window_size)


def plot_p6(logdir, imagedir, window_size):
    plot_learning_curves(logdir, imagedir, "ll", window_size)


def plot_p7(logdir, imagedir, window_size):
    plot_learning_curves(logdir, imagedir, "hc", window_size)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--problem', '-pb', type=int)
    parser.add_argument('--logdir', type=str, default=DEFAULT_LOG_DIR)
    parser.add_argument('--imagedir', type=str, default=DEFAULT_IMAGE_DIR)
    parser.add_argument('--window-size', '-ws', type=int, default=1)
    args = parser.parse_args()

    os.makedirs(args.imagedir, exist_ok=True)

    if args.problem == 3:
        plot_p3(args.logdir, args.imagedir, args.window_size)
    elif args.problem == 4:
        plot_p4(args.logdir, args.imagedir, args.window_size)
    elif args.problem == 6:
        plot_p6(args.logdir, args.imagedir, args.window_size)
    elif args.problem == 7:
        plot_p7(args.logdir, args.imagedir, args.window_size)
