import glob
import math
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


ALL_ENVS = ('Ant-v2', 'HalfCheetah-v2', 'Hopper-v2', 'Humanoid-v2', 'Walker2d')
DEFAULT_LOG_DIR = "./cs285/data"
DEFAULT_IMAGE_DIR = "./cs285/images"
EPS = 1e-8


def plot_question2(
        logdir,
        imagedir,
        exp_name='s1q2',
        env_ids=ALL_ENVS,
        width=0.35,
    ):
    data = {}

    # gather data from log files
    bc_avgs = []
    bc_stds = []
    expert_avgs = []
    expert_stds = []
    for env_id in env_ids:
        # find corresponding logdir
        run_prefix = os.path.join(logdir, "bc_{}_{}*".format(exp_name, env_id))
        run_logdir = glob.glob(run_prefix)
        if len(run_logdir) == 0:
            raise ValueError("Log directory with prefix {} not found."
                             .format(run_prefix))
        run_logdir = run_logdir[-1] # consider the latest run

        # read pickle file
        kvs = None
        with open(os.path.join(run_logdir, "log.pkl"), "rb") as f:
            kvs = pickle.load(f)

        # gather relevant data
        bc_avgs.append(kvs['Eval_AverageReturn'][-1])
        bc_stds.append(kvs['Eval_StdReturn'][-1])
        expert_avgs.append(kvs['Train_AverageReturn'][-1])
        expert_stds.append(kvs['Train_StdReturn'][-1])

    # prepare to plot images
    os.makedirs(imagedir, exist_ok=True)
    x = np.arange(len(env_ids)) # plot one group per env
    bc_avgs, bc_stds = np.array(bc_avgs), np.array(bc_stds)
    expert_avgs, expert_stds = np.array(expert_avgs), np.array(expert_stds)

    # plot first figure
    fig, ax = plt.subplots()
    ax.bar(x - width/2, bc_avgs, width, yerr=bc_stds, label='Behaviour Cloning')
    ax.bar(x + width/2, expert_avgs, width, yerr=expert_stds, label='Expert')
    ax.set_ylabel('episode return')
    ax.set_title('Episode return per environment')
    ax.set_xticks(x)
    ax.set_xticklabels(env_ids)
    ax.legend()
    fig.tight_layout()

    plt.savefig(os.path.join(imagedir, "{}.png".format(exp_name)))

    # plot second figure
    scale = 100. / expert_avgs
    normalized_avgs = scale * bc_avgs
    normalized_stds = scale * bc_stds

    fig, ax = plt.subplots()
    ax.axhline(30, color='gray', linestyle='--', zorder=-1)
    ax.bar(x, normalized_avgs, width, yerr=normalized_stds)
    ax.set_ylabel('normalized return (% of expert return)')
    ax.set_title('Normalized return of BC agents per environment')
    ax.set_xticks(x)
    ax.set_xticklabels(env_ids)
    fig.tight_layout()

    plt.savefig(os.path.join(imagedir, "{}_normalized.png".format(exp_name)))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, default=DEFAULT_LOG_DIR)
    parser.add_argument('--imagedir', type=str, default=DEFAULT_IMAGE_DIR)
    parser.add_argument('--q2-exp-name', type=str, default="s1q2")
    parser.add_argument('--q3-exp-name', type=str, default="s1q3")
    args = parser.parse_args()

    plot_question2(args.logdir, args.imagedir, exp_name=args.q2_exp_name)