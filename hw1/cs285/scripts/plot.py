import glob
import math
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ALL_ENVS = ('Ant-v2', 'HalfCheetah-v2', 'Hopper-v2', 'Humanoid-v2', 'Walker2d')
DEFAULT_LOG_DIR = "./cs285/data"
DEFAULT_IMAGE_DIR = "./cs285/images"


def plot_s1q2(
        logdir,
        imagedir,
        exp_name='s1q2',
        env_ids=ALL_ENVS,
        width=0.35,
    ):
    # gather data from log files
    bc_avgs = []
    bc_stds = []
    expert_avgs = []
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

    # prepare to plot images
    os.makedirs(imagedir, exist_ok=True)
    x = np.arange(len(env_ids)) # plot one group per env
    bc_avgs, bc_stds = np.array(bc_avgs), np.array(bc_stds)
    expert_avgs = np.array(expert_avgs)

    # plot normalized bc scores per environment
    scale = 100. / expert_avgs
    normalized_avgs = scale * bc_avgs
    normalized_stds = scale * bc_stds

    fig, ax = plt.subplots()
    ax.axhline(30, color='gray', linestyle='--', zorder=-1)
    ax.bar(x, normalized_avgs, width, yerr=normalized_stds)
    ax.set_ylabel('normalized return (% of expert)')
    ax.set_title('Normalized return of BC agents per environment')
    ax.set_xticks(x)
    ax.set_xticklabels(env_ids)
    fig.tight_layout()

    plt.savefig(os.path.join(imagedir, "{}.png".format(exp_name)))


def plot_s1q3(
        logdir,
        imagedir,
        env_id,
        param,
        exp_name='s1q3',
    ):
    # gather relevant data from logs
    param_values = []
    normalized_returns = []
    run_prefix = "bc_{}*_{}*".format(exp_name, env_id)
    run_logdirs = glob.glob(os.path.join(logdir, run_prefix))

    for run_logdir in run_logdirs:
        with open(os.path.join(run_logdir, "log.pkl"), "rb") as f:
            kvs = pickle.load(f)
        returns = kvs['Eval_Returns'][-1]
        value = kvs['Params'][-1][param]
        expert = kvs['Train_AverageReturn'][-1]
        for r in returns:
            r = (100 / expert) * r # expert return is 100%
            normalized_returns.append(r)
            param_values.append(value)
    normalized_returns = np.array(normalized_returns)
    param_values = np.array(param_values)

    # plot returns per param value
    fig = plt.figure()

    lp = sns.lineplot(x=param_values, y=normalized_returns, ci="sd")
    lp.set(xscale="log")
    plt.xlabel('{}'.format(param))
    plt.ylabel('normalized return (% of expert)')
    plt.title('Effect of {} for BC on {}.'.format(param, env_id))
    fig.tight_layout()

    plt.savefig(os.path.join(imagedir, '{}.png'.format(exp_name)))

def plot_s2q2(
        logdir,
        imagedir,
        env_id,
        exp_name="s2q2",
    ):
    # gather relevant data from logs
    def load_kvs(prefix, latest=True):
        rundirs = glob.glob(os.path.join(logdir, prefix))
        if len(rundirs) == 0:
            raise ValueError("Log directory with prefix {} not found.".format(prefix))
        kvs = []
        for rundir in rundirs:
            with open(os.path.join(rundir, "log.pkl"), "rb") as f:
                kvs.append(pickle.load(f))
        return kvs

    bc_kvs = load_kvs("bc_{}_{}*".format(exp_name, env_id))[-1] # get latest if multiple
    dagger_kvs = load_kvs("dagger_{}_{}*".format(exp_name, env_id))

    bc_score = bc_kvs['Eval_AverageReturn'][-1]
    expert_score = bc_kvs['Train_AverageReturn'][-1]
    returns = []
    iterations = []
    for kvs in dagger_kvs:
        for i, r in enumerate(kvs['Eval_Returns']):
            returns.extend(r)
            iterations.extend([i] * len(r))

    # plot dagger performance per iteration
    fig = plt.figure()

    lp = sns.lineplot(x=iterations, y=returns, ci="sd",
        label="DAgger")
    plt.axhline(y=bc_score, color='gray', linestyle='--', zorder=-1,
        label='Behaviour Cloning')
    plt.axhline(y=expert_score, color='black', linestyle='--', zorder=-1,
        label='Expert')
    plt.ylabel('Return')
    plt.xlabel('Iteration')
    plt.title('DAgger results on {}.'.format(env_id))
    plt.legend()

    plt.savefig(os.path.join(imagedir, '{}.png'.format(exp_name)))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, default=DEFAULT_LOG_DIR)
    parser.add_argument('--imagedir', type=str, default=DEFAULT_IMAGE_DIR)
    parser.add_argument('--s1q2-exp-name', type=str, default="s1q2")
    parser.add_argument('--s1q3-exp-name', type=str, default="s1q3")
    parser.add_argument('--s1q3-env-id', type=str, default='Hopper-v2')
    parser.add_argument('--s1q3-param', type=str, default="num_agent_train_steps_per_iter")
    parser.add_argument('--s2q2-exp-name', type=str, default="s2q2")
    parser.add_argument('--s2q2-env-id', type=str, default="Hopper-v2")
    args = parser.parse_args()

    plot_s1q2(args.logdir, args.imagedir, exp_name=args.s1q2_exp_name)
    plot_s1q3(args.logdir, args.imagedir, args.s1q3_env_id, args.s1q3_param, exp_name=args.s1q3_exp_name)
    plot_s2q2(args.logdir, args.imagedir, args.s2q2_env_id, exp_name=args.s2q2_exp_name)