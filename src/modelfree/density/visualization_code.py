from glob import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

DENSITY_DIR = "density"


def opponent_convert(x):
    if 'zoo' in x:
        return 'zoo'
    else:
        return x


def get_full_directory(env, victim_id, n_components, covariance):
    hp_dir = f"{DENSITY_DIR}/gmm_{n_components}_components_{covariance}"
    exp_dir = glob(hp_dir + "/*")[0]
    env_dir = f"{env}-v0_victim_zoo_{victim_id}"
    full_env_dir = os.path.join(exp_dir, 'fitted', env_dir)
    return full_env_dir


def get_train_test_merged_df(env, victim_id, n_components, covariance):
    full_env_dir = get_full_directory(env, victim_id, n_components, covariance)
    train_df = pd.read_csv(os.path.join(full_env_dir, "train_metadata.csv"))
    train_df['is_train'] = True
    test_df = pd.read_csv(os.path.join(full_env_dir, "test_metadata.csv"))
    test_df['is_train'] = False
    merged = pd.concat([train_df, test_df])
    merged['broad_opponent'] = merged['opponent_id'].apply(lambda x: opponent_convert(x))
    merged['raw_proba'] = np.exp(merged['log_proba'])
    return merged


def get_metrics_dict(env, victim_id, n_components, covariance):
    full_env_dir = get_full_directory(env, victim_id, n_components, covariance)
    with open(os.path.join(full_env_dir, 'metrics.json'), 'r') as fp:
        metric_dict = json.load(fp)
    return metric_dict


def comparative_densities(group_var, env_name, victim, shade=False, cutoff_point=None):
    df = get_train_test_merged_df(env_name, victim)
    plt.figure(figsize=(10, 7))
    if cutoff_point is not None:
        subset = df[df['log_proba'] > cutoff_point]
    else:
        subset = df.copy()
    grped = subset.groupby(group_var)
    for name, grp in grped:
        # clean up random_none to just random
        name = name.replace('_none', '')
        avg_log_proba = round(np.mean(grp['log_proba']), 2)
        sns.kdeplot(grp['log_proba'], label=f"{name}: {avg_log_proba}", shade=shade)
    plt.suptitle(f"{env_name} Densities, Victim Zoo {victim}: Trained on Zoo 1", y=0.95)
    plt.title("Avg Log Proba* in Legend")

    # Note that this was for running in a notebook with matplotlib inline; probably would want
    # to dynamically construct filename and do savefig, but I'll leave it this way for
    # efficiency's sake
    plt.show()


def heatmap_plot(env_name, metric, victim=1, savefile=None, error_val=-1):
    n_component_grid = [1, 2, 3, 5, 10]
    covariance_grid = ['diag', 'spherical', 'full']
    metric_grid = np.zeros(shape=(5, 3))
    if isinstance(metric, str):
        metric_name = metric
    else:
        metric_name = metric.__name__

    for i, n_components in enumerate(n_component_grid):
        for j, covariance in enumerate(covariance_grid):
            try:
                metrics = get_metrics_dict(env_name, victim, n_components, covariance)
                if isinstance(metric, str):
                    metric_grid[i][j] = metrics[metric]
                else:
                    metric_grid[i][j] = metric(metrics)
            except FileNotFoundError:
                print(
                    f"Hit exception on {env_name}, {n_components} components {covariance},"
                    f" filling in {error_val}")
                metric_grid[i][j] = error_val

    ll_df = pd.DataFrame(metric_grid, index=n_component_grid, columns=covariance_grid)
    plt.figure(figsize=(10, 7))
    sns.heatmap(ll_df, annot=True)
    plt.title(f"HP Search on {env_name}, Victim {victim}: {metric_name}")
    if savefile is not None:
        plt.savefig(savefile)


if __name__ == "__main__":
    # Example Density plot
    comparative_densities(group_var='opponent_id', env_name='KickAndDefend', victim='1',
                          cutoff_point=-1000)

    # Make heatmaps
    def train_bic_in_millions(x):
        return x['train_bic'] / 1000000
    output_dir = "density_plots"

    for env in ['KickAndDefend', 'SumoAnts', 'SumoHumans', 'YouShallNotPass']:
        heatmap_plot(env_name=env, metric=train_bic_in_millions,
                     savefile=f"{output_dir}/{env}_train_bic.png")
        heatmap_plot(env_name=env, metric='validation_log_likelihood',
                     savefile=f"{output_dir}/{env}_validation_log_likelihood.png")
