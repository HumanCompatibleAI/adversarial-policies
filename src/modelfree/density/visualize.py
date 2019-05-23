import collections
from glob import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from modelfree.visualize import util
from modelfree.visualize.styles import STYLES

DENSITY_DIR = "data/density"

PRETTY_COLS = collections.OrderedDict([
    ('zoo_1_train', 'Zoo*1T'),
    ('zoo_1_test', 'Zoo*1V'),
    ('zoo_2', 'Zoo*2'),
    ('zoo_3', 'Zoo*3'),
    ('random', 'Rand'),
    ('ppo2_1', 'Adv'),
])

PRETTY_ENVS = collections.OrderedDict([
    ('KickAndDefend', 'Kick and\nDefend'),
    ('YouShallNotPassHumans', 'You Shall\nNot Pass'),
    ('SumoHumans', 'Sumo\nHumans'),
])

HUE_ORDER = ['Adv', 'Zoo*1T', 'Rand', 'Zoo*1V', 'Zoo*2', 'Zoo*3']


def get_full_directory(env, victim_id, n_components, covariance):
    hp_dir = f"{DENSITY_DIR}/gmm_{n_components}_components_{covariance}"
    exp_dir = glob(hp_dir + "/*")[0]
    env_dir = f"{env}-v0_victim_zoo_{victim_id}"
    full_env_dir = os.path.join(exp_dir, 'fitted', env_dir)
    return full_env_dir


def get_train_test_merged_df(env, victim_id, n_components, covariance):
    full_env_dir = get_full_directory(env, victim_id, n_components, covariance)

    train_df = pd.read_csv(os.path.join(full_env_dir, "train_metadata.csv"))
    train_df = train_df.loc[train_df['opponent_id'] == 'zoo_1', :]
    train_df['opponent_id'] = 'zoo_1_train'

    test_df = pd.read_csv(os.path.join(full_env_dir, "test_metadata.csv"))
    test_df.loc[test_df['opponent_id'] == 'zoo_1', 'opponent_id'] = 'zoo_1_test'

    merged = pd.concat([train_df, test_df])
    return merged


def get_metrics_dict(env, victim_id, n_components, covariance):
    full_env_dir = get_full_directory(env, victim_id, n_components, covariance)
    with open(os.path.join(full_env_dir, 'metrics.json'), 'r') as fp:
        metric_dict = json.load(fp)
    return metric_dict


def comparative_densities(env_name, victim, n_components, covariance,
                          savefile=None, shade=False, cutoff_point=None):
    df = get_train_test_merged_df(env_name, victim, n_components, covariance)
    fig = plt.figure(figsize=(10, 7))

    grped = df.groupby('opponent_id')
    log_probs = {}
    for name, grp in grped:
        # clean up random_none to just random
        name = name.replace('_none', '')
        avg_log_proba = np.mean(grp['log_proba'])
        sns.kdeplot(grp['log_proba'], label=f"{name}: {round(avg_log_proba, 2)}", shade=shade)
        log_probs[name] = avg_log_proba

    xmin, xmax = plt.xlim()
    xmin = max(xmin, cutoff_point)
    plt.xlim((xmin, xmax))

    plt.suptitle(f"{env_name} Densities, Victim Zoo {victim}: Trained on Zoo 1", y=0.95)
    plt.title("Avg Log Proba* in Legend")

    if savefile is not None:
        fig.savefig(f'{savefile}.pdf')

    return log_probs


def bar_chart(log_probs, savefile=None):
    log_probs = pd.DataFrame(log_probs).T
    log_probs.index = log_probs.index.map(lambda env: PRETTY_ENVS.get(env, env))
    log_probs = log_probs.rename(columns=PRETTY_COLS)
    log_probs = log_probs.loc[PRETTY_ENVS.values(), PRETTY_COLS.values()]

    log_probs.index.name = 'Environment'
    log_probs.columns.name = 'Opponent'
    longform = log_probs.stack().reset_index()
    longform = longform.rename(columns={0: 'Mean Log Probability'})

    width, height = plt.rcParams.get('figure.figsize')
    legend_height = 0.4
    left_margin_in = 0.55
    top_margin_in = legend_height + 0.05
    bottom_margin_in = 0.5
    gridspec_kw = {
        'left': left_margin_in / width,
        'top': 1 - (top_margin_in / height),
        'bottom': bottom_margin_in / height,
    }
    fig, ax = plt.subplots(1, 1, gridspec_kw=gridspec_kw)

    # Make colors consistent with previous figures
    standard_cycle = list(plt.rcParams['axes.prop_cycle'])
    palette = {label: standard_cycle[HUE_ORDER.index(label)]['color']
               for label in PRETTY_COLS.values()}

    # Actually plot
    sns.barplot(x='Environment', y='Mean Log Probability',
                hue='Opponent', data=longform, palette=palette)
    util.rotate_labels(ax, xrot=0)

    # Plot our own legend
    ax.get_legend().remove()
    legend_entries = ax.get_legend_handles_labels()
    util.outside_legend(legend_entries, 3, fig, ax, ax,
                        legend_padding=0.05, legend_height=0.6,
                        handletextpad=0.2)

    if savefile is not None:
        fig.savefig(savefile)

    return fig


def heatmap_plot(env_name, metric, victim=1, savefile=None, error_val=-1):
    n_component_grid = [5, 10, 20, 40, 80]
    covariance_grid = ['diag', 'full']
    metric_grid = np.zeros(shape=(len(n_component_grid), len(covariance_grid)))
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
    fig = plt.figure(figsize=(10, 7))
    sns.heatmap(ll_df, annot=True)
    plt.title(f"HP Search on {env_name}, Victim {victim}: {metric_name}")
    if savefile is not None:
        fig.savefig(savefile)


def main():
    output_dir = "data/density/visualize"
    os.makedirs(output_dir, exist_ok=True)

    styles = ['paper', 'density_twocol']
    sns.set_style("whitegrid")
    for style in styles:
        plt.style.use(STYLES[style])

    def train_bic_in_millions(x):
        return x['train_bic'] / 1000000

    log_probs = {}
    for env in ['KickAndDefend', 'SumoAnts', 'SumoHumans', 'YouShallNotPassHumans']:
        heatmap_plot(env_name=env, metric=train_bic_in_millions,
                     savefile=f"{output_dir}/{env}_train_bic.pdf")
        heatmap_plot(env_name=env, metric='validation_log_likelihood',
                     savefile=f"{output_dir}/{env}_validation_log_likelihood.pdf")

        savefile = f"{output_dir}/{env}_20_full_comparative_density"
        log_probs[env] = comparative_densities(env_name=env, victim='1', n_components=20,
                                               covariance='full', savefile=savefile,
                                               cutoff_point=-1000)

    with open(f'{output_dir}/log_probs.json', 'w') as f:
        json.dump(log_probs, f)

    bar_chart(log_probs, savefile=f"{output_dir}/bar_chart.pdf")


if __name__ == "__main__":
    main()
