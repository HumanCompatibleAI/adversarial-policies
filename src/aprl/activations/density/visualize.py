"""Visualizes fitted density model: bar charts, CDFs and others."""

import collections
from glob import glob
import json
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from aprl.visualize import styles as vis_styles
from aprl.visualize import util

logger = logging.getLogger("aprl.density.visualize")

TRAIN_ID = "zoo_1"
DENSITY_DIR = "data/density"

ENV_NAMES = ["KickAndDefend", "SumoHumans", "YouShallNotPassHumans"]

PRETTY_ENVS = collections.OrderedDict(
    [
        ("KickAndDefend", "Kick and\nDefend"),
        ("YouShallNotPassHumans", "You Shall\nNot Pass"),
        ("SumoHumans", "Sumo\nHumans"),
    ]
)

PRETTY_OPPONENTS = collections.OrderedDict(
    [
        ("zoo_1_train", "Zoo*1T"),
        ("zoo_1_test", "Zoo*1V"),
        ("zoo_2", "Zoo*2"),
        ("zoo_3", "Zoo*3"),
        ("random_none", "Rand"),
        ("ppo2_1", "Adv"),
    ]
)

CYCLE_ORDER = ["Adv", "Zoo*1T", "Rand", "Zoo*1V", "Zoo*2", "Zoo*3"]
BAR_ORDER = ["Zoo*1T", "Zoo*1V", "Zoo*2", "Zoo*3", "Rand", "Adv"]


def get_full_directory(env, victim_id, n_components, covariance):
    """Finds directory containing result for specified environment and parameters.

    :param env: (str) environment name, e.g. SumoHumans
    :param victim_id: (str) victim ID, e.g. zoo_1
    :param n_components: (int) number of components of GMM
    :param covariance: (str) type of covariance matrix, e.g. diag.
    :return A path to the directory"""
    hp_dir = f"{DENSITY_DIR}/gmm_{n_components}_components_{covariance}"
    print(hp_dir)
    exp_dir = glob(hp_dir + "/*")[0]
    env_dir = f"{env}-v0_victim_zoo_{victim_id}"
    full_env_dir = os.path.join(exp_dir, "fitted", env_dir)
    return full_env_dir


def load_metadata(env, victim_id, n_components, covariance):
    """Load metadata for specified environment and parameters.
    Parameters are the same as get_full_directory."""
    full_env_dir = get_full_directory(env, victim_id, n_components, covariance)
    metadata_path = os.path.join(full_env_dir, "metadata.csv")
    logger.debug(f"Loading from {metadata_path}")
    df = pd.read_csv(metadata_path)

    # We want to evaluate on both the train and test set for the train opponent.
    # To disambiguate, we'll change the opponent_id for the train opponent in the test set.
    # For all other opponents, doesn't matter if we evaluate on "train" or "test" set
    # as they were trained on neither; we use the test set.
    is_train_opponent = df["opponent_id"] == TRAIN_ID
    # Rewrite opponent_id for train opponent in test set
    df.loc[is_train_opponent & ~df["is_train"], "opponent_id"] = TRAIN_ID + "_test"
    # Discard all non-test data, except for train opponent
    df = df.loc[is_train_opponent | ~df["is_train"]]

    return df


def load_metrics_dict(env, victim_id, n_components, covariance):
    """Load metrics for specified environment and parameters.
    See get_full_directory for argument descriptions."""
    full_env_dir = get_full_directory(env, victim_id, n_components, covariance)
    metrics_path = os.path.join(full_env_dir, "metrics.json")
    logger.debug(f"Loading from {metrics_path}")
    with open(metrics_path, "r") as f:
        metric_dict = json.load(f)
    return metric_dict


def comparative_densities(
    env, victim_id, n_components, covariance, cutoff_point=None, savefile=None, **kwargs
):
    """PDF of different opponents density distribution.
    For unspecified parameters, see get_full_directory.

    :param cutoff_point: (float): left x-limit.
    :param savefile: (None or str) path to save figure to.
    :param kwargs: (dict) passed through to sns.kdeplot."""
    df = load_metadata(env, victim_id, n_components, covariance)
    fig = plt.figure(figsize=(10, 7))

    grped = df.groupby("opponent_id")
    for name, grp in grped:
        # clean up random_none to just random
        name = name.replace("_none", "")
        avg_log_proba = np.mean(grp["log_proba"])
        sns.kdeplot(grp["log_proba"], label=f"{name}: {round(avg_log_proba, 2)}", **kwargs)

    xmin, xmax = plt.xlim()
    xmin = max(xmin, cutoff_point)
    plt.xlim((xmin, xmax))

    plt.suptitle(f"{env} Densities, Victim Zoo {victim_id}: Trained on Zoo 1", y=0.95)
    plt.title("Avg Log Proba* in Legend")

    if savefile is not None:
        fig.savefig(f"{savefile}.pdf")


def heatmap_plot(metric, env, victim_id="1", savefile=None):
    """Heatmap of metric for all possible hyperparameters, against victim.

    :param metric: (str) a key into metrics.json
    :param env: (str) environment name
    :param victim_id: (str) victim ID
    :param savefile: (None or str) path to save figure to.
    """
    n_component_grid = [5, 10, 20, 40, 80]
    covariance_grid = ["diag", "full"]
    metric_grid = np.zeros(shape=(len(n_component_grid), len(covariance_grid)))
    if isinstance(metric, str):
        metric_name = metric
    else:
        metric_name = metric.__name__

    for i, n_components in enumerate(n_component_grid):
        for j, covariance in enumerate(covariance_grid):
            try:
                metrics = load_metrics_dict(env, victim_id, n_components, covariance)
                if isinstance(metric, str):
                    metric_grid[i][j] = metrics[metric]
                else:
                    metric_grid[i][j] = metric(metrics)
            except FileNotFoundError:
                logger.warning(
                    f"Hit exception on {env}, {n_components} components {covariance}", exc_info=True
                )
                metric_grid[i][j] = np.nan

    ll_df = pd.DataFrame(metric_grid, index=n_component_grid, columns=covariance_grid)
    fig = plt.figure(figsize=(10, 7))
    sns.heatmap(ll_df, annot=True, mask=ll_df.isnull())
    plt.title(f"HP Search on {env}, Victim {victim_id}: {metric_name}")
    if savefile is not None:
        fig.savefig(savefile)


def bar_chart(envs, victim_id, n_components, covariance, savefile=None):
    """Bar chart of mean log probability for all opponent types, grouped by environment.
    For unspecified parameters, see get_full_directory.

    :param envs: (list of str) list of environments.
    :param savefile: (None or str) path to save figure to.
    """
    dfs = []
    for env in envs:
        df = load_metadata(env, victim_id, n_components, covariance)
        df["Environment"] = PRETTY_ENVS.get(env, env)
        dfs.append(df)
    longform = pd.concat(dfs)
    longform["opponent_id"] = longform["opponent_id"].apply(PRETTY_OPPONENTS.get)
    longform = longform.reset_index(drop=True)

    width, height = plt.rcParams.get("figure.figsize")
    legend_height = 0.4
    left_margin_in = 0.55
    top_margin_in = legend_height + 0.05
    bottom_margin_in = 0.5
    gridspec_kw = {
        "left": left_margin_in / width,
        "top": 1 - (top_margin_in / height),
        "bottom": bottom_margin_in / height,
    }
    fig, ax = plt.subplots(1, 1, gridspec_kw=gridspec_kw)

    # Make colors consistent with previous figures
    standard_cycle = list(plt.rcParams["axes.prop_cycle"])
    palette = {
        label: standard_cycle[CYCLE_ORDER.index(label)]["color"]
        for label in PRETTY_OPPONENTS.values()
    }

    # Actually plot
    sns.barplot(
        x="Environment",
        y="log_proba",
        hue="opponent_id",
        order=PRETTY_ENVS.values(),
        hue_order=BAR_ORDER,
        data=longform,
        palette=palette,
        errwidth=1,
    )
    ax.set_ylabel("Mean Log Probability Density")
    plt.locator_params(axis="y", nbins=4)
    util.rotate_labels(ax, xrot=0)

    # Plot our own legend
    ax.get_legend().remove()
    legend_entries = ax.get_legend_handles_labels()
    util.outside_legend(
        legend_entries, 3, fig, ax, ax, legend_padding=0.05, legend_height=0.6, handletextpad=0.2
    )

    if savefile is not None:
        fig.savefig(savefile)

    return fig


def plot_heatmaps(output_dir):
    def train_bic_in_millions(x):
        return x["train_bic"] / 1000000

    for env in ENV_NAMES:
        heatmap_plot(
            env=env, metric=train_bic_in_millions, savefile=f"{output_dir}/{env}_train_bic.pdf"
        )
        heatmap_plot(
            env=env,
            metric="validation_log_likelihood",
            savefile=f"{output_dir}/{env}_validation_log_likelihood.pdf",
        )


def plot_comparative_densities(output_dir):
    for env in ENV_NAMES:
        comparative_densities(
            env=env,
            victim_id="1",
            n_components=20,
            covariance="full",
            cutoff_point=-1000,
            savefile=f"{output_dir}/{env}_20_full_comparative_density",
        )


def main():
    logging.basicConfig(level=logging.DEBUG)

    output_dir = "data/density/visualize"
    os.makedirs(output_dir, exist_ok=True)

    styles = ["paper", "density_twocol"]
    sns.set_style("whitegrid")
    for style in styles:
        plt.style.use(vis_styles.STYLES[style])

    plot_heatmaps(output_dir)
    plot_comparative_densities(output_dir)
    bar_chart(
        ENV_NAMES,
        victim_id="1",
        n_components=20,
        covariance="full",
        savefile=f"{output_dir}/bar_chart.pdf",
    )


if __name__ == "__main__":
    main()
