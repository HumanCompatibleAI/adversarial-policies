from distutils.dir_util import copy_tree
import functools
import logging
import os.path

import matplotlib.pyplot as plt
import pandas as pd
import sacred
from sacred import Experiment
from sacred.observers import FileStorageObserver
import seaborn as sns

from aprl.envs import VICTIM_INDEX
from aprl.envs.gym_compete import is_symmetric
from aprl.visualize import styles as vis_styles
from aprl.visualize import tb, util

# TODO(adam): remove once sacred issue #499 closed
sacred.SETTINGS.CONFIG.READ_ONLY_CONFIG = False

logger = logging.getLogger("aprl.visualize.training")
visualize_training_ex = Experiment("visualize_training")

# Helper methods


def save_figs(fig_dir, figs):
    figs = [(",".join(tuple(str(x) for x in key)), fig) for key, fig in figs]
    return list(util.save_figs(fig_dir, figs))


# Plotting


def _aggregate_data(data, xcol, ycols, in_split_keys, data_fns, subsample=100000):
    if data_fns is None:
        data_fns = []

    dfs = []
    for d in data:
        events = d["events"]
        df = pd.DataFrame(events)

        df = df.set_index(xcol)
        df = df[~df.index.duplicated(keep="first")]
        df = df.dropna(how="any")

        s = (df.index / subsample).astype(int)
        df = df.groupby(s).mean()
        df.index = df.index * subsample

        for k in in_split_keys:
            df[k] = d["config"][k]

        for data_fn in data_fns:
            df = data_fn(d["config"]["env_name"], df)

        for k in in_split_keys:
            df = df.set_index(k, append=True)
        df = df.sort_index(axis=0).sort_index(axis=1)

        dfs.append(df)

    longform = pd.concat(dfs)
    longform = longform[ycols]
    longform = longform.sort_index()
    longform = longform.reset_index()

    return longform


def lineplot_multi_fig(
    outer_key,
    data,
    xcol,
    ycols,
    ci,
    in_split_keys,
    plot_cfg=None,
    data_fns=None,
    plot_fns=None,
    **kwargs,
):
    """General method for line plotting TensorBoard datasets with smoothing and subsampling.

    Returns one figure for each plot."""
    if plot_fns is None:
        plot_fns = []

    # Aggregate data and convert to 'tidy' or longform format Seaborn expects
    longform = _aggregate_data(data, xcol, ycols, in_split_keys, data_fns)

    # Plot one figure per ycol
    for ycol in ycols:
        gridspec = {
            "left": 0.22,
            "bottom": 0.22,
        }
        fig, ax = plt.subplots(gridspec_kw=gridspec)

        sns.lineplot(x=xcol, y=ycol, data=longform, ci=ci, linewidth=1, label="Adv", **kwargs)
        for plot_fn in plot_fns:
            plot_fn(locals(), ax)

        yield (ycol,), fig


def lineplot_monolithic(
    outer_key,
    data,
    xcol,
    ycols,
    ci,
    in_split_keys,
    plot_cfg,
    data_fns=None,
    plot_fns=None,
    **kwargs,
):
    """General method for line plotting TensorBoard datasets with smoothing and subsampling.

    Returns a single figure, with plots on multiple axes. This is much harder to get right,
    but produces more compact figures that are good for publications."""
    assert len(ycols) == 1
    ycol = ycols[0]

    if plot_fns is None:
        plot_fns = []

    # Aggregate data and convert to 'tidy' or longform format Seaborn expects
    longform = _aggregate_data(data, xcol, ycols, in_split_keys, data_fns)

    subplot_cfg = plot_cfg["subplots"]
    nrows = len(subplot_cfg)
    ncols = len(subplot_cfg[0])
    assert all(len(x) <= ncols for x in subplot_cfg)

    width, height = plt.rcParams.get("figure.figsize")
    bottom_margin_in = 0.4
    top_margin_in = 0.5
    gridspec_kw = {
        "wspace": 0.15,
        "left": 0.08,
        "right": 0.98,
        "top": 1 - (top_margin_in / height),
        "bottom": bottom_margin_in / height,
    }
    fig, axs = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        sharex=True,
        sharey=True,
        squeeze=False,
        gridspec_kw=gridspec_kw,
        figsize=(width, height),
    )

    for i, cfg_row in enumerate(subplot_cfg):
        for j, cfg in enumerate(cfg_row):
            subset = longform
            for key, val in cfg["filter"].items():
                assert key in in_split_keys
                subset = subset[subset[key] == val]

            ax = axs[i][j]
            if plot_cfg.get("aggregated", True):
                group = subset.groupby(subset[xcol])[ycol]
                group_min, group_median, group_max = group.min(), group.median(), group.max()
                ax.fill_between(x=group_median.index, y1=group_min, y2=group_max, alpha=0.4)
                group_median.plot(label=vis_styles.PRETTY_LABELS["Adv"], ax=ax)
            else:
                sns.lineplot(
                    x=xcol,
                    y=ycol,
                    data=subset,
                    ci=ci,
                    linewidth=1,
                    ax=ax,
                    legend="full",
                    hue="seed",
                    **kwargs,
                )
                # Plot legend in order to add handles, but remove as we'll draw our own later
                ax.get_legend().remove()
            for plot_fn in plot_fns:
                plot_fn(locals(), ax)
            if "title" in cfg:
                ax.set_title(cfg["title"])

    for unused_j in range(j + 1, ncols):
        axs[i][unused_j].remove()

    legend_entries = ax.get_legend_handles_labels()
    legend_ncol = len(legend_entries[0])
    util.outside_legend(legend_entries, legend_ncol, fig, axs[0][0], axs[0][-1])

    yield ("monolithic",), fig


def _win_rate_data_convert(env_name, df):
    """Convert win proportions to percentage and rename columns."""
    victim_index = VICTIM_INDEX[env_name]

    COLUMNS = {
        f"game_win{victim_index}": "Victim Win",
        f"game_win{1 - victim_index}": "Opponent Win",
        "game_tie": "Ties",
    }
    df = df.rename(columns=COLUMNS)

    for col in COLUMNS.values():
        df[col] *= 100

    return df


def _win_rate_make_fig(x, lineplot_fn, fig_dir, **kwargs):
    outer_key, data = x
    generator = lineplot_fn(outer_key, data, **kwargs)
    figs = [(outer_key + inner_key, fig) for inner_key, fig in generator]
    return save_figs(fig_dir, figs)


def _win_rate_labels(variables, ax):
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Win rate (%)")


@visualize_training_ex.capture
def win_rate(
    tb_dir,
    lineplot_fn,
    out_split_keys,
    in_split_keys,
    data_fns,
    plot_fns,
    fig_dir,
    xcol,
    ci,
    plot_cfg,
    ycols=None,
    **kwargs,
):
    if ycols is None:
        ycols = ["Opponent Win", "Victim Win", "Ties"]

    configs, events = tb.load_tb_data(tb_dir, keys=["game_win0", "game_win1", "game_tie"])
    make_fig_wrapped = functools.partial(
        _win_rate_make_fig,
        lineplot_fn=lineplot_fn,
        xcol=xcol,
        ycols=ycols,
        ci=ci,
        plot_cfg=plot_cfg,
        in_split_keys=in_split_keys,
        data_fns=[_win_rate_data_convert] + data_fns,
        plot_fns=[_win_rate_labels] + plot_fns,
        fig_dir=fig_dir,
        **kwargs,
    )
    return tb.tb_apply(configs, events, split_keys=out_split_keys, fn=make_fig_wrapped)


LINESTYLES = {
    "Zoo": "-.",
    "Rand": "--",
    "Zero": ":",
}


def plot_baselines(env_name, victim_path, ycol, ax, baseline):
    victim_name = f"Zoo{victim_path}" if is_symmetric(env_name) else f"ZooV{victim_path}"
    scores = baseline.loc[(env_name, victim_name), :]
    num_episodes = util.num_episodes(scores)
    scores = scores / num_episodes * 100  # convert to percent

    scores = scores[ycol]
    zoo_mask = scores.index.str.startswith("Zoo")
    zoo_score = scores.loc[zoo_mask].max()
    scores["Zoo"] = zoo_score
    scores = scores.loc[["Zoo", "Rand", "Zero"]]

    num_lines = len(ax.get_legend_handles_labels()[0])
    for i, (opponent, score) in enumerate(scores.items()):
        label = vis_styles.PRETTY_LABELS[opponent]
        ax.axhline(
            y=score,
            label=label,
            color=f"C{num_lines + i}",
            linewidth=1,
            linestyle=LINESTYLES[opponent],
        )


def plot_baselines_multi_fig(variables, ax, baseline):
    outer_key = variables["outer_key"]
    env_name, victim_path = outer_key
    ycol = variables["ycol"]
    return plot_baselines(env_name, victim_path, ycol, ax, baseline)


def win_rate_per_victim_env(tb_dir, baseline):
    out_split_keys = ["env_name", "victim_path"]
    in_split_keys = ["seed"]

    plot_baselines_wrapped = functools.partial(plot_baselines_multi_fig, baseline=baseline)
    return win_rate(
        tb_dir,
        lineplot_multi_fig,
        out_split_keys,
        in_split_keys,
        data_fns=[],
        plot_fns=[plot_baselines_wrapped],
    )


def win_rate_per_env(tb_dir, baseline):
    out_split_keys = ["env_name"]
    in_split_keys = ["victim_path", "seed"]

    return win_rate(
        tb_dir,
        lineplot_multi_fig,
        out_split_keys,
        in_split_keys,
        data_fns=[],
        plot_fns=[],
        hue="victim_path",
    )


def plot_baselines_monolithic(variables, ax, baseline):
    var_filter = variables["cfg"]["filter"]
    env_name = var_filter["env_name"]
    victim_path = var_filter["victim_path"]
    ycol = variables["ycol"]
    return plot_baselines(env_name, victim_path, ycol, ax, baseline)


def opponent_win_rate_per_victim_env(tb_dir, baseline):
    out_split_keys = []
    in_split_keys = ["env_name", "victim_path", "seed"]

    plot_baselines_wrapped = functools.partial(plot_baselines_monolithic, baseline=baseline)
    return win_rate(
        tb_dir,
        lineplot_monolithic,
        out_split_keys,
        in_split_keys,
        data_fns=[],
        plot_fns=[plot_baselines_wrapped],
        ycols=["Opponent Win"],
    )


# Sacred config and commands


@visualize_training_ex.config
def default_config():
    command = win_rate_per_victim_env
    fig_dir = os.path.join("data", "figs", "training")
    plot_cfg = None
    score_paths = [
        os.path.join("data", "aws", "score_agents", "normal", x)
        for x in ["fixed_baseline.json", "zoo_baseline.json"]
    ]
    tb_dir = None
    styles = ["paper", "a4"]
    xcol = "step"
    envs = None
    ci = 95  # 95% confidence interval
    seed = 0  # we don't use it for anything, but stop config changing each time as we version it
    _ = locals()  # quieten flake8 unused variable warning
    del _


def _summary_plot():
    command = opponent_win_rate_per_victim_env
    # Plot for each environment against victim with median (adversary win rate - best zoo win rate)
    plot_cfg = {
        "subplots": [
            [
                {
                    "filter": {"env_name": "multicomp/KickAndDefend-v0", "victim_path": 2},
                    "title": "Kick and Defend 2",
                },
                {
                    "filter": {"env_name": "multicomp/YouShallNotPassHumans-v0", "victim_path": 1},
                    "title": "You Shall Not Pass 1",
                },
                {
                    "filter": {"env_name": "multicomp/SumoHumansAutoContact-v0", "victim_path": 2},
                    "title": "Sumo Humans 2",
                },
            ],
        ]
    }
    ci = None
    tb_dir = os.path.join("data", "aws", "multi_train", "paper", "20190429_011349")
    return locals()


@visualize_training_ex.named_config
def paper_config():
    locals().update(_summary_plot())
    fig_dir = os.path.expanduser("~/dev/adversarial-policies-paper/figs/training_single")
    styles = ["paper", "monolithic"]
    _ = locals()  # quieten flake8 unused variable warning
    del _


@visualize_training_ex.named_config
def slides_config():
    locals().update(_summary_plot())
    fig_dir = os.path.expanduser("~/tmp/adversarial_slides")
    styles = ["paper", "slides"]
    _ = locals()  # quieten flake8 unused variable warning
    del _


def _gen_cell(env_name, victim_path):
    return {
        "filter": {"env_name": env_name, "victim_path": victim_path},
        "title": f"{vis_styles.PRETTY_ENV.get(env_name, env_name)} {victim_path}",
    }


@visualize_training_ex.named_config
def supplementary_config():
    command = opponent_win_rate_per_victim_env
    fig_dir = os.path.expanduser("~/dev/adversarial-policies-paper/figs/training")
    plot_cfg = {
        "subplots": [
            [
                _gen_cell("multicomp/KickAndDefend-v0", 1),
                _gen_cell("multicomp/KickAndDefend-v0", 2),
                _gen_cell("multicomp/KickAndDefend-v0", 3),
            ],
            [
                _gen_cell("multicomp/SumoHumansAutoContact-v0", 1),
                _gen_cell("multicomp/SumoHumansAutoContact-v0", 2),
                _gen_cell("multicomp/SumoHumansAutoContact-v0", 3),
            ],
            [
                _gen_cell("multicomp/SumoAntsAutoContact-v0", 1),
                _gen_cell("multicomp/SumoAntsAutoContact-v0", 2),
                _gen_cell("multicomp/SumoAntsAutoContact-v0", 3),
            ],
            [
                _gen_cell("multicomp/SumoAntsAutoContact-v0", 4),
                _gen_cell("multicomp/YouShallNotPassHumans-v0", 1),
            ],
        ]
    }
    styles = ["paper"]
    ci = None
    tb_dir = os.path.join("data", "aws", "multi_train", "paper", "20190429_011349")
    _ = locals()  # quieten flake8 unused variable warning
    del _


@visualize_training_ex.named_config
def debug_config():
    fig_dir = "data/debug/figs_training"
    styles = ["paper", "threecol"]
    tb_dir = os.path.join("data", "debug", "best_guess")
    ci = None  # much faster
    _ = locals()  # quieten flake8 unused variable warning
    del _


@visualize_training_ex.named_config
def debug_paper_config():
    command = opponent_win_rate_per_victim_env
    fig_dir = "data/debug/figs_training_single"
    plot_cfg = [
        [
            {
                "filter": {"env_name": "multicomp/KickAndDefend-v0", "victim_path": 1},
                "title": "Kick and Defend 1",
            },
            {
                "filter": {"env_name": "multicomp/KickAndDefend-v0", "victim_path": 2},
                "title": "Kick and Defend 2",
            },
        ]
    ]
    ci = None  # much faster
    styles = ["paper", "monolithic"]
    tb_dir = os.path.join("data", "debug", "best_guess")
    _ = locals()  # quieten flake8 unused variable warning
    del _


@visualize_training_ex.main
def visualize_score(command, styles, tb_dir, score_paths, fig_dir):
    baseline = [util.load_datasets(path) for path in score_paths]
    baseline = pd.concat(baseline)

    sns.set_style("whitegrid")
    for style in styles:
        plt.style.use(vis_styles.STYLES[style])

    out_paths = command(tb_dir, baseline)
    for out_path in out_paths:
        visualize_training_ex.add_artifact(filename=out_path)

    for observer in visualize_training_ex.observers:
        if hasattr(observer, "dir"):
            logger.info(f"Copying from {observer.dir} to {fig_dir}")
            copy_tree(observer.dir, fig_dir)
            break


def main():
    observer = FileStorageObserver(os.path.join("data", "sacred", "visualize_training"))
    visualize_training_ex.observers.append(observer)
    visualize_training_ex.run_commandline()


if __name__ == "__main__":
    main()
