import json
import logging
import os.path

import matplotlib.backends.backend_pdf
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from modelfree.configs.multi.common import VICTIM_INDEX
from modelfree.visualize.styles import STYLES

logger = logging.getLogger('modelfree.visualize.util')

PRETTY_ENV = {
    'multicomp/KickAndDefend-v0': 'Kick and Defend',
    'multicomp/SumoAntsAutoContact-v0': 'Sumo Ants',
    'multicomp/SumoHumansAutoContact-v0': 'Sumo Humans',
    'multicomp/YouShallNotPassHumans-v0': 'You Shall Not Pass',
}

# Data loading & manipulation


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def load_scores(path, reindex_victim=False):
    raw = load_json(path)
    res = {}
    for d in raw:
        assert d.keys() == {'k', 'v'}
        k, v = tuple(d['k']), d['v']
        env_name, agent_a_type, agent_a_path, agent_b_type, agent_b_path = k
        victim_index = VICTIM_INDEX[env_name]
        our_index = 1 - victim_index
        v = {
            'Ties': v['ties'],
            'Victim Win': v[f'win{victim_index}'],
            'Opponent Win': v[f'win{our_index}'],
        }
        if victim_index == 1:  # victim is agent_b
            k = (env_name, agent_b_type, agent_b_path, agent_a_type, agent_a_path)
        assert k not in res
        res[k] = v

    df = pd.DataFrame(res).T
    df.index.names = ['env_name', 'victim_type', 'victim_path',
                      'adversary_type', 'adversary_path']
    cols = ['Opponent Win', 'Victim Win', 'Ties']
    return df.loc[:, cols].copy()


def load_zoo_baseline(path):
    """Returns DataFrame with index (env_name, agent 0 zoo path, agent 1 zoo path)."""
    scores = load_scores(path)
    # drop the 'types' which are always 'zoo'
    scores.index = scores.index.droplevel(level=3).droplevel(level=1)
    scores.index.names = ['Environment', 'Agent 0', 'Agent 1']
    assert scores.index.is_unique
    return scores


def load_fixed_baseline(path):
    """Returns DataFrame with index (env_name, victim path, opponent type)."""
    scores = load_scores(path)
    # drop the always 'zoo' type for victim and the
    scores.index = scores.index.droplevel(level=4).droplevel(level=1)
    scores.index.names = ['Environment', 'Victim', 'Baseline']
    assert scores.index.is_unique
    return scores


def load_transfer_baseline(path):
    """Returns DataFrame with index (env_name, victim path, adversary trained on victim path)."""
    scores = load_scores(path)
    # drop the always 'zoo' and 'ppo2' type for victim and adversary
    scores.index = scores.index.droplevel(level=3).droplevel(level=1)
    adversary_paths = scores.index.get_level_values(2)
    trained_on = adversary_paths.str.replace('.*victim_path=', '').str.replace('[^0-9].*', '')
    scores.index = pd.MultiIndex.from_tuples([(x[0], x[1], path)
                                              for x, path in zip(scores.index, trained_on)])
    scores.index.names = ['Environment', 'Victim', 'Adversary For']
    assert scores.index.is_unique
    return scores


def prefix_level(df, prefix, level):
    levels = prefix + df.index.levels[level]
    df.index = df.index.set_levels(levels, level=level)
    return df


def combine_all(fixed, zoo, transfer):
    fixed = fixed.copy()
    zoo = zoo.copy()
    transfer = transfer.copy()

    fixed.index = fixed.index.set_levels(['Rand', 'Zero'], level=2)
    zoo = prefix_level(zoo, 'Zoo', 2)
    transfer = prefix_level(transfer, 'Adv', 2)

    combined = pd.concat([transfer, zoo, fixed], axis=0)
    combined = prefix_level(combined, 'Zoo', 1)
    combined = combined.sort_index(level=0, sort_remaining=False)
    combined.index = combined.index.set_names('Opponent', level=2)
    return combined


def load_datasets(score_dir):
    fixed = load_fixed_baseline(os.path.join(score_dir, 'fixed_baseline.json'))
    zoo = load_zoo_baseline(os.path.join(score_dir, 'zoo_baseline.json'))
    transfer = load_transfer_baseline(os.path.join(score_dir, 'adversary_transfer.json'))
    return combine_all(fixed, zoo, transfer)

# Visualization


def apply_per_env(scores, fn, *args, suptitle=True, **kwargs):
    envs = scores.index.levels[0]
    for i, env_name in enumerate(envs):
        single_env = pd.DataFrame(scores.loc[env_name])
        single_env.name = env_name
        fig = fn(single_env, *args, **kwargs)

        if suptitle:
            pretty_env = PRETTY_ENV.get(env_name, env_name)
            fig.suptitle(pretty_env)

        yield env_name, fig


def save_figs(out_dir, generator, combine=False):
    if combine:
        combined = matplotlib.backends.backend_pdf.PdfPages(os.path.join(out_dir, 'combined.pdf'))
    for fig_name, fig in generator:
        fig_name = fig_name.replace('/', '_').replace(' ', '_')
        out_path = os.path.join(out_dir, fig_name + '.pdf')
        logger.info(f"Saving to '{out_path}'")
        fig.savefig(out_path)
        if combine:
            combined.savefig(fig)
        yield out_path
        plt.close(fig)
    if combine:
        combined.close()


def num_episodes(single_env):
    """Compute number of episodes in dataset"""
    num_episodes = pd.unique(single_env.sum(axis=1))
    assert len(num_episodes) == 1
    return num_episodes[0]


def heatmap_full(single_env, cols=None):
    # Figure layout calculations
    if cols is None:
        cols = single_env.columns
    ncols = len(cols) + 1

    num_xticks = len(set(single_env.index.get_level_values(1)))
    rotate_xticks = num_xticks > 4
    gridspec_kw = {
        'top': 0.8,
        'bottom': 0.35 if rotate_xticks else 0.25,
        'wspace': 0.05,
        'width_ratios': [1.0] * len(cols) + [1/15],
    }
    width, height = plt.rcParams.get('figure.figsize')
    height = min(height, width / len(cols))

    # Consistent color map scale
    vmax = num_episodes(single_env)

    # Actually plot the heatmap
    fig, axs = plt.subplots(ncols=ncols, gridspec_kw=gridspec_kw, figsize=(width, height))
    cbar_ax = axs[-1]
    for i, col in enumerate(cols):
        ax = axs[i]
        yaxis = i == 0
        cbar = i == len(cols) - 1
        sns.heatmap(single_env[col].unstack(), vmin=0, vmax=vmax,
                    annot=True, annot_kws={'fontsize': 6}, fmt='d',
                    ax=ax, cbar=cbar, cbar_ax=cbar_ax, yticklabels=yaxis)
        ax.get_yaxis().set_visible(yaxis)
        if len(cols) > 1:
            ax.set_title(col)

        if rotate_xticks:
            plt.xticks(rotation=90)

    return fig


def heatmap_one_col(single_env, col, cbar, ylabel):
    gridspec_kw = {
        'bottom': 0.35,
        'left': 0.23 if ylabel else 0.13,
    }
    if cbar:
        gridspec_kw.update({
            'width_ratios': (1.0, 0.1),
            'wspace': 0.2,
            'right': 0.75,
        })
        fig, (ax, cbar_ax) = plt.subplots(ncols=2, gridspec_kw=gridspec_kw)
    else:
        fig, ax = plt.subplots(1, gridspec_kw=gridspec_kw)
        cbar_ax = None

    single_env *= 100 / num_episodes(single_env)  # convert to percentages
    sns.heatmap(single_env[col].unstack(), vmin=0, vmax=100,
                annot=True, annot_kws={'fontsize': 8}, fmt='.0f',
                ax=ax, cbar=cbar, cbar_ax=cbar_ax)
    if not ylabel:
        ax.set_ylabel('')
    plt.xticks(rotation=90)

    return fig


# Argument parsing

def directory(path):
    if not os.path.exists(path):
        raise ValueError(f"Path '{path}' does not exist")
    if not os.path.isdir(path):
        raise ValueError(f"Path '{path}' is not a directory")
    return path


def style(key):
    if key not in STYLES:
        raise ValueError(f"Unrecognized style '{key}'")
    return key
