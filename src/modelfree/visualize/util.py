import json
import logging
import os.path

import matplotlib.backends.backend_pdf
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from modelfree.configs.multi.common import VICTIM_INDEX

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
            'Adversary Win': v[f'win{our_index}'],
        }
        if victim_index == 1:  # victim is agent_b
            k = (env_name, agent_b_type, agent_b_path, agent_a_type, agent_a_path)
        assert k not in res
        res[k] = v

    df = pd.DataFrame(res).T
    df.index.names = ['env_name', 'victim_type', 'victim_path',
                      'adversary_type', 'adversary_path']
    cols = ['Adversary Win', 'Victim Win', 'Ties']
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

# Visualization


def apply_per_env(scores, fn, *args, **kwargs):
    envs = scores.index.levels[0]
    for env_name in envs:
        single_env = scores.loc[env_name]
        fig = fn(single_env, *args, **kwargs)
        pretty_env = PRETTY_ENV.get(env_name, env_name)
        fig.suptitle(pretty_env)
        yield env_name, fig


def save_figs(out_dir, generator):
    combined = matplotlib.backends.backend_pdf.PdfPages(os.path.join(out_dir, 'combined.pdf'))
    for env_name, fig in generator:
        out_path = os.path.join(out_dir, env_name.replace('/', '_') + '.pdf')
        logger.info(f"Saving to '{out_path}'")
        fig.savefig(out_path)
        combined.savefig(fig)
    combined.close()


def heatmap(single_env):
    # Consistent color map scale
    vmin = 0
    num_episodes = pd.unique(single_env.sum(axis=1))
    assert len(num_episodes) == 1
    vmax = num_episodes[0]

    # Figure layout calculations
    cols = single_env.columns
    ncols = len(cols) + 1
    gridspec_kw = {
        'top': 0.8,
        'bottom': 0.25,
        'wspace': 0.05,
        'width_ratios': [1.0] * len(cols) + [1/15],
    }
    width, height = plt.rcParams.get('figure.figsize')
    height = min(height, width / len(cols))

    # Actually plot the heatmap
    fig, axs = plt.subplots(ncols=ncols, gridspec_kw=gridspec_kw, figsize=(width, height))
    cbar_ax = axs[-1]
    for i, col in enumerate(single_env.columns):
        ax = axs[i]
        yaxis = i == 0
        cbar = i == len(cols) - 1
        sns.heatmap(single_env[col].unstack(), vmin=vmin, vmax=vmax, annot=True, fmt='d',
                    ax=ax, cbar=cbar, cbar_ax=cbar_ax, yticklabels=yaxis)
        ax.get_yaxis().set_visible(yaxis)
        ax.set_title(col)

    return fig
