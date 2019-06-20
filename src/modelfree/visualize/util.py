import json
import logging
import os.path

import matplotlib.backends.backend_pdf
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from modelfree.envs import VICTIM_INDEX, gym_compete

logger = logging.getLogger('modelfree.visualize.util')

PRETTY_ENV = {
    'multicomp/KickAndDefend-v0': 'Kick and Defend',
    'multicomp/SumoAntsAutoContact-v0': 'Sumo Ants',
    'multicomp/SumoAnts-v0': 'Sumo Ants',
    'multicomp/SumoHumansAutoContact-v0': 'Sumo Humans',
    'multicomp/SumoHumans-v0': 'Sumo Humans',
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


def agent_index_suffix(env_name, victim_name, opponent_name):
    if not gym_compete.is_symmetric(env_name):
        if victim_name.startswith('Zoo'):
            victim_name = f'{victim_name[:-1]}V{victim_name[-1]}'
        if opponent_name.startswith('Zoo'):
            opponent_name = f'{opponent_name[:-1]}O{opponent_name[-1]}'
    return env_name, victim_name, opponent_name


def combine_all(fixed, zoo, transfer, victim_suffix, opponent_suffix):
    dfs = []

    if transfer is not None:
        transfer = transfer.copy()
        transfer = prefix_level(transfer, 'Adv' + opponent_suffix, 2)
        dfs.append(transfer)

    if zoo is not None:
        zoo = zoo.copy()
        zoo = prefix_level(zoo, 'Zoo', 2)
        dfs.append(zoo)

    if fixed is not None:
        fixed = fixed.copy()
        fixed.index = fixed.index.set_levels(['Rand', 'Zero'], level=2)
        dfs.append(fixed)

    combined = pd.concat(dfs, axis=0)
    combined = prefix_level(combined, 'Zoo' + victim_suffix, 1)
    combined = combined.sort_index(level=0, sort_remaining=False)
    combined.index = combined.index.set_names('Opponent', level=2)

    new_index = [agent_index_suffix(*entry) for entry in combined.index]
    combined.index = pd.MultiIndex.from_tuples(new_index)

    return combined


def load_datasets(timestamped_path, victim_suffix='', opponent_suffix=''):
    score_dir = os.path.dirname(timestamped_path)
    try:
        fixed_path = os.path.join(score_dir, 'fixed_baseline.json')
        fixed = load_fixed_baseline(fixed_path)
    except FileNotFoundError:
        logger.warning(f"No fixed baseline at '{fixed_path}'")
        fixed = None

    try:
        zoo_path = os.path.join(score_dir, 'zoo_baseline.json')
        zoo = load_zoo_baseline(zoo_path)
    except FileNotFoundError:
        logger.warning(f"No fixed baseline at '{zoo_path}'")
        zoo = None

    transfer = load_transfer_baseline(os.path.join(timestamped_path, 'adversary_transfer.json'))
    return combine_all(fixed, zoo, transfer,
                       victim_suffix=victim_suffix, opponent_suffix=opponent_suffix)

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


def rotate_labels(ax, xrot=90, yrot=0):
    for label in ax.get_xticklabels():
        label.set_rotation(xrot)
    for label in ax.get_yticklabels():
        label.set_rotation(yrot)


def outside_legend(legend_entries, legend_ncol, fig, ax_left, ax_right,
                   legend_padding=0.25, legend_height=0.3, **kwargs):
    width, height = fig.get_size_inches()

    pos_left = ax_left.get_position(original=True)
    pos_right = ax_right.get_position(original=True)
    legend_left = pos_left.x0
    legend_right = pos_right.x0 + pos_right.width
    legend_width = legend_right - legend_left
    legend_bottom = pos_left.y0 + pos_left.height + legend_padding / height
    legend_height = legend_height / height
    bbox = (legend_left, legend_bottom, legend_width, legend_height)
    fig.legend(*legend_entries, loc='lower center', ncol=legend_ncol,
               bbox_to_anchor=bbox, mode="expand",
               borderaxespad=0, frameon=True, **kwargs)


GROUPS = {
    'rows': [r'ZooV?[0-9]', r'ZooMV?[0-9]'],
    'cols': [r'^Adv[0-9]', r'^ZooO?[0-9]', r'Zero|Rand']
}


def _split_groups(df):
    group_members = {}
    num_matches = {}
    index = df.index.remove_unused_levels()
    for kind, groups in GROUPS.items():
        level = 1 if kind == 'cols' else 0
        level_values = index.levels[level]
        masks = [level_values.str.contains(pattern) for pattern in groups]
        group_members[kind] = [level_values[mask] for mask in masks]
        num_matches[kind] = [mask.sum() for mask in masks]
    return group_members, num_matches


class DogmaticNormalize(matplotlib.colors.Normalize):
    """Workaround heatmap resetting vmin and vmax internally."""
    def __init__(self, vmin, vmax):
        self._real_vmin = vmin
        self._real_vmax = vmax
        super(DogmaticNormalize, self).__init__(vmin, vmax, clip=True)

    def __call__(self, *args, **kwargs):
        self.vmin = self._real_vmin
        self.vmax = self._real_vmax
        return super(DogmaticNormalize, self).__call__(*args, **kwargs)


DIRECTIONS = {  # Which way is better?
    'Opponent Win': 1,  # bigger better
    'Victim Win': -1,  # smaller better
    'Ties': 0,  # neither
}


def _pretty_heatmap(single_env, col, cmap, fig, gridspec_kw,
                    xlabel=False, ylabel=False, cbar_width=0.0, yaxis=True):
    group_members, num_matches = _split_groups(single_env)
    single_kind = single_env[col].unstack()
    direction = DIRECTIONS[col]

    gridspec_kw = dict(gridspec_kw)
    gridspec_kw.update({
        'width_ratios': num_matches['cols'],
        'height_ratios': num_matches['rows'],
        'bottom': gridspec_kw['bottom'] + (0.02 if xlabel else 0.0),
        'left': gridspec_kw['left'] + (0.05 if ylabel else 0.0),
    })
    nrows = len(num_matches['rows'])
    ncols = len(num_matches['cols'])
    axs = fig.subplots(nrows=nrows, ncols=ncols, gridspec_kw=gridspec_kw)

    cbar = cbar_width > 0
    cbar_ax = None
    if cbar > 0:
        gs = matplotlib.gridspec.GridSpec(1, 1)
        cbar_ax = fig.add_subplot(gs[0, 0])
        margin_width = cbar_width * 1 / 9
        bar_width = cbar_width * 3 / 9
        gs.update(bottom=gridspec_kw['bottom'], top=gridspec_kw['top'],
                  left=gridspec_kw['right'] + margin_width,
                  right=gridspec_kw['right'] + margin_width + bar_width)

    norm = DogmaticNormalize(vmin=-10, vmax=100)
    for i, (row_axs, row_members) in enumerate(zip(axs, group_members['rows'])):
        first_col = True
        best_vals = (direction * single_kind.loc[row_members, :]).max(axis=1)
        for ax, col_members in zip(row_axs, group_members['cols']):
            subset = single_kind.loc[row_members, col_members]

            # Plot heat map
            sns.heatmap(subset, cbar=cbar, cbar_ax=cbar_ax, norm=norm, cmap=cmap,
                        vmin=0, vmax=100, annot=True, fmt='.0f', ax=ax)

            # Red border around maximal entries
            if direction != 0:
                best = (direction * subset.T >= best_vals).T

                subplot_rows, subplot_cols = best.shape
                for m in range(subplot_rows):
                    for n in range(subplot_cols):
                        if best.iloc[m, n]:
                            rectangle = Rectangle((m, n), 1, 1, fill=False, edgecolor='red', lw=1)
                            ax.add_patch(rectangle)

            ax.get_yaxis().set_visible(yaxis and first_col)
            ax.get_xaxis().set_visible(i == nrows - 1)

            rotate_labels(ax)
            first_col = False
            cbar = False

    if xlabel:
        midpoint = (axs[-1, 0].get_position().x0 + axs[-1, -1].get_position().x1) / 2
        fig.text(midpoint, 0.01, 'Opponent', ha='center')
    if ylabel:
        fig.text(0.01, 0.5 * (gridspec_kw['left'] + 1), 'Victim', va='center', rotation='vertical')

    return axs


def heatmap_one_col(single_env, col, cbar, xlabel, ylabel, cmap='Blues'):
    width, height = plt.rcParams['figure.figsize']
    if xlabel:
        height += 0.17
    fig = plt.figure(figsize=(width, height))

    cbar_width = 0.15 if cbar else 0.0
    gridspec_kw = {
        'left': 0.2,
        'right': 0.98 - cbar_width,
        'bottom': 0.28,
        'top': 0.95,
        'wspace': 0.05,
        'hspace': 0.05,
    }
    single_env *= 100 / num_episodes(single_env)  # convert to percentages

    _pretty_heatmap(single_env, col, cmap, fig, gridspec_kw,
                    xlabel=xlabel, ylabel=ylabel, cbar_width=cbar_width)
    return fig


def heatmap_full(single_env, cmap='Blues', cols=None):
    # Figure layout calculations
    if cols is None:
        cols = single_env.columns

    cbar_width_in = 0.41
    reserved = {  # in inches
        'left': 0.605,  # for y-axis
        'top': 0.28,  # for title
        'bottom': 0.49,  # for x-axis
        'right': cbar_width_in + 0.02,  # for color bar plus margin
    }

    reserved_height = reserved['top'] + reserved['bottom']
    width, nominal_height = plt.rcParams.get('figure.figsize')
    portion_for_heatmap = nominal_height - reserved_height
    # We want height to vary depending on number of labels. Take figure size as specifying the
    # height for a 'typical' figure with 6 rows.
    height_per_row = nominal_height * portion_for_heatmap / 6
    num_rows = len(pd.unique(single_env.index.get_level_values(0)))
    height_for_heatmap = height_per_row * max(num_rows, 4)

    height = reserved_height + height_for_heatmap
    gridspec_kw = {
        'top': 1 - reserved['top'] / height,
        'bottom': reserved['bottom'] / height,
        'wspace': 0.05,
        'hspace': 0.05,
    }

    # Actually plot the heatmap
    subplot_wspace = 0.05 / width
    left = reserved['left'] / width
    max_right = 1 - reserved['right'] / width
    per_plot_width = (max_right - left) / len(cols)

    single_env *= 100 / num_episodes(single_env)  # convert to percentages
    fig = plt.figure(figsize=(width, height))
    for i, col in enumerate(cols):
        right = left + per_plot_width - subplot_wspace
        gridspec_kw.update({'left': left, 'right': right})

        cbar = i == len(cols) - 1
        subplot_cbar_width = cbar_width_in / width if cbar else 0.0

        _pretty_heatmap(single_env, col, cmap, fig, gridspec_kw,
                        cbar_width=subplot_cbar_width, yaxis=i == 0)

        if len(cols) > 1:
            mid_x = (left + right - subplot_cbar_width) / 2
            mid_y = (1 + gridspec_kw['top']) / 2
            plt.figtext(mid_x, mid_y, col, va="center", ha="center",
                        size=plt.rcParams.get('axes.titlesize'))

        left += per_plot_width

    return fig
