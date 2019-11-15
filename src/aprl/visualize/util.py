import functools
import json
import logging
import os.path
import re

import matplotlib.backends.backend_pdf
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from aprl import train
from aprl.configs import DATA_LOCATION
from aprl.envs import VICTIM_INDEX, gym_compete
from aprl.visualize import styles

logger = logging.getLogger('aprl.visualize.util')

# Data loading & manipulation


def load_json(path: str):
    with open(path, 'r') as f:
        return json.load(f)


def load_scores(path: str) -> pd.DataFrame:
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
                      'opponent_type', 'opponent_path']
    cols = ['Opponent Win', 'Victim Win', 'Ties']
    return df.loc[:, cols].copy()


def abbreviate_agent_config(env_name: str, agent_type: str, agent_path: str,
                            suffix: str, victim: bool) -> str:
    """Convert an agent configuration into a short abbreviation."""
    if agent_type == 'zoo':
        prefix = f'Zoo{suffix}'
        if not gym_compete.is_symmetric(env_name):
            prefix += 'V' if victim else 'O'
        assert isinstance(agent_path, str) and len(agent_path) == 1 and agent_path.isnumeric()
        return f'{prefix}{agent_path}'
    elif agent_type == 'zero':
        return f'Zero{suffix}'
    elif agent_type == 'random':
        return f'Rand{suffix}'
    elif agent_type == 'ppo2':
        components = agent_path.split(os.path.sep)
        components = components[:-3]  # chop off baselines/*/final_model
        components = components[components.index('multi_train'):]  # make relative
        sacred_path = os.path.join(DATA_LOCATION, *components,
                                   'sacred', 'train', '1', 'config.json')

        with open(sacred_path, 'r') as f:
            cfg = json.load(f)

        load_policy = cfg['load_policy']
        finetuned = load_policy['path'] is not None
        if finetuned:
            orig = abbreviate_agent_config(env_name, load_policy['type'], load_policy['path'],
                                           suffix='', victim=victim)

        try:
            embed_types, embed_paths, _ = train.resolve_embed(
                cfg['embed_type'], cfg['embed_path'], cfg['embed_types'], cfg['embed_paths'], {})
        except KeyError:
            # TODO(adam): this is for backward compatibility, remove after retraining old policies
            embed_types = [cfg['victim_type']]
            embed_paths = [str(cfg['victim_path'])]

        if victim:
            assert finetuned
            assert load_policy['type'] == 'zoo'

            single = set(embed_types) == {'ppo2'}
            dual = set(embed_types) == {'ppo2', 'zoo'}
            assert single or dual
            defense_type = 'S' if single else 'D'
            return orig.replace('Zoo', f'Zoo{suffix}{defense_type}')
        else:  # opponent
            assert len(embed_types) == 1
            victim_abbv = abbreviate_agent_config(env_name, embed_types[0], embed_paths[0],
                                                  suffix='', victim=True)
            victim_abbv = re.sub(r'Zoo([SD]?)[OV]?(.*)', r'\1\2', victim_abbv)
            prefix = 'F' if finetuned else ''
            return f'{prefix}Adv{suffix}{victim_abbv}'
    else:
        raise ValueError(f"Unknown agent_type '{agent_type}'")


def victim_abbrev(x, suffix) -> str:
    env_name, victim_type, victim_path, _, _ = x
    return abbreviate_agent_config(env_name, victim_type, victim_path,
                                   suffix=suffix, victim=True)


def opponent_abbrev(x, suffix) -> str:
    env_name, _, _, opponent_type, opponent_path = x
    return abbreviate_agent_config(env_name, opponent_type, opponent_path,
                                   suffix=suffix, victim=False)


# Longer description: for website
FRIENDLY_AGENT_LABEL_LONG = {
    "Rand": "Random",
    "Zero": "Lifeless",
    r"Zoo[VO]?[0-9]": "Normal",
    r"ZooM[VO]?[0-9]": "Masked",
    r"ZooMS[VO]?[0-9]": "Masked Single Fine-tuned",
    r"ZooMD[VO]?[0-9]": "Masked Dual Fine-tuned",
    r"ZooS[VO]?[0-9]": "Single Fine-tuned",
    r"ZooD[VO]?[0-9]": "Dual Fine-tuned",
    r"Adv([0-9])": "Adversary",
    r"Adv[SD]([0-9])": "Retrained Adversary",
}

# Shorter description: for videos
FRIENDLY_AGENT_LABEL_SHORT = {
    "Rand": "Random",
    "Zero": "Lifeless",
    r"Zoo[VO]?[0-9]": "Normal",
    r"ZooM[SD]?[VO]?[0-9]": "Masked",
    r"Zoo[SD][VO]?[0-9]": "Fine-tuned",
    r"Adv[SD]?([0-9])": "Adversary",
}


def friendly_agent_label(abbrev: str, short: bool = False) -> str:
    labels = FRIENDLY_AGENT_LABEL_SHORT if short else FRIENDLY_AGENT_LABEL_LONG
    matches = {pattern: label for pattern, label in labels.items()
               if re.match(pattern, abbrev)}
    if len(matches) == 0:
        raise ValueError(f"No friendly label for '{abbrev}'")
    if len(matches) > 1:
        raise ValueError(f"Ambiguous friendly label for '{abbrev}'")
    return list(matches.values())[0]


def load_datasets(path: str, victim_suffix: str = '', opponent_suffix: str = '') -> pd.DataFrame:
    """Loads scores from path, using `abbreviate_agent_config` to pretty-print agents.

    :param path: Path to a JSON file to load using `load_scores`.
    :param victim_suffix: A suffix for victim agents.
    :param opponent_suffix: A suffix for opponent agents.
    :return A DataFrame containing columns "Opponent Win", "Victim Win" and "Ties" with MultiIndex
        with levels "env_name", "victim_type", "victim_path", "opponent_type" and "opponent_path".
    """
    scores = load_scores(path)

    assert scores.index.is_unique
    idx = scores.index.to_frame()
    victims = idx.apply(functools.partial(victim_abbrev, suffix=victim_suffix),
                        axis='columns')
    opponents = idx.apply(functools.partial(opponent_abbrev, suffix=opponent_suffix),
                          axis='columns')
    idx['victim_abbrev'] = victims
    idx['opponent_abbrev'] = opponents
    idx = idx.drop(columns=['victim_type', 'victim_path', 'opponent_type', 'opponent_path'])
    idx = pd.MultiIndex.from_frame(idx)
    assert idx.is_unique, "two distinct agents mapped to same abbreviation"
    scores.index = idx

    return scores

# Visualization


def apply_per_env(scores, fn, *args, suptitle=True, **kwargs):
    envs = scores.index.levels[0]
    for i, env_name in enumerate(envs):
        single_env = pd.DataFrame(scores.loc[env_name])
        single_env.name = env_name
        fig = fn(single_env, *args, **kwargs)

        if suptitle:
            pretty_env = styles.PRETTY_ENV.get(env_name, env_name)
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
    'rows': [r'^ZooV?[0-9]', r'^ZooSV?[0-9]', r'^ZooDV?[0-9]', r'^ZooMV?[0-9]'],
    'cols': [r'^Adv[0-9]', r'^F?AdvS[0-9]', r'^F?AdvD[0-9]', r'^ZooO?[0-9]', r'^Zero|^Rand']
}


def _split_groups(df):
    group_members = {}
    num_matches = {}
    index = df.index.remove_unused_levels()
    for kind, groups in GROUPS.items():
        level = 1 if kind == 'cols' else 0
        level_values = index.levels[level]
        masks = [level_values.str.contains(pattern) for pattern in groups]
        group_members[kind] = [level_values[mask] for mask in masks if np.any(mask)]
        num_matches[kind] = [mask.sum() for mask in masks if np.any(mask)]
    return group_members, num_matches


class DogmaticNormalize(matplotlib.colors.Normalize):  # pytype:disable=module-attr
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
    single_env = pd.DataFrame(single_env)
    # Blank out names so Seaborn doesn't plot them
    single_env.index.names = [None for _ in single_env.index.names]

    group_members, num_matches = _split_groups(single_env)
    single_kind = single_env[col].unstack()
    direction = DIRECTIONS[col]

    gridspec_kw = dict(gridspec_kw)
    gridspec_kw.update({
        'width_ratios': num_matches['cols'],
        'height_ratios': num_matches['rows'],
        'bottom': gridspec_kw['bottom'] + (0.03 if xlabel else 0.01),
        'left': gridspec_kw['left'] + (0.05 if ylabel else 0.0),
    })
    nrows = len(num_matches['rows'])
    ncols = len(num_matches['cols'])
    axs = fig.subplots(nrows=nrows, ncols=ncols, gridspec_kw=gridspec_kw)

    cbar = cbar_width > 0
    cbar_ax = None
    if cbar > 0:
        gs = matplotlib.gridspec.GridSpec(1, 1)  # pytype:disable=module-attr
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
