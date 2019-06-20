import glob
import logging
import os
import os.path as osp
import tempfile

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sacred
from sacred.observers import FileStorageObserver

from modelfree.common import utils
from modelfree.visualize.styles import PRETTY_LABELS, STYLES

visualize_ex = sacred.Experiment('tsne_visualize')
logger = logging.getLogger('modelfree.tsne.visualize')


@visualize_ex.config
def main_config():
    model_glob = None                  # path name with wildcards specifying model directories
    output_root = None                 # directory to save figures to
    subsample_rate = 0.15
    opacity = 0.75
    dot_size = 0.25
    palette_name = 'paper'
    save_type = 'pdf'
    styles = ['paper', 'threecol']
    ordering = ['Adv', 'Zoo', 'Rand']
    pretty_labels = PRETTY_LABELS

    internal_legend = False
    external_legend_params = {
        'legend_styles': ['paper'],
        'legend_height': 0.3,
    }
    _ = locals()
    del _


@visualize_ex.named_config
def inline_config():
    styles = ['paper', 'twocol']
    internal_legend = True
    external_legend_params = None
    pretty_labels = {
        'Adv': 'Adv',
        'Zoo': 'Zoo',
        'Rand': 'Rand',
    }

    model_glob = 'data/tsne/default/20190505_193250/fitted/*'
    output_root = 'data/tsne/default/20190505_193250/figures_inline/'

    _ = locals()
    del _


ABBREVIATIONS = {
    'ppo2': 'Adv',
    'zoo': 'Zoo',
    'random': 'Rand',
}


PALETTES = {
    'paper': {
        'Adv': '#1f77b4',
        'Zoo': '#ff7f0e',
        'Rand': '#2ca02c',
    },
    'bright': {
        'Adv': '#e7298a',
        'Zoo': '#66c2a5',
        'Rand': '#fdb462',
    },
    'cube_bright': {
        'Adv': '#016b61',
        'Zoo': '#b87903',
        'Rand': '#d2b5ff',
    },
}


@visualize_ex.capture
def _make_handles(palette_name, ordering, pretty_labels):
    palette = PALETTES[palette_name]
    handles, labels = [], []
    for key in ordering:
        color = palette[key]
        handle = matplotlib.lines.Line2D(range(1), range(1), color=color,
                                         marker='o', markerfacecolor=color, linewidth=0)
        handles.append(handle)
        labels.append(pretty_labels[key])
    return handles, labels


@visualize_ex.capture
def _plot_and_save_chart(save_path, datasets, opacity, dot_size, palette_name,
                         styles, internal_legend, external_legend_params):
    assert not internal_legend or external_legend_params is None
    with plt.style.context([STYLES[style] for style in styles]):
        plt.rc('axes.spines', **{'bottom': False, 'left': False, 'right': False, 'top': False})

        if palette_name is None:
            palette_name = 'bright'
        palette = PALETTES[palette_name]

        # Make a tight figure (deleting axes)
        width, height = plt.rcParams['figure.figsize']
        ncols = len(datasets)
        width = width * ncols

        gridspec_kw = {'wspace': 0.0, 'top': 0.85 if internal_legend else 1.0}
        fig, axs = plt.subplots(figsize=(width, height), nrows=1, ncols=ncols, squeeze=False,
                                sharex=True, sharey=True, gridspec_kw=gridspec_kw)

        # Color-coded scatter-plot
        for data, ax in zip(datasets, axs[0]):
            ax.axis('off')
            ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
            ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
            ax.margins(x=0.01, y=0.01)

            hues = data['opponent_id'].apply(palette.get)
            ax.scatter(data['ax_1'], data['ax_2'], c=hues, alpha=opacity, s=dot_size,
                       edgecolors='none', linewidth=0)

            if internal_legend is not None:
                handles, labels = _make_handles()
                fig.legend(handles=handles, labels=labels, ncol=len(handles),
                           loc='lower center', bbox_to_anchor=(0.05, 0.88, 0.9, 0.05),
                           mode='expand', borderaxespad=0, frameon=True)

        kwargs = {} if internal_legend else {'bbox_inches': 'tight', 'pad_inches': 0.0}
        fig.savefig(save_path, dpi=800, **kwargs)
        plt.close(fig)


@visualize_ex.capture(prefix='external_legend_params')
def _external_legend(save_path, legend_styles, legend_height):
    with plt.style.context([STYLES[style] for style in legend_styles]):
        width, height = plt.rcParams['figure.figsize']
        height = legend_height
        legend_fig = plt.figure(figsize=(width, height))

        handles, labels = _make_handles()
        legend_fig.legend(handles=handles, labels=labels, loc='lower left', mode='expand',
                          ncol=len(handles), bbox_to_anchor=(0.0, 0.0, 1.0, 1.0))
        legend_fig.savefig(save_path)
        plt.close(legend_fig)


@visualize_ex.capture
def _visualize_helper(model_dir, output_dir, subsample_rate, save_type,
                      ordering, external_legend_params):
    logger.info("Generating figures")

    # Data
    metadata_df = pd.read_csv(os.path.join(model_dir, 'metadata.csv'))
    cluster_ids = np.load(os.path.join(model_dir, 'cluster_ids.npy'))
    metadata_df['ax_1'] = cluster_ids[:, 0]
    metadata_df['ax_2'] = cluster_ids[:, 1]
    metadata_df['opponent_id'] = metadata_df['opponent_id'].apply(ABBREVIATIONS.get)

    def save_path(prefix):
        return osp.join(output_dir, f'{prefix}.{save_type}')

    counts = pd.value_counts(metadata_df['opponent_id'])
    min_counts = counts.min()
    opponent_groups = metadata_df.groupby('opponent_id')
    opponent_dfs = {name: group.sample(n=min_counts) for name, group in opponent_groups}
    opponent_dfs = [opponent_dfs[label] for label in ordering]
    metadata_df = pd.concat(opponent_dfs)

    _plot_and_save_chart(save_path('combined'), [metadata_df])
    _plot_and_save_chart(save_path('subsampled'), [metadata_df.sample(frac=subsample_rate)])
    _plot_and_save_chart(save_path('sidebyside'), opponent_dfs)

    if external_legend_params is not None:
        _external_legend(osp.join(output_dir, 'external_legend.pdf'))

    logger.info("Visualization complete")


@visualize_ex.main
def visualize(_run, model_glob, output_root):
    # Output directory
    tmp_dir = None
    if output_root is None:
        tmp_dir = tempfile.TemporaryDirectory()
        output_root = tmp_dir.name

    for model_dir in glob.glob(model_glob):
        model_name = os.path.basename(model_dir)
        output_dir = osp.join(output_root, model_name)
        os.makedirs(output_dir)
        _visualize_helper(model_dir, output_dir)

    utils.add_artifacts(_run, output_root, ingredient=visualize_ex)
    if tmp_dir is not None:
        tmp_dir.cleanup()


def main():
    observer = FileStorageObserver.create(osp.join('data', 'sacred', 'tsne_visualize'))
    visualize_ex.observers.append(observer)
    visualize_ex.run_commandline()


if __name__ == '__main__':
    main()
