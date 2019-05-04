import glob
import logging
import os
import os.path as osp
import tempfile

import matplotlib
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
import sacred
from sacred.observers import FileStorageObserver

from modelfree.visualize.styles import STYLES

vis_tsne_ex = sacred.Experiment('visualize_tsne')
logger = logging.getLogger('modelfree.interpretation.visualize_tsne')


@vis_tsne_ex.config
def main_config():
    model_glob = None
    output_root = None
    subsample_rate = 0.15
    opacity = 0.75
    dot_size = 0.25
    palette_name = 'cube_bright'
    save_type = 'pdf'
    styles = ['paper', 'threecol']
    ordering = ['Adv', 'Zoo', 'Rand']

    external_legend_params = {
        'legend_styles': ['paper'],
        'legend_height': 0.3,
    }
    _ = locals()
    del _


ABBREVIATIONS = {
    'ppo2': 'Adv',
    'zoo': 'Zoo',
    'random': 'Rand',
}


PALETTES = {
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


@vis_tsne_ex.capture
def _make_handles(palette_name, ordering):
    palette = PALETTES[palette_name]
    handles, labels = [], []
    for label in ordering:
        color = palette[label]
        handle = matplotlib.lines.Line2D(range(1), range(1), color=color,
                                         marker='o', markerfacecolor=color, linewidth=0)
        handles.append(handle)
        labels.append(label)
    return handles, labels


@vis_tsne_ex.capture
def _plot_and_save_chart(save_path, datasets, opacity, dot_size, palette_name,
                         styles, external_legend_params):
    legend = external_legend_params is None
    with plt.style.context([STYLES[style] for style in styles]):
        plt.rc('axes.spines', **{'bottom': False, 'left': False, 'right': False, 'top': False})

        if palette_name is None:
            palette_name = 'bright'
        palette = PALETTES[palette_name]

        # Make a tight figure (deleting axes)
        width, height = plt.rcParams['figure.figsize']
        ncols = len(datasets)
        width = width * ncols
        fig, axs = plt.subplots(figsize=(width, height), nrows=1, ncols=ncols, squeeze=False,
                                sharex=True, sharey=True, gridspec_kw={'wspace': 0.0})

        # Color-coded scatter-plot
        for data, ax in zip(datasets, axs[0]):
            ax.axis('off')
            ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
            ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
            ax.margins(x=0.01, y=0.01)

            hues = data['opponent_id'].apply(palette.get)
            ax.scatter(data['ax_1'], data['ax_2'], c=hues, alpha=opacity, s=dot_size,
                       edgecolors='none', linewidth=0)

            if legend:
                handles, labels = _make_handles()
                ax.legend(handles=handles, labels=labels,
                          loc=9, ncol=len(handles), bbox_to_anchor=(0.48, 1.18))

        fig.savefig(save_path, dpi=800, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

    vis_tsne_ex.add_artifact(save_path)


@vis_tsne_ex.capture(prefix='external_legend_params')
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


@vis_tsne_ex.capture
def _visualize_tsne_helper(model_dir, output_dir, subsample_rate, save_type,
                           ordering, external_legend_params):
    # Data
    metadata_df = pd.read_csv(os.path.join(model_dir, 'metadata.csv'))
    cluster_ids = np.load(os.path.join(model_dir, 'cluster_ids.npy'))
    metadata_df['ax_1'] = cluster_ids[:, 0]
    metadata_df['ax_2'] = cluster_ids[:, 1]
    metadata_df['opponent_id'] = metadata_df['opponent_id'].apply(ABBREVIATIONS.get)

    def save_path(prefix):
        return osp.join(output_dir, f'{prefix}.{save_type}')

    _plot_and_save_chart(save_path('combined'), [metadata_df])
    _plot_and_save_chart(save_path('subsampled'), [metadata_df.sample(frac=subsample_rate)])

    opponent_groups = metadata_df.groupby('opponent_id')
    opponent_dfs = {name: group for name, group in opponent_groups}
    opponent_dfs = [opponent_dfs[label] for label in ordering]
    _plot_and_save_chart(save_path('sidebyside'), opponent_dfs)

    if external_legend_params is not None:
        _external_legend(osp.join(output_dir, 'external_legend.pdf'))


@vis_tsne_ex.main
def visualize_tsne(model_glob, output_root):
    # Output directory
    tmp_dir = None
    if output_root is None:
        tmp_dir = tempfile.TemporaryDirectory()
        output_root = tmp_dir.name

    for model_dir in glob.glob(model_glob):
        model_name = os.path.basename(model_dir)
        output_dir = osp.join(output_root, model_name)
        os.makedirs(output_dir)
        _visualize_tsne_helper(model_dir, output_dir)

    if tmp_dir is not None:
        tmp_dir.cleanup()


def main():
    observer = FileStorageObserver.create(osp.join('data', 'sacred', 'visualize_tsne'))
    vis_tsne_ex.observers.append(observer)
    vis_tsne_ex.run_commandline()


if __name__ == '__main__':
    main()
