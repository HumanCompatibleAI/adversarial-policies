import logging
import os
import os.path as osp
import tempfile

import altair as alt
import matplotlib
import numpy as np
import pandas as pd
import sacred
from sacred.observers import FileStorageObserver

from modelfree.visualize.styles import STYLES

# WARNING: isort has been disabled on this file to allow this.
# Needed as matplotlib.use has to run before pyplot is imported.
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

vis_tsne_ex = sacred.Experiment('visualize_tsne')
logger = logging.getLogger('modelfree.interpretation.visualize_tsne')


@vis_tsne_ex.config
def main_config():
    model_dir = None
    output_dir = None
    subsample_rate = 0.15
    opacity = 0.75
    dot_size = 0.25
    palette_name = 'cube_bright'
    save_type = 'pdf'
    hue_order = ['Adv', 'Rand', 'Zoo']
    styles = ['paper', 'threecol']

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
        'Zoo': '#66c2a5',
        'Rand': '#fdb462',
        'Adv': '#e7298a',
    },
    'cube_bright': {
        'Zoo': '#b87903',
        'Rand': '#d2b5ff',
        'Adv': '#016b61',
    }
}


@vis_tsne_ex.capture
def _plot_and_save_chart(save_path, data, opacity, dot_size, palette_name,
                         hue_order, styles, external_legend_params):
    legend = external_legend_params is None
    with plt.style.context([STYLES[style] for style in styles]):
        plt.rc('axes.spines', **{'bottom': False, 'left': False, 'right': False, 'top': False})

        if palette_name is None:
            palette_name = 'bright'
        palette = PALETTES[palette_name]

        fig, ax = plt.subplots()
        sns.scatterplot(data=data, x="ax_1", y="ax_2", hue="opponent_id",
                        alpha=opacity, s=dot_size, edgecolors='none', linewidth=0,
                        palette=palette, hue_order=hue_order, ax=ax)

        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

        handles, labels = ax.get_legend_handles_labels()
        handles, labels = handles[1:], labels[1:]
        if legend:
            ax.legend(handles=handles, labels=labels,
                      loc=9, ncol=len(handles), bbox_to_anchor=(0.48, 1.18))
        else:
            ax.get_legend().remove()  # still need seaborn to plot legend to get handles, labels


        fig.savefig(save_path, dpi=800)
        plt.close(fig)

    vis_tsne_ex.add_artifact(save_path)

    return handles, labels


@vis_tsne_ex.capture(prefix='external_legend_params')
def _external_legend(save_path, handles, labels, legend_styles, legend_height):
    with plt.style.context([STYLES[style] for style in legend_styles]):
        width, height = plt.rcParams['figure.figsize']
        height = legend_height
        legend_fig = plt.figure(figsize=(width, height))

        legend_fig.legend(handles=handles, labels=labels, loc='lower left', mode='expand',
                          ncol=len(handles), bbox_to_anchor=(0.0, 0.0, 1.0, 1.0))
        legend_fig.savefig(save_path)
        plt.close(legend_fig)


@vis_tsne_ex.main
def visualize_tsne(model_dir, output_dir, subsample_rate, save_type, external_legend_params):
    # Data
    metadata_df = pd.read_csv(os.path.join(model_dir, 'metadata.csv'))
    cluster_ids = np.load(os.path.join(model_dir, 'cluster_ids.npy'))
    metadata_df['ax_1'] = cluster_ids[:, 0]
    metadata_df['ax_2'] = cluster_ids[:, 1]
    metadata_df['opponent_id'] = metadata_df['opponent_id'].apply(ABBREVIATIONS.get)

    # Output directory
    tmp_dir = None
    if output_dir is None:
        tmp_dir = tempfile.TemporaryDirectory()
        output_dir = tmp_dir.name
    else:
        os.makedirs(output_dir)

    def save_path(prefix):
        return osp.join(output_dir, f'{prefix}.{save_type}')

    handles, labels = _plot_and_save_chart(save_path('opponent_chart'), metadata_df)
    if external_legend_params is not None:
        _external_legend(osp.join(output_dir, 'external_legend.pdf'), handles, labels)

    _plot_and_save_chart(save_path('no_random_chart'), metadata_df.query("opponent_id != 'Rand'"))
    _plot_and_save_chart(save_path('no_adversary_chart'),
                         metadata_df.query("opponent_id != 'Adv'"))
    _plot_and_save_chart(save_path('subsample_chart'), metadata_df.sample(frac=subsample_rate))

    opponent_groups = metadata_df.groupby('opponent_id')
    for name, group in opponent_groups:
        _plot_and_save_chart(save_path(f'{name}_chart'), group)

    if tmp_dir is not None:
        tmp_dir.cleanup()


def main():
    observer = FileStorageObserver.create(osp.join('data', 'sacred', 'visualize_tsne'))
    vis_tsne_ex.observers.append(observer)
    vis_tsne_ex.run_commandline()


if __name__ == '__main__':
    main()
