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
    # TODO: cross-platform
    model_dir = 'data/tsne/debug/20190502_203906/fitted/SumoAnts-v0_victim_zoo_1'
    output_dir = None
    subsample_rate = 0.15
    video_path = None
    chart_type = 'seaborn'
    opacity = 1.0
    dot_size = 0.25
    palette_name = 'cube_bright'
    save_type = 'pdf'
    hue_order = ['Adv', 'Rand', 'Zoo']
    styles = ['paper', 'twocol']
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
def _plot_and_save_chart(save_path, data, chart_type, opacity, dot_size, palette_name,
                         hue_order):
    if chart_type == 'altair':
        chart = alt.Chart(data).mark_point().configure_mark(
            opacity=opacity).encode(
            x='ax_1', y='ax_2', color='opponent_id')
        chart.save(save_path)
    elif chart_type == 'seaborn':
        if palette_name is None:
            palette_name = 'bright'
        palette = PALETTES[palette_name]
        fig, ax = plt.subplots()
        sns.scatterplot(data=data, x="ax_1", y="ax_2", hue="opponent_id",
                        alpha=opacity, s=dot_size, edgecolors='none', linewidth=0,
                        palette=palette, hue_order=hue_order, ax=ax)
        handles, labels = ax.get_legend_handles_labels()
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.legend(handles=handles[1:], labels=labels[1:],
                  loc=9, ncol=3, bbox_to_anchor=(0.48, 1.18))
        plt.savefig(save_path, dpi=800)
        plt.close()
    vis_tsne_ex.add_artifact(save_path)


@vis_tsne_ex.main
def visualize_tsne(model_dir, output_dir, styles, subsample_rate, save_type):
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

    # Plotting
    for style in styles:
        plt.style.use(STYLES[style])
    plt.rc('axes.spines', **{'bottom': False, 'left': False, 'right': False, 'top': False})

    _plot_and_save_chart(osp.join(output_dir, "no_random_chart.{}".format(save_type)),
                         metadata_df.query("opponent_id != 'Rand'"))
    _plot_and_save_chart(osp.join(output_dir, "no_adversary_chart.{}".format(save_type)),
                         metadata_df.query("opponent_id != 'Adv'"),)
    _plot_and_save_chart(osp.join(output_dir, "opponent_chart.{}".format(save_type)),
                         metadata_df)
    _plot_and_save_chart(osp.join(output_dir, "subsample_chart.{}".format(save_type)),
                         metadata_df.sample(frac=subsample_rate))
    opponent_groups = metadata_df.groupby('opponent_id')

    for name, group in opponent_groups:
        _plot_and_save_chart(osp.join(output_dir, "{}_chart.{}".format(name, save_type)), group)

    if tmp_dir is not None:
        tmp_dir.cleanup()


def main():
    observer = FileStorageObserver.create(osp.join('data', 'sacred', 'visualize_tsne'))
    vis_tsne_ex.observers.append(observer)
    vis_tsne_ex.run_commandline()


if __name__ == '__main__':
    main()
