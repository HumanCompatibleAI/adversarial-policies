import json
import logging
import os
import os.path as osp
import tempfile

import altair as alt
import cv2
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

vis_tsne_ex = sacred.Experiment('vis_tsne')
logger_obj = logging.getLogger('modelfree.interpretation.visualize_tsne')

abbreviation_lookup = {
    "adversary": "Adv",
    "zoo": "Zoo1",
    "random": "Rand"
}


@vis_tsne_ex.config
def main_config():
    base_path = None
    sacred_id = None
    subsample_rate = 0.15
    perplexity = 250
    video_path = None
    chart_type = "seaborn"
    opacity = 1.0
    dot_size = 0.25
    palette_name = "cube_bright"
    save_type = "pdf"
    hue_order = ["Adv", "Zoo1", "Rand"]
    _ = locals()
    del _


def _get_latest_sacred_dir_with_params(base_path, param_dict=None):
    sacred_dirs = os.listdir(base_path)
    max_int_dir = -1
    for sd in sacred_dirs:
        if param_dict is not None:
            try:
                with open(os.path.join(base_path, sd, "config.json")) as fp:
                    config_params = json.load(fp)
            except (NotADirectoryError, FileNotFoundError):
                logger_obj.info("No config json found at {}".format(sd))
                continue
            all_match = True
            for param in param_dict:
                if param not in config_params or config_params[param] != param_dict[param]:
                    all_match = False
                    break
            if not all_match:
                continue
        try:
            int_dir = int(sd)
            if int_dir > max_int_dir:
                max_int_dir = int_dir
        except ValueError:
            continue
    if max_int_dir < 0:
        format_str = "No sacred directory found for base path {}, param dict {}"
        raise ValueError(format_str.format(base_path, param_dict))
    return str(max_int_dir)


@vis_tsne_ex.capture
def _plot_and_save_chart(data, fname, chart_type, opacity, dot_size, palette_name,
                         hue_order):
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, fname)

        if chart_type == 'altair':
            chart = alt.Chart(data).mark_point().configure_mark(
                opacity=opacity).encode(
                x='ax_1', y='ax_2', color='opponent_id')
            chart.save(fname)
        elif chart_type == 'seaborn':
            if palette_name == "bright" or palette_name is None:
                palette_name = {
                    "Zoo1": "#66c2a5",
                    "Rand": "#fdb462",
                    "Adv": '#e7298a'
                }
            if palette_name == "cube_bright":
                palette_name = {
                    "Zoo1": "#b87903",
                    "Rand": "#d2b5ff",
                    "Adv": '#016b61'
                }
            fig, ax = plt.subplots(figsize=(2.75, 2.0625))
            sns.scatterplot(data=data, x="ax_1", y="ax_2", hue="opponent_id_remapped",
                            alpha=opacity, s=dot_size, edgecolors='none', linewidth=0,
                            palette=palette_name, hue_order=hue_order, ax=ax)
            handles, labels = ax.get_legend_handles_labels()
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
            plt.style.use(STYLES['paper'])
            ax.legend(handles=handles[1:], labels=labels[1:],
                      loc=9, ncol=3, bbox_to_anchor=(0.48, 1.18))
            plt.rc('axes.spines', **{'bottom': False, 'left': False, 'right': False, 'top': False})
            plt.savefig(fname, dpi=800)
            plt.close()
        vis_tsne_ex.add_artifact(fname)


def sample_range(df, xrange, yrange):
    xrange_satisfies = df[(df['ax_1'] > xrange[0]) & (df['ax_1'] < xrange[1])]
    both_satisfies = xrange_satisfies[(xrange_satisfies['ax_2'] > yrange[0]) & (
        xrange_satisfies['ax_2'] < yrange[1])]
    return both_satisfies.sample(1).iloc[0]


@vis_tsne_ex.capture
def get_frame_shot(opponent, episode, observation, video_path):
    latest_dir = _get_latest_sacred_dir_with_params(os.path.join(video_path, opponent))
    fname = "{}_episode_{}_frame_{}.jpg".format(opponent, episode, observation)
    frame_path = os.path.join(video_path, latest_dir, str(episode), fname)
    img = cv2.imread(frame_path)
    cv2.imshow('image', img)
    cv2.waitKey(3)
    cv2.destroyAllWindows()


def sample_points_from_ranges(data):
    ranges = [
        {'xrange': (-5, 10), 'yrange': (5, 15)},
        {'xrange': (-25, -15), 'yrange': (-20, 5)}
    ]
    num_samples = 4
    for range_obj in ranges:
        print(range_obj)
        for _ in range(num_samples):
            sample_series = sample_range(data, **range_obj)
            print(sample_series)


@vis_tsne_ex.main
def visualize_tsne(base_path, sacred_id, subsample_rate, perplexity, save_type):

    if sacred_id is None:
        sacred_id = _get_latest_sacred_dir_with_params(base_path, {'perplexity': perplexity})

    logger_obj.info("For perplexity {}, using sacred dir {}".format(perplexity, sacred_id))

    if base_path is None:
        full_sacred_dir = sacred_id
    else:
        full_sacred_dir = os.path.join(base_path, sacred_id)

    metadata_df = pd.read_csv(os.path.join(full_sacred_dir, 'metadata.csv'))
    cluster_ids = np.load(os.path.join(full_sacred_dir, 'cluster_ids.npy'))
    metadata_df['ax_1'] = cluster_ids[:, 0]
    metadata_df['ax_2'] = cluster_ids[:, 1]
    remapped = metadata_df['opponent_id'].apply(lambda x: abbreviation_lookup.get(x, None))
    metadata_df['opponent_id_remapped'] = remapped

    _plot_and_save_chart(metadata_df.query("opponent_id != 'random'"),
                         "no_random_chart.{}".format(save_type))
    _plot_and_save_chart(metadata_df.query("opponent_id != 'adversary'"),
                         "no_adversary_chart.{}".format(save_type))
    _plot_and_save_chart(metadata_df, "opponent_chart.{}".format(save_type))
    _plot_and_save_chart(metadata_df.sample(frac=subsample_rate),
                         fname="subsample_chart.{}".format(save_type))
    opponent_groups = metadata_df.groupby('opponent_id')

    for name, group in opponent_groups:
        _plot_and_save_chart(group, "{}_chart.{}".format(name, save_type))


def main():
    observer = FileStorageObserver.create(osp.join('data', 'sacred', 'vis_tsne'))
    vis_tsne_ex.observers.append(observer)
    vis_tsne_ex.run_commandline()


if __name__ == '__main__':
    main()
