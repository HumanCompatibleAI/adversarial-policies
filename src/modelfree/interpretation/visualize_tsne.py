import matplotlib
matplotlib.use('Agg')
## Breaking linting here so that this runs in a Virtualenv; this command needs to run prior to
## Pyplot or Seaborn being imported

import json
import logging
import os
import tempfile

import altair as alt
import cv2

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sacred
from sacred.observers import FileStorageObserver
import seaborn as sns
from swissarmy import logger



tsne_vis_ex = sacred.Experiment('tsne-visualization')
tsne_vis_ex.observers.append(FileStorageObserver.create(
    '/Users/cody/Data/adversarial_policies/tsne_visualization'))
logger_obj = logger.get_logger_object(cl_level=logging.DEBUG)


@tsne_vis_ex.config
def main_config():
    base_path = "/Users/cody/Data/adversarial_policies/tsne_runs"
    sacred_id = None
    subsample_rate = 0.15
    perplexity = 250
    video_path = "/Users/cody/Data/adversarial_policies/video_frames"
    chart_type = "altair"
    opacity = 0.10
    dot_size = 2
    palette_name = None
    hue_order = ["zoo", "random", "adversary"]
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
        raise ValueError(
            "No sacred directory found for base path {}, param dict {}".format(base_path, param_dict))
    return str(max_int_dir)


@tsne_vis_ex.capture
def _plot_and_save_chart(data, fname, chart_type, opacity, dot_size, palette_name, hue_order):
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, fname)

        if chart_type == 'altair':
            chart = alt.Chart(data).mark_point().configure_mark(
                opacity=opacity).encode(
                x='ax_1', y='ax_2', color='opponent_id')
            chart.save(fname)
        elif chart_type == 'seaborn':
            if palette_name is None:
                palette_name = {
                    "zoo": "#66c2a5",
                    "random": "#fdb462",
                    "adversary": '#e7298a'
                }
            plt.figure(figsize=(12, 8))
            sns.scatterplot(data=data, x="ax_1", y="ax_2", hue="opponent_id",
                            alpha=opacity, s=dot_size, edgecolors='none',
                            palette=palette_name, hue_order=hue_order)
            plt.savefig(fname)
            plt.close()
        tsne_vis_ex.add_artifact(fname)


def sample_range(df, xrange, yrange):
    xrange_satisfies = df[(df['ax_1'] > xrange[0]) & (df['ax_1'] < xrange[1])]
    both_satisfies = xrange_satisfies[(xrange_satisfies['ax_2'] > yrange[0]) & (
        xrange_satisfies['ax_2'] < yrange[1])]
    return both_satisfies.sample(1).iloc[0]


@tsne_vis_ex.capture
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


@tsne_vis_ex.automain
def experiment_main(base_path, sacred_id, subsample_rate, perplexity):

    if sacred_id is None:
        sacred_id = _get_latest_sacred_dir_with_params(base_path, {'perplexity': perplexity})

    logger_obj.info("For perplexity {}, using sacred dir {}".format(perplexity, sacred_id))

    full_sacred_dir = os.path.join(base_path, sacred_id)

    metadata_df = pd.read_csv(os.path.join(full_sacred_dir, 'metadata.csv'))
    cluster_ids = np.load(os.path.join(full_sacred_dir, 'cluster_ids.npy'))
    metadata_df['ax_1'] = cluster_ids[:, 0]
    metadata_df['ax_2'] = cluster_ids[:, 1]

    _plot_and_save_chart(metadata_df.query("opponent_id != 'random'"), "no_random_chart.png")
    _plot_and_save_chart(metadata_df.query("opponent_id != 'adversary'"), "no_adversary_chart.png")
    _plot_and_save_chart(metadata_df, "opponent_chart.png")
    _plot_and_save_chart(metadata_df.sample(frac=subsample_rate), fname="subsample_chart.png")
    opponent_groups = metadata_df.groupby('opponent_id')

    for name, group in opponent_groups:
        _plot_and_save_chart(group, "{}_chart.png".format(name))
