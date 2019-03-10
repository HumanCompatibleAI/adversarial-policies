"""Logging for RL algorithms.

Configures Baseline's logger and TensorBoard appropriately."""

import datetime
import os
from os import path as osp

from stable_baselines import logger
from tensorboard.plugins.custom_scalar import layout_pb2
import tensorboard.summary as summary_lib
from tensorflow.core.util import event_pb2


def gen_multiline_charts(cfg):
    charts = []
    for title, tags in cfg:
        charts.append(layout_pb2.Chart(
            title=title,
            multiline=layout_pb2.MultilineChartContent(tag=tags)
        ))
    return charts


def tb_layout():
    episode_rewards = layout_pb2.Category(
        title='Episode Reward',
        chart=gen_multiline_charts([
            ("Shaped Reward", [r'eprewmean_true']),
            ("Episode Length", [r'eplenmean']),
            ("Sparse Reward", [r'epsparsemean']),
            ("Dense Reward", [r'epdensemean']),
            ("Dense Reward Annealing", [r'rew_anneal']),
            ("Unshaped Reward", [r'ep_rewmean']),
        ]),
    )

    game_outcome = layout_pb2.Category(
        title="Game Outcomes",
        chart=gen_multiline_charts([
            ("Agent 0 Win Proportion", [r'game_win0']),
            ("Agent 1 Win Proportion", [r'game_win1']),
            ("Tie Proportion", [r'game_tie']),
            ("# of games", [r'game_total']),
        ]),
    )

    training = layout_pb2.Category(
        title="Training",
        chart=gen_multiline_charts([
            ("Policy Loss", [r'policy_loss']),
            ("Value Loss", [r'value_loss']),
            ("Policy Entropy", [r'policy_entropy']),
            ("Explained Variance", [r'explained_variance']),
            ("Approx KL", [r'approxkl']),
            ("Clip Fraction", [r'clipfrac']),
        ])
    )

    # Intentionally unused:
    # + serial_timesteps (just total_timesteps / num_envs)
    # + time_elapsed (TensorBoard already logs wall-clock time)
    # + nupdates (this is already logged as step)
    time = layout_pb2.Category(
        title="Time",
        chart=gen_multiline_charts([
            ("Total Timesteps", [r'total_timesteps']),
            ("FPS", [r'fps']),
        ])
    )

    categories = [episode_rewards, game_outcome, training, time]
    return summary_lib.custom_scalar_pb(layout_pb2.Layout(category=categories))


def make_timestamp():
    ISO_TIMESTAMP = "%Y%m%d_%H%M%S"
    return datetime.datetime.now().strftime(ISO_TIMESTAMP)


def setup_logger(out_dir='results', exp_name='test', output_formats=None):
    timestamp = make_timestamp()
    exp_name = exp_name.replace('/', '_')  # environment names can contain /'s
    out_dir = osp.join(out_dir, '{}-{}'.format(timestamp, exp_name))
    os.makedirs(out_dir, exist_ok=True)

    logger.configure(folder=osp.join(out_dir, 'rl'),
                     format_strs=['tensorboard', 'stdout'])
    logger_instance = logger.Logger.CURRENT

    if output_formats is not None:
        logger_instance.output_formats += output_formats

    for fmt in logger_instance.output_formats:
        if isinstance(fmt, logger.TensorBoardOutputFormat):
            writer = fmt.writer
            layout = tb_layout()
            event = event_pb2.Event(summary=layout)
            writer.WriteEvent(event)
            writer.Flush()

    return out_dir, logger_instance
