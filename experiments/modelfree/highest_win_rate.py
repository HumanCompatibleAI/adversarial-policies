#!/usr/bin/env python3

"""Processes experimental output to find adversarial policies with maximal win rate."""

import argparse
import collections
import json
import logging
import os.path

import numpy as np
import tensorflow as tf

logger = logging.getLogger('scripts.highest_win_rate')


def event_files(path):
    for root, dirs, files in os.walk(path, followlinks=True):
        if root.endswith('tb'):  # looking for paths of form */data/baselines/*/rl/tb
            for name in files:
                if 'tfevents' in name:
                    yield os.path.join(root, name)


def get_stats(event_path, episode_window):
    events = collections.defaultdict(list)
    last_step = 0
    for event in tf.train.summary_iterator(event_path):
        for value in event.summary.value:
            if value.tag in ['game_win0', 'game_win1', 'game_tie']:
                events[value.tag].append(value.simple_value)
                last_step = event.step

    logger.info(f"Read {len(events['game_win0'])} events up to {last_step} from '{event_path}'")
    means = {k: np.mean(v[-episode_window:]) for k, v in events.items()}

    return means


def _strip_up_to(path, dirname):
    path_components = path.split(os.path.sep)
    try:
        path_index = len(path_components) - 1 - path_components[::-1].index(dirname)
    except ValueError as e:
        raise ValueError(f"Error stripping '{dirname}' in '{path_components}': {e}")
    return os.path.join(*path_components[0:path_index])


def get_sacred_config(event_path):
    root = _strip_up_to(event_path, 'baselines')
    sacred_config_path = os.path.join(root, 'sacred', 'train', '1', 'config.json')
    with open(sacred_config_path, 'r') as f:
        return json.load(f)


def get_final_model_path(event_path):
    root = _strip_up_to(event_path, 'rl')
    return os.path.join(root, 'final_model')


def unstack(d):
    d = collections.OrderedDict(sorted(d.items()))
    res = collections.OrderedDict()
    for k, v in d.items():
        env_name, opp_index, opp_path = k
        res.setdefault(env_name, {}).setdefault(opp_index, {})[opp_path] = v
    return res


def find_best(logdirs, episode_window):
    # keys: (env_name, opp_index, opp_path)
    # value: path to policy evaluated on env_name against opponent opp_path playing opp_index
    best_policy = {}
    best_winrate = collections.defaultdict(float)

    for logdir in logdirs:
        for event_path in event_files(logdir):
            stats = get_stats(event_path=event_path, episode_window=episode_window)
            config = get_sacred_config(event_path)
            env_name = config['env_name']
            opp_index = config['victim_index']
            opp_type = config['victim_type']
            # multi_score is not set up to handle multiple victim types
            assert opp_type == 'zoo'
            opp_path = config['victim_path']

            our_index = 1 - opp_index
            key = (env_name, opp_index, opp_path)
            our_winrate = stats[f'game_win{our_index}']

            if our_winrate > best_winrate[key]:
                best_policy[key] = get_final_model_path(event_path)
                best_winrate[key] = our_winrate

    result = {
        'policies': unstack(best_policy),
        'winrates': unstack(best_winrate),
    }

    return result


def directory_type(path):
    if not os.path.isdir(path):
        raise ValueError(f"'{path}' does not exist")
    return path


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', nargs='*', type=directory_type)
    parser.add_argument('--episode-window', type=int, default=50)
    parser.add_argument('output_path')
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO)
    args = get_args()

    with open(args.output_path, 'w') as f:  # fail fast if output_path inaccessible
        result = find_best(args.logdir, args.episode_window)
        json.dump(result, f)


if __name__ == '__main__':
    main()
