import collections
import fnmatch
import functools
import itertools
import json
import logging
import multiprocessing
import os
import traceback

import tensorflow as tf

logger = logging.getLogger('aprl.visualize.tb')


def find_tfevents(log_dir):
    result = []
    for root, dirs, files in os.walk(log_dir, followlinks=True):
        if root.endswith('rl/tb'):
            for name in files:
                if fnmatch.fnmatch(name, 'events.out.tfevents.*'):
                    result.append(os.path.join(root, name))
    return result


def exp_root_from_event(event_path):
    # tb_dirname = ...experiment/data/baselines/TIMESTAMP/rl/tb/events.*
    # exp_root = ...experiment/
    return os.path.sep.join(event_path.split(os.path.sep)[:-6])


def read_events_file(events_filename, keys=None):
    events = []
    try:
        for event in tf.train.summary_iterator(events_filename):
            row = {'wall_time': event.wall_time, 'step': event.step}
            for value in event.summary.value:
                if keys is not None and value.tag not in keys:
                    continue
                row[value.tag] = value.simple_value
            events.append(row)
    except Exception:
        logger.error(f"While reading '{events_filename}': {traceback.print_exc()}")
    return events


def read_sacred_config(exp_root, kind):
    sacred_config_path = os.path.join(exp_root, 'data', 'sacred', kind, '1', 'config.json')
    with open(sacred_config_path, 'r') as f:
        return json.load(f)


def load_tb_data(log_dir, keys=None):
    event_paths = find_tfevents(log_dir)

    pool = multiprocessing.Pool()
    events_by_path = pool.map(functools.partial(read_events_file, keys=keys), event_paths)

    events_by_dir = {}
    for event_path, events in zip(event_paths, events_by_path):
        exp_root = exp_root_from_event(event_path)
        if exp_root not in events_by_dir:
            events_by_dir[exp_root] = []
        events_by_dir[exp_root] += events

    config_by_dir = {dirname: read_sacred_config(dirname, 'train')
                     for dirname in events_by_dir.keys()}

    return config_by_dir, events_by_dir


def split_by_keys(configs, events, keys):
    res = collections.defaultdict(list)
    for dirname, config in configs.items():
        event = events[dirname]
        cfg_vals = tuple(config[k] for k in keys)
        res[cfg_vals].append({
            'dir': dirname,
            'config': config,
            'events': event,
        })
    return res


def tb_apply(configs, events, split_keys, fn, **kwargs):
    events_by_plot = split_by_keys(configs, events, split_keys)

    pool = multiprocessing.Pool()
    map_fn = functools.partial(fn, **kwargs)
    res = pool.map(map_fn, events_by_plot.items())
    res = itertools.chain(*res)
    return res
