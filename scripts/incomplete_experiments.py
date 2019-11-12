import argparse
import logging
import os

import pandas as pd


logger = logging.getLogger('scripts.incomplete_experiments')


def directory_type(path):
    if not os.path.isdir(path):
        raise ValueError(f"'{path}' does not exist")
    return path


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=directory_type)
    return parser.parse_args()


def get_stats(data_dir):
    started = {}
    completed = {}
    for root, dirs, files in os.walk(data_dir, followlinks=True):
        # checkpoint directories are irrelevant and will slow down search
        logger.debug(f"Searching '{root}'")
        dirs[:] = list(filter(lambda x: x not in ['checkpoint', 'mon', 'tb'], dirs))
        components = root.split(os.path.sep)

        if 'final_model' in dirs:
            # root is of format .../exp_name/timestamp/run_id/data/baselines/run_id
            assert components[-2] == 'baselines'
            logger.debug(f"Found final_model in '{root}'")
            exp_name = os.path.relpath(os.path.join(*components[:-5]), data_dir)
            completed[exp_name] = completed.get(exp_name, 0) + 1
            dirs[:] = []  # no need to search further in data/baselines/*
        elif 'sacred' in dirs:
            # root is of format ../exp_name/timestamp/run_id/data/sacred
            assert components[-1] == 'data'
            logger.debug(f"Found sacred at '{root}'")
            exp_name = os.path.relpath(os.path.join(*components[:-3]), data_dir)
            started[exp_name] = started.get(exp_name, 0) + 1
            dirs.remove('sacred')  # don't need to search inside it

    return started, completed


def compute_incompletes(started, completed):
    incomplete = {k: num_started - completed.get(k, 0) for k, num_started in started.items()}
    percent_incomplete = {k: num_incomplete / started[k]
                          for k, num_incomplete in incomplete.items()}
    percent_incomplete = pd.Series(percent_incomplete)
    percent_incomplete = percent_incomplete.sort_values(ascending=False)
    percent_incomplete.index.name = 'path'
    percent_incomplete.name = 'percent_incomplete'
    return percent_incomplete


def main():
    logging.basicConfig(level=logging.INFO)
    args = get_args()
    started, completed = get_stats(args.data_dir)
    percent_incomplete = compute_incompletes(started, completed)
    print(percent_incomplete.to_csv(header=True))


if __name__ == '__main__':
    main()
