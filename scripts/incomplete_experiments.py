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
        dirs[:] = list(filter(lambda x: x != 'checkpoint', dirs))
        components = root.split(os.path.sep)

        # components of format .../exp_name/timestamp/run_id/data/baselines/run_id/final_model
        finalized = components[-1] == 'final_model' and components[-3] == 'baselines'
        if finalized:
            exp_name = os.path.relpath(os.path.join(*components[:-5]), data_dir)
            completed[exp_name] = completed.get(exp_name, 0) + 1

        # components of format ../exp_name/timestamp/run_id/data/sacred
        sacred_exists = components[-1] == 'sacred' and components[-2] == 'data'
        if sacred_exists:
            exp_name = os.path.relpath(os.path.join(*components[:-3]), data_dir)
            started[exp_name] = started.get(exp_name, 0) + 1

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
