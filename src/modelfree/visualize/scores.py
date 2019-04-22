import argparse
import os.path

import matplotlib.pyplot as plt

from modelfree.visualize import util
from modelfree.visualize.styles import STYLES


def directory(path):
    if not os.path.exists(path):
        raise ValueError(f"Path '{path}' does not exist")
    if not os.path.isdir(path):
        raise ValueError(f"Path '{path}' is not a directory")
    return path


def style(key):
    if key not in STYLES:
        raise ValueError(f"Unrecognized style '{key}'")
    return key


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fig-dir', type=directory,
                        default=os.path.join('data', 'figs', 'scores'))
    parser.add_argument('--score-dir', type=directory,
                        default=os.path.join('data', 'score_agents'))
    parser.add_argument('--style', type=style, default=['paper', 'a4'], nargs='+')
    return parser.parse_args()


def load_datasets(args):
    score_dir = args.score_dir
    fixed = util.load_fixed_baseline(os.path.join(score_dir, 'fixed_baseline.json'))
    zoo = util.load_zoo_baseline(os.path.join(score_dir, 'zoo_baseline.json'))
    transfer = util.load_transfer_baseline(os.path.join(score_dir, 'adversary_transfer.json'))
    return util.combine_all(fixed, zoo, transfer)


def main():
    args = get_args()
    dataset = load_datasets(args)

    for style in args.style:
        plt.style.use(STYLES[style])

    generator = util.apply_per_env(dataset, util.heatmap)
    util.save_figs(args.fig_dir, generator)


if __name__ == '__main__':
    main()
