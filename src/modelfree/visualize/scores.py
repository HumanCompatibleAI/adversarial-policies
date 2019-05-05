from distutils.dir_util import copy_tree
import logging
import os.path

import matplotlib.pyplot as plt
import pandas as pd
from sacred import Experiment
from sacred.observers import FileStorageObserver

from modelfree.visualize import util
from modelfree.visualize.styles import STYLES

logger = logging.getLogger('modelfree.visualize.scores')
visualize_score_ex = Experiment('visualize_score')


def heatmap_opponent(single_env):
    cbar = single_env.name == 'multicomp/YouShallNotPassHumans-v0'
    ylabel = single_env.name == 'multicomp/KickAndDefend-v0'
    return util.heatmap_one_col(single_env, col='Opponent Win', cbar=cbar, ylabel=ylabel)


@visualize_score_ex.config
def default_config():
    fig_dir = os.path.join('data', 'figs', 'scores')
    transfer_score_root = os.path.join('data', 'aws', 'score_agents')
    transfer_score_paths = {}
    if not hasattr(transfer_score_paths, 'fixed'):  # work-around Sacred issue #238
        transfer_score_paths[None] = os.path.join('normal', '2019-04-29T14:11:08-07:00')
    styles = ['paper', 'a4']
    command = util.heatmap_full
    publication = False
    seed = 0  # we don't use it for anything, but stop config changing each time as we version it
    _ = locals()  # quieten flake8 unused variable warning
    del _


@visualize_score_ex.named_config
def masked_config():
    transfer_score_paths = {  # noqa: F841
        'N': os.path.join('normal', '2019-04-29T14:11:08-07:00'),
        'B': os.path.join('victim_masked_init', '2019-04-29T14:11:08-07:00'),
    }


@visualize_score_ex.named_config
def use_heatmap_opponent():
    command = heatmap_opponent  # noqa: F841


@visualize_score_ex.named_config
def paper_config():
    fig_dir = os.path.expanduser('~/dev/adversarial-policies-paper/figs/scores_single')
    styles = ['paper', 'scores_threecol']
    command = heatmap_opponent
    publication = True
    _ = locals()  # quieten flake8 unused variable warning
    del _


@visualize_score_ex.named_config
def supplementary_config():
    fig_dir = os.path.expanduser('~/dev/adversarial-policies-paper/figs/scores')
    styles = ['paper', 'scores_monolithic']
    publication = True
    _ = locals()  # quieten flake8 unused variable warning
    del _


@visualize_score_ex.main
def visualize_score(command, styles, publication, fig_dir,
                    transfer_score_root, transfer_score_paths):
    print(transfer_score_paths)
    datasets = [util.load_datasets(os.path.join(transfer_score_root, path),
                                   victim_suffix=k if k is not None else "")
                for k, path in transfer_score_paths.items()]
    dataset = pd.concat(datasets).sort_index()

    for style in styles:
        plt.style.use(STYLES[style])

    suptitle = not publication
    combine = not publication
    generator = util.apply_per_env(dataset, command, suptitle=suptitle)
    for out_path in util.save_figs(fig_dir, generator, combine=combine):
        visualize_score_ex.add_artifact(filename=out_path)

    for observer in visualize_score_ex.observers:
        if hasattr(observer, 'dir'):
            logger.info(f"Copying from {observer.dir} to {fig_dir}")
            copy_tree(observer.dir, fig_dir)
            break


def main():
    observer = FileStorageObserver.create(os.path.join('data', 'sacred', 'visualize_score'))
    visualize_score_ex.observers.append(observer)
    visualize_score_ex.run_commandline()


if __name__ == '__main__':
    main()
