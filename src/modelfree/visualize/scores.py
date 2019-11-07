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


@visualize_score_ex.capture
def heatmap_opponent(single_env, cmap, row_starts, row_ends, col_ends):
    xlabel = single_env.name in col_ends
    ylabel = single_env.name in row_starts
    cbar = single_env.name in row_ends
    return util.heatmap_one_col(single_env, col='Opponent Win', cmap=cmap,
                                xlabel=xlabel, ylabel=ylabel, cbar=cbar)


SMALL_TRANSFER_SCORE_PATHS = [
    {'victim_suffix': '', 'path': os.path.join('normal', '2019-05-05T18:12:24+00:00')},
    {
        'victim_suffix': 'M',
        'path': os.path.join('victim_masked_init', '2019-05-05T18:12:24+00:00'),
    },
]


@visualize_score_ex.config
def default_config():
    transfer_score_root = os.path.join('data', 'aws', 'score_agents')
    transfer_score_paths = [{'path': os.path.join('normal', '2019-05-05T18:12:24+00:00')}]

    command = util.heatmap_full
    styles = ['paper', 'a4']
    palette = 'Blues'
    publication = False

    fig_dir = os.path.join('data', 'figs', 'scores')

    seed = 0  # we don't use it for anything, but stop config changing each time as we version it

    _ = locals()  # quieten flake8 unused variable warning
    del _


@visualize_score_ex.named_config
def full_masked_config():
    transfer_score_paths = [  # noqa: F841
        {'victim_suffix': 'N', 'path': os.path.join('normal', '2019-05-05T18:12:24+00:00')},
        {
            'victim_suffix': 'BI',
            'path': os.path.join('victim_masked_init', '2019-05-05T18:12:24+00:00')
        },
        {
            'victim_suffix': 'BZ',
            'path': os.path.join('victim_masked_zero', '2019-05-05T18:12:24+00:00'),
        },
        {
            'victim_suffix': 'N',
            'opponent_suffix': 'BI',
            'path': os.path.join('adversary_masked_init', '2019-05-05T18:12:24+00:00'),
        },
    ]


@visualize_score_ex.named_config
def paper_config():
    transfer_score_paths = SMALL_TRANSFER_SCORE_PATHS

    styles = ['paper', 'scores_twocol']
    row_starts = ['multicomp/KickAndDefend-v0', 'multicomp/SumoHumansAutoContact-v0']
    row_ends = ['multicomp/YouShallNotPassHumans-v0', 'multicomp/SumoAntsAutoContact-v0']
    col_ends = ['multicomp/SumoHumansAutoContact-v0', 'multicomp/SumoAntsAutoContact-v0']
    command = heatmap_opponent
    publication = True

    fig_dir = os.path.expanduser('~/dev/adversarial-policies-paper/figs/scores_single')

    _ = locals()  # quieten flake8 unused variable warning
    del _


@visualize_score_ex.named_config
def supplementary_config():
    transfer_score_paths = SMALL_TRANSFER_SCORE_PATHS

    styles = ['paper', 'scores_monolithic']
    publication = True

    fig_dir = os.path.expanduser('~/dev/adversarial-policies-paper/figs/scores')

    _ = locals()  # quieten flake8 unused variable warning
    del _


@visualize_score_ex.named_config
def poster_config():
    transfer_score_paths = SMALL_TRANSFER_SCORE_PATHS

    styles = ['poster', 'scores_poster_threecol']
    row_starts = ['multicomp/KickAndDefend-v0']
    row_ends = ['multicomp/YouShallNotPassHumans-v0']
    col_ends = ['multicomp/KickAndDefend-v0', 'multicomp/SumoHumansAutoContact-v0',
                'multicomp/YouShallNotPassHumans-v0']
    command = heatmap_opponent
    publication = True

    fig_dir = os.path.expanduser('~/dev/adversarial-policies-paper/figs/scores_poster')

    _ = locals()  # quieten flake8 unused variable warning
    del _


@visualize_score_ex.main
def visualize_score(command, styles, palette, publication, fig_dir,
                    transfer_score_root, transfer_score_paths):
    datasets = [util.load_datasets(os.path.join(transfer_score_root, item['path']),
                                   victim_suffix=item.get('victim_suffix', ''),
                                   opponent_suffix=item.get('opponent_suffix', ''))
                for item in transfer_score_paths]
    dataset = pd.concat(datasets)

    for style in styles:
        plt.style.use(STYLES[style])

    suptitle = not publication
    combine = not publication
    generator = util.apply_per_env(dataset, command, suptitle=suptitle, cmap=palette)
    for out_path in util.save_figs(fig_dir, generator, combine=combine):
        visualize_score_ex.add_artifact(filename=out_path)

    for observer in visualize_score_ex.observers:
        if hasattr(observer, 'dir'):
            logger.info(f"Copying from {observer.dir} to {fig_dir}")
            copy_tree(observer.dir, fig_dir)
            break


def main():
    observer = FileStorageObserver(os.path.join('data', 'sacred', 'visualize_score'))
    visualize_score_ex.observers.append(observer)
    visualize_score_ex.run_commandline()


if __name__ == '__main__':
    main()
