import logging
import os
import os.path as osp

import sacred
from sacred.observers import FileStorageObserver

from modelfree.common import utils
from modelfree.interpretation.fit_tsne import fit_tsne_ex
from modelfree.interpretation.visualize_tsne import vis_tsne_ex
from modelfree.multi.score import extract_data, run_external

tsne_ex = sacred.Experiment('tsne')
logger = logging.getLogger('modelfree.interpretation.tsne_pipeline')


@tsne_ex.config
def activation_storing_config():
    # TODO: portability
    adversary_path = osp.join('data', 'score_agents',
                              '2019-04-29T14:11:08-07:00_best_adversaries.json')
    ray_upload_dir = 'data'  # where Ray will upload multi.score outputs. 'data' works on local
    root_dir = 'data/tsne'
    score_configs = ['zoo_baseline', 'fixed_baseline', 'adversary_transfer']
    score_update = {}

    perplexity = 250
    fit_tsne_configs = dict(
        relative_dirs=['adversary', 'random', 'zoo'],
        data_type='ff_policy',
        num_components=2,
        sacred_dir_ids=None,
        np_file_name="victim_activations.npz",
        num_observations=None,
        env_name='multicomp/YouShallNotPassHumans-v0',
        seed=0,
        perplexity=perplexity,
    )

    visualize_configs = dict(
        base_path=None,
        subsample_rate=0.15,
        perplexity=perplexity,
        video_path="data/video_frames",
        chart_type="seaborn",
        opacity=0.75,
        dot_size=0.25,
        palette_name="cube_bright",
        save_type="pdf",
        hue_order=["adversary", "zoo", "random"],
    )

    skip_scoring = False
    exp_name = 'default'

    _ = locals()
    del _


@tsne_ex.named_config
def debug_config():
    score_configs = ['debug']
    exp_name = 'debug'

    _ = locals()
    del _


# TODO: make separate experiment?
@tsne_ex.capture
def store_activations(score_configs, score_update, adversary_path):
    """Uses multi.score to save activations."""
    return run_external(score_configs, post_named_configs=['save_activations'],
                        config_updates=score_update, adversary_path=adversary_path)


@tsne_ex.capture
def extract_activations(out_dir, activation_dirs, ray_upload_dir):
    def path_generator(trial_root, env_name, victim_index, victim_type, victim_path,
                       opponent_type, opponent_path):
        src_path = osp.join(trial_root, 'data',
                            'trajectories', f'agent_{victim_index}.npz')
        new_name = (f'{env_name}_victim_{victim_type}_{victim_path}'
                    f'_opponent_{opponent_type}_{opponent_path}.npz')
        return src_path, new_name

    return extract_data(path_generator, out_dir, activation_dirs, ray_upload_dir)


@tsne_ex.main
def pipeline(root_dir, exp_name, fit_tsne_configs, visualize_configs):
    out_dir = osp.join(root_dir, exp_name, utils.make_timestamp())
    os.makedirs(out_dir)

    logger.info("Generating activations")
    activation_dirs = store_activations()
    extract_activations(out_dir, activation_dirs)
    logger.info("Activations saved")

    logger.info("Fitting t-SNE")
    fitted_tsne_path = os.path.join(tsne_ex.observers[0].dir, "fit")
    fitting_observer = FileStorageObserver.create(fitted_tsne_path)
    fit_tsne_ex.observers.append(fitting_observer)
    fit_tsne_configs['sacred_dir_ids'] = activation_dirs
    fit_tsne_ex.run(config_updates=fit_tsne_configs)
    logger.info("Fitting complete")

    visualize_tsne_path = os.path.join(tsne_ex.observers[0].dir, "visualize")
    visualize_observer = FileStorageObserver.create(visualize_tsne_path)
    vis_tsne_ex.observers.append(visualize_observer)
    visualize_configs["sacred_id"] = fitting_observer.dir
    vis_tsne_ex.run(config_updates=visualize_configs)
    logger.info("Visualization complete")


def main():
    observer = FileStorageObserver.create(osp.join('data', 'sacred', 'tsne'))
    tsne_ex.observers.append(observer)
    tsne_ex.run_commandline()


if __name__ == '__main__':
    main()
