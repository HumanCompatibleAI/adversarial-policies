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
    score_configs = ['zoo_baseline', 'random_baseline', 'adversary_trained']
    score_update = {}

    perplexity = 250
    fit_tsne_configs = dict(
        data_type='ff_policy',
        num_components=2,
        num_observations=None,
        seed=0,
        perplexity=perplexity,
    )

    visualize_configs = dict(
        subsample_rate=0.15,
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
                    f'_opponent_{opponent_type}_{opponent_path}')
        return src_path, new_name, 'npz'

    return extract_data(path_generator, out_dir, activation_dirs, ray_upload_dir)


@tsne_ex.main
def pipeline(root_dir, exp_name, fit_tsne_configs, visualize_configs):
    out_dir = osp.join(root_dir, exp_name, utils.make_timestamp())
    os.makedirs(out_dir)

    logger.info("Generating activations")
    activation_src_dirs = store_activations()
    activation_dst_dir = osp.join(out_dir, 'activations')
    os.makedirs(activation_dst_dir)
    extract_activations(activation_dst_dir, activation_src_dirs)
    logger.info("Activations saved")

    # TODO: multiple applications
    # TODO: parallelize?
    logger.info("Fitting t-SNE")
    fit_tsne_configs['activation_path'] = activation_dst_dir
    model_dir = osp.join(out_dir, 'fitted')
    fit_tsne_configs['output_root'] = model_dir
    fit_tsne_ex.run(config_updates=fit_tsne_configs)
    logger.info("Fitting complete")

    logger.info("Generating figures")
    # TODO: multiple applications, remove hardcoding
    visualize_configs['model_dir'] = osp.join(model_dir, 'SumoAnts-v0_victim_zoo_1')
    visualize_configs['output_dir'] = osp.join(out_dir, 'figures')
    vis_tsne_ex.run(config_updates=visualize_configs)
    logger.info("Visualization complete")


def main():
    observer = FileStorageObserver.create(osp.join('data', 'sacred', 'tsne'))
    tsne_ex.observers.append(observer)
    tsne_ex.run_commandline()


if __name__ == '__main__':
    main()
