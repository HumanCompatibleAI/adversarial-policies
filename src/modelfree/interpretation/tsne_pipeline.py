import logging
import os
import os.path as osp

import sacred
from sacred.observers import FileStorageObserver

from modelfree.common import utils
from modelfree.interpretation.fit_tsne import fit_tsne_ex
from modelfree.interpretation.visualize_tsne import vis_tsne_ex
from modelfree.multi.score import extract_data, run_external

tsne_ex = sacred.Experiment('tsne', ingredients=[fit_tsne_ex, vis_tsne_ex])
logger = logging.getLogger('modelfree.interpretation.tsne_pipeline')


@tsne_ex.config
def activation_storing_config():
    adversary_path = osp.join('data', 'aws', 'score_agents',
                              '2019-04-29T14:11:08-07:00_best_adversaries.json')
    ray_upload_dir = 'data'     # where Ray will upload multi.score outputs. 'data' works on local
    output_root = 'data/tsne'   # where to produce output

    score_configs = ['zoo_baseline', 'random_baseline', 'adversary_trained']
    score_update = {}

    exp_name = 'default'        # experiment name

    _ = locals()    # quieten flake8 unused variable warning
    del _


@tsne_ex.named_config
def debug_config():
    score_configs = ['debug_one_each_type']
    exp_name = 'debug'

    _ = locals()    # quieten flake8 unused variable warning
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
def pipeline(output_root, exp_name, fit_tsne, visualize_tsne):
    out_dir = osp.join(output_root, exp_name, utils.make_timestamp())
    os.makedirs(out_dir)

    logger.info("Generating activations")
    activation_src_dirs = store_activations()
    activation_dst_dir = osp.join(out_dir, 'activations')
    os.makedirs(activation_dst_dir)
    extract_activations(activation_dst_dir, activation_src_dirs)
    logger.info("Activations saved")

    logger.info("Fitting t-SNE")
    fit_tsne['activation_dir'] = activation_dst_dir
    model_dir = osp.join(out_dir, 'fitted')
    fit_tsne['output_root'] = model_dir
    fit_tsne_ex.run(config_updates=fit_tsne)
    logger.info("Fitting complete")

    logger.info("Generating figures")
    visualize_tsne['model_dir'] = osp.join(model_dir, '*')
    visualize_tsne['output_dir'] = osp.join(out_dir, 'figures')
    vis_tsne_ex.run(config_updates=visualize_tsne)
    logger.info("Visualization complete")


def main():
    observer = FileStorageObserver.create(osp.join('data', 'sacred', 'tsne'))
    tsne_ex.observers.append(observer)
    tsne_ex.run_commandline()


if __name__ == '__main__':
    main()
