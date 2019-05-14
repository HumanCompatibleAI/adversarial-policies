import logging
import os
import os.path as osp

import sacred
from sacred.observers import FileStorageObserver

from modelfree.common import utils
from modelfree.density.fit_density import fit_model, fit_model_ex
from modelfree.tsne.generate_activations import generate_activations, generate_activations_ex

density_ex = sacred.Experiment('density', ingredients=[generate_activations_ex, fit_model_ex])
logger = logging.getLogger('modelfree.density.pipeline')


# Questions: How do you set named_configs for ingredients?

@density_ex.config
def activation_storing_config():
    output_root = 'data/density'   # where to produce output
    exp_name = 'default'        # experiment name

    _ = locals()    # quieten flake8 unused variable warning
    del _


@density_ex.named_config
def debug_config(tsne_activations, fit_density_model):
    # Is this the name of an ingredient? Is it being auto-added to config somehow?
    tsne_activations = dict(tsne_activations)
    fit_density_model = fit_density_model.copy()
    tsne_activations['adversary_path'] = os.path.join('data', 'aws', 'score_agents',
                                                      '2019-05-05T18:12:24+00:00',
                                                      'best_adversaries.json')

    tsne_activations['score_configs'] = ['debug_one_each_type']

    exp_name = 'debug'
    fit_density_model['num_observations'] = 1000
    fit_density_model['model_type'] = 'KDE'
    _ = locals()    # quieten flake8 unused variable warning
    del _


@density_ex.main
def pipeline(_run, output_root, exp_name):
    out_dir = osp.join(output_root, exp_name, utils.make_timestamp())
    os.makedirs(out_dir)

    activation_dst_dir = osp.join(out_dir, 'activations')
    generate_activations(out_dir=activation_dst_dir)

    model_dir = osp.join(out_dir, 'fitted')
    fit_model(activation_dir=activation_dst_dir, output_root=model_dir)

    return out_dir


def main():
    observer = FileStorageObserver.create(osp.join('data', 'sacred', 'density'))
    density_ex.observers.append(observer)
    density_ex.run_commandline()


if __name__ == '__main__':
    main()
