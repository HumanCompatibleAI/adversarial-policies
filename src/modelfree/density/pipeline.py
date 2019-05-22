import logging
import os
import os.path as osp

import sacred
from sacred.observers import FileStorageObserver

from modelfree.common import utils
from modelfree.common.generate_activations import generate_activations, generate_activations_ex
from modelfree.density.fit_density import fit_model, fit_model_ex, gen_exp_name

density_ex = sacred.Experiment('density', ingredients=[generate_activations_ex, fit_model_ex])
logger = logging.getLogger('modelfree.density.pipeline')



@density_ex.config
def main_config(generate_activations, fit_density_model):
    generate_activations = dict(generate_activations)
    generate_activations['score_update'] = {'score': {'timesteps': 40000}}

    output_root = osp.join('data', 'density')                  # where to produce output
    _ = locals()    # quieten flake8 unused variable warning
    del _


@density_ex.named_config
def debug_config(generate_activations, fit_density_model):
    # Is this the name of an ingredient? Is it being auto-added to config somehow?
    generate_activations = dict(generate_activations)
    fit_density_model = fit_density_model.copy()

    generate_activations['score_configs'] = ['debug_one_each_type']
    exp_name = 'debug'
    fit_density_model['num_observations'] = 1000
    _ = locals()    # quieten flake8 unused variable warning
    del _


@density_ex.main
def pipeline(_run, output_root, fit_density_model):
    exp_name = gen_exp_name(fit_density_model['model_class'], fit_density_model['model_kwargs'])
    out_dir = osp.join(output_root, exp_name, utils.make_timestamp())
    os.makedirs(out_dir)

    activation_dir = fit_density_model['activation_dir']
    if activation_dir is None:
        activation_dir = osp.join(out_dir, 'activations')
        generate_activations(out_dir=activation_dir)

    model_dir = osp.join(out_dir, 'fitted')
    fit_model(activation_dir=activation_dir, output_root=model_dir)

    return out_dir


def main():
    observer = FileStorageObserver.create(osp.join('data', 'sacred', 'density'))
    density_ex.observers.append(observer)
    density_ex.run_commandline()


if __name__ == '__main__':
    main()
