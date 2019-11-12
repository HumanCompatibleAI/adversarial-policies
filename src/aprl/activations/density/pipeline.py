"""Records activations from victim's policy network and then fits a density model."""

import logging
import os
import os.path as osp

import sacred
from sacred.observers import FileStorageObserver

from aprl.activations import generate_activations
from aprl.activations.density.fit_density import fit_model, fit_model_ex
from aprl.common import utils

density_ex = sacred.Experiment('density',
                               ingredients=[generate_activations.generate_activations_ex,
                                            fit_model_ex])
logger = logging.getLogger('aprl.density.pipeline')


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
    output_root = '/tmp/density-debug'
    generate_activations = dict(generate_activations)
    fit_density_model = dict(fit_density_model)

    generate_activations['score_configs'] = [('debug_two_agents', )]
    generate_activations['score_update'] = {'score': {'timesteps': 100}}
    fit_density_model['max_timesteps'] = 100
    fit_density_model['model_kwargs'] = {'n_components': 2}

    _ = locals()    # quieten flake8 unused variable warning
    del _


@density_ex.main
def pipeline(_run, output_root, fit_density_model):
    out_dir = osp.join(output_root, utils.make_timestamp())
    os.makedirs(out_dir)

    activation_glob = fit_density_model['activation_glob']
    if activation_glob is None:
        activation_dir = osp.join(out_dir, 'activations')
        generate_activations.generate_activations(out_dir=activation_dir)
        activation_glob = osp.join(activation_dir, '*')

    # This is unsuitable for hyperparameter sweeps, as can only run one model fitting step.
    # See experiments/modelfree/density.sh for a bash script hyperparameter sweep, that
    # re-uses activations.
    # SOMEDAY: Add support for running multiple fitting configs?
    # (Does not neatly fit into Sacred model.)
    model_dir = osp.join(out_dir, 'fitted')
    fit_model(activation_glob=activation_glob, output_root=model_dir)

    return out_dir


def main():
    observer = FileStorageObserver(osp.join('data', 'sacred', 'density'))
    density_ex.observers.append(observer)
    density_ex.run_commandline()


if __name__ == '__main__':
    main()
