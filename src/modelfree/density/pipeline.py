import logging
import os
import os.path as osp

import sacred
from sacred.observers import FileStorageObserver
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity

from modelfree.common import utils
from modelfree.common.generate_activations import generate_activations, generate_activations_ex
from modelfree.density.fit_density import fit_model, fit_model_ex

density_ex = sacred.Experiment('density', ingredients=[generate_activations_ex, fit_model_ex])
logger = logging.getLogger('modelfree.density.pipeline')


class PCAPreDensity(object):
    def __init__(self, density_class, pca_components, **kwargs):
        super(PCAPreDensity, self).__init__()
        self.density_class = density_class
        self.num_components = pca_components
        self.kwargs = kwargs
        self.density_obj = self.density_class(**self.kwargs)
        self.pca_obj = PCA(n_components=self.num_components)

    def fit(self, X):
        reduced_representation = self.pca_obj.fit_transform(X)
        self.density_obj.fit(reduced_representation)

    def score_samples(self, X):
        reduced_test_representation = self.pca_obj.transform(X)
        return self.density_obj.score_samples(reduced_test_representation)

# Questions: How do you set named_configs for ingredients?

def _exp_name(fit_density_model):
    if fit_density_model['model_class'] == GaussianMixture:
        n_components = fit_density_model['model_kwargs'].get('n_components', 1)
        covariance_type = fit_density_model['model_kwargs'].get('covariance_type', 'full')
        return f'gmm_{n_components}_components_{covariance_type}'
    else:
        return 'default'


@density_ex.config
def main_config(generate_activations, fit_density_model):
    output_root = osp.join('data', 'density')                  # where to produce output
    exp_name = _exp_name(fit_density_model)                    # experiment name
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


@density_ex.named_config
def kde(fit_density_model):
    fit_density_model = dict(fit_density_model)
    fit_density_model['model_class'] = KernelDensity
    exp_name = "kdd"
    _ = locals()  # quieten flake8 unused variable warning
    del _


@density_ex.named_config
def gmm(fit_density_model, generate_activations):
    fit_density_model = dict(fit_density_model)
    fit_density_model['model_class'] = GaussianMixture
    generate_activations = dict(generate_activations)
    generate_activations['score_update'] = {'timesteps': 40000}  # number of timesteps to save activations for
    _ = locals()  # quieten flake8 unused variable warning
    del _


@density_ex.named_config
def pca_kde(fit_density_model):
    fit_density_model = dict(fit_density_model)
    fit_density_model['model_class'] = PCAPreDensity
    fit_density_model['model_kwargs'] = {'density_class': KernelDensity}


@density_ex.named_config
def pca_gmm(fit_density_model):
    fit_density_model = dict(fit_density_model)
    fit_density_model['model_class'] = PCAPreDensity
    fit_density_model['model_kwargs'] = {'density_class': GaussianMixture}


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
