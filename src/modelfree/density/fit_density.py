import json
import logging
import os
import os.path as osp
import pickle
import re
import tempfile

import numpy as np
import pandas as pd
import ray
import sacred
from sacred.observers import FileStorageObserver
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity

from modelfree.common import utils

fit_model_ex = sacred.Experiment('fit_density_model')
logger = logging.getLogger('modelfree.density.fit_density')


def gen_exp_name(model_class, model_kwargs):
    if model_class == GaussianMixture:
        n_components = model_kwargs.get('n_components', 1)
        covariance_type = model_kwargs.get('covariance_type', 'full')
        return f'gmm_{n_components}_components_{covariance_type}'
    elif model_class == PCAPreDensity:
        if model_kwargs['density_class'] == KernelDensity:
            return 'pca_kde'
        elif model_kwargs['density_class'] == 'pca_gmm':
            return 'pca_gmm'
        else:
            return 'pca_unknown'
    elif model_class == KernelDensity:
        return 'kde'
    else:
        return 'default'


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


@fit_model_ex.config
def base_config():
    ray_server = None  # by default will launch a server
    activation_dir = None
    output_root = None
    data_type = 'ff_policy'
    num_observations = None
    seed = 0
    model_class = GaussianMixture
    model_kwargs = {'n_components': 10}
    train_opponent = 'zoo_1'
    train_percentage = 0.5
    _ = locals()  # quieten flake8 unused variable warning
    del _


@fit_model_ex.named_config
def debug_config():
    num_observations = 1000
    model_class = KernelDensity
    _ = locals()  # quieten flake8 unused variable warning
    del _


@fit_model_ex.named_config
def gmm():
    model_class = GaussianMixture
    _ = locals()  # quieten flake8 unused variable warning
    del _


@fit_model_ex.named_config
def pca_kde():
    model_class = PCAPreDensity
    model_kwargs = {'density_class': KernelDensity}
    _ = locals()  # quieten flake8 unused variable warning
    del _


@fit_model_ex.named_config
def pca_gmm():
    model_class = PCAPreDensity
    model_kwargs = {'density_class': GaussianMixture}
    _ = locals()  # quieten flake8 unused variable warning
    del _


@fit_model_ex.named_config
def kde():
    model_class = KernelDensity
    _ = locals()  # quieten flake8 unused variable warning
    del _


def _load_and_reshape_single_file(np_path, opponent_type, data_type):
    traj_data = np.load(np_path, allow_pickle=True)
    episode_list = traj_data[data_type].tolist()
    episode_lengths = [len(episode) for episode in episode_list]
    episode_id = []
    observation_index = []
    relative_observation_index = []
    for i, episode_length in enumerate(episode_lengths):
        episode_id += [i] * episode_length
        episode_observation_ids = list(range(episode_length))
        observation_index += episode_observation_ids
        relative_observation_index += [el / episode_length for el in episode_observation_ids]

    concatenated_data = np.concatenate(episode_list)
    opponent_type = [opponent_type] * len(concatenated_data)

    metadata_df = pd.DataFrame({'episode_id': episode_id,
                                'observation_index': observation_index,
                                'relative_observation_index': relative_observation_index,
                                'opponent_id': opponent_type})
    return concatenated_data, metadata_df


@ray.remote
def density_fitter(activation_paths, output_dir,
                   model_class, model_kwargs,
                   num_observations, data_type, train_opponent, train_percentage):

    logger.info(f"Starting density fitting, saving to {output_dir}")

    all_file_data = []
    all_metadata = []
    for opponent_type, path in activation_paths.items():
        logger.debug(f"Loaded data for {opponent_type} from {path}")
        file_data, metadata = _load_and_reshape_single_file(path, opponent_type, data_type)
        all_file_data.append(file_data)
        all_metadata.append(metadata)

    merged_file_data = np.concatenate(all_file_data)
    merged_metadata = pd.concat(all_metadata)

    # Optionally, sub-sample
    if num_observations is None:
        num_observations = len(merged_metadata)
    sub_data = merged_file_data[0:num_observations].reshape(num_observations, 128)
    sub_meta = merged_metadata[0:num_observations]
    # Save metadata

    train_meta_path = os.path.join(output_dir, 'train_metadata.csv')
    test_meta_path = os.path.join(output_dir, 'test_metadata.csv')

    train_data_path = os.path.join(output_dir, 'train_data.npz')
    test_data_path = os.path.join(output_dir, 'test_data.npz')

    opponent_mask = sub_meta['opponent_id'] == train_opponent
    percentage_mask = np.random.choice([True, False], size=len(sub_meta),
                                       p=[train_percentage, 1 - train_percentage])

    train_mask = opponent_mask & percentage_mask
    train_opponent_validation_mask = opponent_mask & ~percentage_mask
    test_mask = ~train_mask

    train_meta = sub_meta[train_mask]
    test_meta = sub_meta[test_mask]

    train_data = sub_data[train_mask]
    test_data = sub_data[test_mask]
    same_opponent_test = sub_data[train_opponent_validation_mask]

    model_obj = model_class(**model_kwargs)
    model_obj.fit(train_data)

    train_probas = model_obj.score_samples(train_data)
    test_probas = model_obj.score_samples(test_data)
    train_meta['log_proba'] = train_probas
    test_meta['log_proba'] = test_probas

    if model_class.__name__ == 'GaussianMixture':
        train_bic = model_obj.bic(train_data)
        validation_bic = model_obj.bic(same_opponent_test)
        train_log_likelihood = model_obj.score(train_data)
        validation_log_likelihood = model_obj.score(same_opponent_test)
        metrics = {
            'n_components': model_kwargs.get('n_components', 1),
            'covariance_type': model_kwargs.get('covariance_type', 'full'),
            'train_bic': train_bic,
            'validation_bic': validation_bic,
            'train_log_likelihood': train_log_likelihood,
            'validation_log_likelihood': validation_log_likelihood
        }
        with open(os.path.join(output_dir, "metrics.json"), 'w') as fp:
            json.dump(metrics, fp)

    # Save results out

    train_meta.to_csv(train_meta_path, index=False)
    test_meta.to_csv(test_meta_path, index=False)

    np.save(train_data_path, train_data)
    np.save(test_data_path, test_data)

    # Save weights
    weights_path = os.path.join(output_dir, f'fitted_{model_class.__name__}_model.pkl')
    with open(weights_path, "wb") as fp:
        pickle.dump(model_obj, fp)

    # Save cluster IDs

    logger.info(
        f"Completed fitting of {model_class.__name__} model with args {model_kwargs}, "
        f"saved to {output_dir}")

    return metrics


@fit_model_ex.main
def fit_model(_run, ray_server, activation_dir, output_root, num_observations, data_type,
              model_class, model_kwargs, train_opponent, train_percentage):
    ray.init(redis_address=ray_server)

    # Find activation paths for each environment & victim-path tuple
    stem_pattern = re.compile(r'(.*)_opponent_.*\.npz')
    opponent_pattern = re.compile(r'.*_opponent_([^\s]+)+\.npz')
    activation_paths = {}

    #
    for fname in os.listdir(activation_dir):
        stem_match = stem_pattern.match(fname)
        if stem_match is None:
            logger.debug(f"Skipping {fname}")
            continue
        stem = stem_match.groups()[0]

        opponent_match = opponent_pattern.match(fname)
        opponent_type = opponent_match.groups()[0]

        path = osp.join(activation_dir, fname)
        activation_paths.setdefault(stem, {})[opponent_type] = path

    # Create temporary output directory (if needed)
    tmp_dir = None
    if output_root is None:
        tmp_dir = tempfile.TemporaryDirectory()
        output_root = tmp_dir.name

    # Fit density model and save weights
    results = []
    for stem, paths in activation_paths.items():
        output_dir = osp.join(output_root, stem)
        os.makedirs(output_dir)
        future = density_fitter.remote(paths, output_dir, model_class, model_kwargs,
                                       num_observations, data_type, train_opponent,
                                       train_percentage)
        results.append(future)

    ray.get(results)  # block until all jobs have finished
    utils.add_artifacts(_run, output_root, ingredient=fit_model_ex)

    # Clean up temporary directory (if needed)
    if tmp_dir is not None:
        tmp_dir.clean()

    ray.shutdown()


def main():
    observer = FileStorageObserver.create(osp.join('data', 'sacred', 'density_fit'))
    fit_model_ex.observers.append(observer)
    fit_model_ex.run_commandline()


if __name__ == '__main__':
    main()
