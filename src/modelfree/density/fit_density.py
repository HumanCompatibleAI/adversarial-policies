"""Fits density model to activations of victim's policy network."""

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
    """Generates experiment name from model class and parameters.

    :param model_class: (type) the class, one of GaussianMixture, PCAPreDensity or KernelDensity.
    :param model_kwargs: (dict) constructor arguments to the class.
    :return A string succinctly encoding the class and parameters."""
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
    """Performs PCA dimensionality reduction before density modelling with density_class."""
    def __init__(self, density_class, num_components, **kwargs):
        """Inits a PCAPreDensity object.

        :param density_class: (type) A class for the density model.
        :param pca_components: (int) Number of PCA components.
        :param kwargs: (dict) Additional keyword arguments passed-through to density_class.
        """
        super(PCAPreDensity, self).__init__()
        self.density_class = density_class
        self.num_components = num_components
        self.kwargs = kwargs
        self.density_obj = self.density_class(**self.kwargs)
        self.pca_obj = PCA(n_components=self.num_components)

    def fit(self, X):
        """Fits PCA transform on X, and then fits wrapped density model on reduced data."""
        reduced_representation = self.pca_obj.fit_transform(X)
        self.density_obj.fit(reduced_representation)

    def score_samples(self, X):
        """Performs PCA transformation on X, and then scores samples using wraped density model."""
        reduced_test_representation = self.pca_obj.transform(X)
        return self.density_obj.score_samples(reduced_test_representation)


@fit_model_ex.config
def base_config():
    ray_server = None  # by default will launch a server
    activation_dir = None  # directory of generated activatoins
    output_root = None  # directory to write output
    data_type = 'ff_policy'  # key into activations
    max_timesteps = None  # if specified, maximum number of timesteps of activations to use
    seed = 0
    model_class = GaussianMixture  # density model to use
    model_kwargs = {'n_components': 10}  # parameters for density model
    train_opponent = 'zoo_1'  # opponent ID to use for fitting density model (extracted from path)
    train_percentage = 0.5  # percentage of data to use for training (remainder is validation)
    _ = locals()  # quieten flake8 unused variable warning
    del _


@fit_model_ex.named_config
def debug_config():
    max_timesteps = 1000
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


def _load_and_reshape_single_file(np_path, col, opponent_id):
    """Loads data from np_path, extracting column col.

    :param np_path: (str) a path to a pickled dict of NumPy arrays.
    :param col: (str) the key to extract from the dict.
    :param opponent_id: (str) an identifier, added to the metadata.
    :return A tuple (concatenated_data, metadata_df) where concatenated_data is a flattened
            array of activations at different timesteps, and metadata_df is a pandas DataFrame
            containing the episode_id, timestep, frac_timestep and opponent_id.
    """
    traj_data = np.load(np_path, allow_pickle=True)
    episode_list = traj_data[col].tolist()

    episode_lengths = [len(episode) for episode in episode_list]
    episode_id = []
    timesteps = []
    frac_timesteps = []
    for i, episode_length in enumerate(episode_lengths):
        episode_id += [i] * episode_length
        episode_timesteps = list(range(episode_length))
        timesteps += episode_timesteps
        frac_timesteps += [el / episode_length for el in episode_timesteps]

    concatenated_data = np.concatenate(episode_list)
    opponent_id = [opponent_id] * len(concatenated_data)

    # TODO: only opponent_id is used in this code. Are the others worth keeping around?
    metadata_df = pd.DataFrame({'episode_id': episode_id,
                                'timesteps': timesteps,
                                'frac_timesteps': frac_timesteps,
                                'opponent_id': opponent_id})
    return concatenated_data, metadata_df


@ray.remote
def density_fitter(activation_paths, output_dir,
                   model_class, model_kwargs,
                   max_timesteps, data_key, train_opponent, train_frac):
    """Fits a density model with data from activation_paths, saving results to output_dir.

    :param activation_paths: (dict) a dictionary mapping from opponent ID to a path
                                    to a pickle file containing activations.
    :param output_dir: (str) path to an output directory.
    :param model_class: (type) a class to use for density modelling.
    :param model_kwargs: (dict) parameters for the model.
    :param max_timesteps: (int or None) the maximum number of timesteps of data to use.
    :param data_key: (str) key into the activation file, specifying e.g. the layer.
    :param train_opponent: (str) the opponent ID to fit the model to.
    :param train_frac: (float) the proportion of data to use for training (remainder is test set).
    :return A dictionary of metrics."""
    logger.info(f"Starting density fitting, saving to {output_dir}")

    # Load data
    activations = []
    metadata = []
    for opponent_type, path in activation_paths.items():
        logger.debug(f"Loaded data for {opponent_type} from {path}")
        act, meta = _load_and_reshape_single_file(path, opponent_type, data_key)
        activations.append(act)
        metadata.append(meta)
    activations = np.concatenate(activations)
    metadata = pd.concat(metadata)
    # Flatten activations (but preserve timestep)
    activations = activations.reshape(activations.shape[0], -1)

    # Optionally, sub-sample
    if max_timesteps is None:
        max_timesteps = len(metadata)
    activations = activations[0:max_timesteps]
    meta = metadata[0:max_timesteps]

    # Split into train, validation and test
    opponent_mask = meta['opponent_id'] == train_opponent
    percentage_mask = np.random.choice([True, False], size=len(meta),
                                       p=[train_frac, 1 - train_frac])

    meta['is_train'] = opponent_mask & percentage_mask
    train_data = activations[meta['is_train']]

    train_opponent_validation_mask = opponent_mask & ~percentage_mask
    train_opponent_validation_data = activations[train_opponent_validation_mask]

    # Fit model and evaluate
    model_obj = model_class(**model_kwargs)
    model_obj.fit(train_data)
    meta['log_proba'] = model_obj.score_samples(activations)

    metrics = {}
    if model_class == GaussianMixture:
        metrics = {
            'n_components': model_kwargs.get('n_components', 1),
            'covariance_type': model_kwargs.get('covariance_type', 'full'),
            'train_bic': model_obj.bic(train_data),
            'validation_bic': model_obj.bic(train_opponent_validation_data),
            'train_log_likelihood': model_obj.score(train_data),
            'validation_log_likelihood': model_obj.score(train_opponent_validation_data),
        }
        with open(os.path.join(output_dir, "metrics.json"), 'w') as f:
            json.dump(metrics, f)

    # Save activations, metadata and model weights
    meta_path = os.path.join(output_dir, 'train_metadata.csv')
    meta.to_csv(meta_path, index=False)

    # TODO: Do we need to store this? We already have activations stored elsewhere on disk.
    activations_path = os.path.join(output_dir, 'activations.npy')
    np.save(activations, activations_path)

    weights_path = os.path.join(output_dir, f'fitted_{model_class.__name__}_model.pkl')
    with open(weights_path, "wb") as f:
        pickle.dump(model_obj, f)

    logger.info(
        f"Completed fitting of {model_class.__name__} model with args {model_kwargs}, "
        f"saved to {output_dir}")

    return metrics


@fit_model_ex.main
def fit_model(_run, ray_server, activation_dir, output_root, max_timesteps, data_type,
              model_class, model_kwargs, train_opponent, train_percentage):
    """Fits density models for each environment and victim type in activation_dir,
       saving resulting models to output_root. Works by repeatedly calling `density_fitter`,
       running in parallel via Ray."""
    ray.init(redis_address=ray_server)

    # Find activation paths for each environment & victim-path tuple
    stem_pattern = re.compile(r'(.*)_opponent_.*\.npz')
    opponent_pattern = re.compile(r'.*_opponent_([^\s]+)+\.npz')
    # activation_paths is indexed by [env_victim][opponent_type] where env_victim is
    # e.g. 'SumoHumans-v0_victim_zoo_1' and opponent_type is e.g. 'ppo2_1'.
    activation_paths = {}

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
                                       max_timesteps, data_type, train_opponent,
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
