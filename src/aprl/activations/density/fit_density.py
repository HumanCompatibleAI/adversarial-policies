"""Fits density model to activations of victim's policy network."""

import glob
import json
import logging
import os
import os.path as osp
import pickle
import re
import tempfile
from typing import Any, Dict

import numpy as np
import pandas as pd
import ray
import sacred
from sacred.observers import FileStorageObserver
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity

from aprl.common import utils

fit_model_ex = sacred.Experiment("fit_density_model")
logger = logging.getLogger("aprl.density.fit_density")


def gen_exp_name(model_class, model_kwargs):
    """Generates experiment name from model class and parameters.

    :param model_class: (type) the class, one of GaussianMixture, PCAPreDensity or KernelDensity.
    :param model_kwargs: (dict) constructor arguments to the class.
    :return A string succinctly encoding the class and parameters."""
    if model_class == GaussianMixture:
        n_components = model_kwargs.get("n_components", 1)
        covariance_type = model_kwargs.get("covariance_type", "full")
        return f"gmm_{n_components}_components_{covariance_type}"
    elif model_class == PCAPreDensity:
        if model_kwargs["density_class"] == KernelDensity:
            return "pca_kde"
        elif model_kwargs["density_class"] == GaussianMixture:
            return "pca_gmm"
        else:
            return "pca_unknown"
    elif model_class == KernelDensity:
        return "kde"
    else:
        return "default"


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
        """Performs PCA transformation on X, then scores samples using wrapped density model."""
        reduced_test_representation = self.pca_obj.transform(X)
        return self.density_obj.score_samples(reduced_test_representation)


@fit_model_ex.config
def base_config():
    ray_server = None  # by default will launch a server
    activation_glob = None  # directory of generated activations
    output_root = None  # directory to write output
    data_type = "ff_policy"  # key into activations
    max_timesteps = None  # if specified, maximum number of timesteps of activations to use
    seed = 0
    model_class = GaussianMixture  # density model to use
    model_kwargs = {"n_components": 10}  # parameters for density model
    train_opponent = "zoo_1"  # opponent ID to use for fitting density model (extracted from path)
    train_percentage = 0.5  # percentage of data to use for training (remainder is validation)
    _ = locals()  # quieten flake8 unused variable warning
    del _


@fit_model_ex.named_config
def debug_config():
    max_timesteps = 100
    model_kwargs = {"n_components": 2}

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
    model_kwargs = {"density_class": KernelDensity}
    _ = locals()  # quieten flake8 unused variable warning
    del _


@fit_model_ex.named_config
def pca_gmm():
    model_class = PCAPreDensity
    model_kwargs = {"density_class": GaussianMixture}
    _ = locals()  # quieten flake8 unused variable warning
    del _


@fit_model_ex.named_config
def kde():
    model_class = KernelDensity
    _ = locals()  # quieten flake8 unused variable warning
    del _


def _load_and_reshape_single_file(np_path, data_key, opponent_id):
    """Loads data from np_path, extracting column col.

    :param np_path: (str) a path to a pickled dict of NumPy arrays.
    :param data_key: (str) the key to extract from the dict.
    :param opponent_id: (str) an identifier, added to the metadata.
    :return A tuple (concatenated_data, metadata_df) where concatenated_data is a flattened
            array of activations at different timesteps, and metadata_df is a pandas DataFrame
            containing the episode_id, timestep, frac_timestep and opponent_id.
    """
    traj_data = np.load(np_path, allow_pickle=True)
    episode_list = traj_data[data_key].tolist()

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

    # We currently just use opponent_id, but keep the others around for exploratory data analysis
    # (for example, spotting patterns in activations depending on position in episode)
    metadata_df = pd.DataFrame(
        {
            "episode_id": episode_id,
            "timesteps": timesteps,
            "frac_timesteps": frac_timesteps,
            "opponent_id": opponent_id,
        }
    )
    return concatenated_data, metadata_df


@ray.remote
def density_fitter(
    activation_paths,
    output_dir,
    model_class,
    model_kwargs,
    max_timesteps,
    data_key,
    train_opponent,
    train_frac,
):
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
    for opponent_id, path in activation_paths.items():
        logger.debug(f"Loaded data for {opponent_id} from {path}")
        act, meta = _load_and_reshape_single_file(path, data_key=data_key, opponent_id=opponent_id)
        activations.append(act)
        metadata.append(meta)
    activations = np.concatenate(activations)
    metadata = pd.concat(metadata)
    # Flatten activations (but preserve timestep)
    activations = activations.reshape(activations.shape[0], -1)

    # Sub-sample
    np.random.shuffle(activations)
    metadata = metadata.sample(frac=1)
    if max_timesteps is None:
        max_timesteps = len(metadata)
    activations = activations[0:max_timesteps]
    metadata = metadata[0:max_timesteps].copy()

    # Split into train, validation and test
    opponent_mask = metadata["opponent_id"] == train_opponent
    percentage_mask = np.random.choice(
        [True, False], size=len(metadata), p=[train_frac, 1 - train_frac]
    )

    metadata["is_train"] = opponent_mask & percentage_mask
    train_data = activations[metadata["is_train"]]

    train_opponent_validation_mask = opponent_mask & ~percentage_mask
    train_opponent_validation_data = activations[train_opponent_validation_mask]

    # Fit model and evaluate
    model_obj = model_class(**model_kwargs)
    model_obj.fit(train_data)
    metadata["log_proba"] = model_obj.score_samples(activations)

    metrics = {}
    if model_class == GaussianMixture:
        metrics = {
            "n_components": model_kwargs.get("n_components", 1),
            "covariance_type": model_kwargs.get("covariance_type", "full"),
            "train_bic": model_obj.bic(train_data),
            "validation_bic": model_obj.bic(train_opponent_validation_data),
            "train_log_likelihood": model_obj.score(train_data),
            "validation_log_likelihood": model_obj.score(train_opponent_validation_data),
        }
        with open(os.path.join(output_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f)

    # Save metadata and model weights
    meta_path = os.path.join(output_dir, "metadata.csv")
    metadata.to_csv(meta_path, index=False)

    weights_path = os.path.join(output_dir, f"fitted_{model_class.__name__}_model.pkl")
    with open(weights_path, "wb") as f:
        pickle.dump(model_obj, f)

    logger.info(
        f"Completed fitting of {model_class.__name__} model with args {model_kwargs}, "
        f"saved to {output_dir}"
    )

    return metrics


@fit_model_ex.main
def fit_model(
    _run,
    ray_server: str,
    init_kwargs: Dict[str, Any],
    activation_glob: str,
    output_root: str,
    max_timesteps: int,
    data_type,
    model_class,
    model_kwargs,
    train_opponent,
    train_percentage,
):
    """Fits density models for each environment and victim type in activation_dir,
    saving resulting models to output_root. Works by repeatedly calling `density_fitter`,
    running in parallel via Ray."""
    try:
        ray.init(address=ray_server, **init_kwargs)

        # Find activation paths for each environment & victim-path tuple
        stem_pattern = re.compile(r"(.*)_opponent_.*\.npz")
        opponent_pattern = re.compile(r".*_opponent_([^\s]+)+\.npz")
        # activation_paths is indexed by [env_victim][opponent_type] where env_victim is
        # e.g. 'SumoHumans-v0_victim_zoo_1' and opponent_type is e.g. 'ppo2_1'.
        activation_paths = {}

        for activation_path in glob.glob(activation_glob):
            activation_dir = os.path.basename(activation_path)
            stem_match = stem_pattern.match(activation_dir)
            if stem_match is None:
                logger.debug(f"Skipping {activation_path}")
                continue
            stem = stem_match.groups()[0]

            opponent_match = opponent_pattern.match(activation_dir)
            opponent_type = opponent_match.groups()[0]

            activation_paths.setdefault(stem, {})[opponent_type] = activation_path

        # Create temporary output directory (if needed)
        tmp_dir = None
        if output_root is None:
            tmp_dir = tempfile.TemporaryDirectory()
            output_root = tmp_dir.name
        else:
            exp_name = gen_exp_name(model_class, model_kwargs)
            output_root = os.path.join(output_root, exp_name)

        # Fit density model and save weights
        results = []
        for stem, paths in activation_paths.items():
            output_dir = osp.join(output_root, stem)
            os.makedirs(output_dir)
            future = density_fitter.remote(
                paths,
                output_dir,
                model_class,
                utils.sacred_copy(model_kwargs),
                max_timesteps,
                data_type,
                train_opponent,
                train_percentage,
            )
            results.append(future)

        ray.get(results)  # block until all jobs have finished
        utils.add_artifacts(_run, output_root, ingredient=fit_model_ex)
    finally:
        # Clean up temporary directory (if needed)
        if tmp_dir is not None:
            tmp_dir.cleanup()

        ray.shutdown()


def main():
    observer = FileStorageObserver(osp.join("data", "sacred", "density_fit"))
    fit_model_ex.observers.append(observer)
    fit_model_ex.run_commandline()


if __name__ == "__main__":
    main()
