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
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity

from modelfree.common import utils

fit_model_ex = sacred.Experiment('fit_density_model')
logger = logging.getLogger('modelfree.density.fit_density')

ALLOWED_MODEL_LOOKUP = {
    'KDE': KernelDensity,
    'GMM': GaussianMixture
}


@fit_model_ex.config
def base_config():
    ray_server = None  # by default will launch a server
    activation_dir = None
    output_root = None
    data_type = 'ff_policy'
    num_observations = None
    seed = 0
    perplexity = 250
    model_type = None
    model_kwargs = dict()
    _ = locals()  # quieten flake8 unused variable warning
    del _


@fit_model_ex.named_config
def debug_config():
    num_observations = 1000
    model_type = 'KDE'
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
                   model_type, model_kwargs,
                   num_observations, data_type):

    if model_type not in ALLOWED_MODEL_LOOKUP:
        raise ValueError(f"Selected model type {model_type} not supported")

    logger.info(f"Starting T-SNE fitting, saving to {output_dir}")

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

    # Save metadata
    metadata_path = os.path.join(output_dir, 'metadata.csv')
    merged_metadata[0:num_observations].to_csv(metadata_path)

    model_class = ALLOWED_MODEL_LOOKUP[model_type]
    model_obj = model_class(**model_kwargs)
    model_obj.fit(sub_data)
    log_probas = model_obj.score_samples(sub_data)

    # Save weights
    weights_path = os.path.join(output_dir, f'fitted_{model_type}_model.pkl')
    with open(weights_path, "wb") as fp:
        pickle.dump(model_obj, fp)

    # Save cluster IDs
    cluster_ids_path = os.path.join(output_dir, 'log_probas.npy')
    np.save(cluster_ids_path, log_probas)

    logger.info(
        f"Completed fitting of {model_type} model with args {model_kwargs}, saved to {output_dir}")


@fit_model_ex.main
def fit_model(_run, ray_server, activation_dir, output_root, num_observations, data_type,
              model_type, model_kwargs):
    ray.init(redis_address=ray_server)

    # Find activation paths for each environment & victim-path tuple
    stem_pattern = re.compile(r'(.*)_opponent_.*\.npz')
    opponent_pattern = re.compile(r'.*_opponent_([^\s]+)_[^\s]+\.npz')
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

    # Fit t-SNE and save model weights
    results = []
    for stem, paths in activation_paths.items():
        output_dir = osp.join(output_root, stem)
        os.makedirs(output_dir)
        future = density_fitter.remote(paths, output_dir, model_type, model_kwargs,
                                       num_observations, data_type)
        results.append(future)

    ray.get(results)  # block until all jobs have finished
    utils.add_artifacts(_run, output_root, ingredient=fit_model_ex)

    # Clean up temporary directory (if needed)
    if tmp_dir is not None:
        tmp_dir.clean()

    ray.shutdown()


def main():
    observer = FileStorageObserver.create(osp.join('data', 'sacred', 'tsne_fit'))
    fit_model_ex.observers.append(observer)
    fit_model_ex.run_commandline()


if __name__ == '__main__':
    main()
