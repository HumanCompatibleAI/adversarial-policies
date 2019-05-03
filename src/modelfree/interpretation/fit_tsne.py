import logging
import os
import os.path as osp
import pickle
import re
import tempfile

import numpy as np
import pandas as pd
import sacred
from sacred.observers import FileStorageObserver
from sklearn.manifold import TSNE

fit_tsne_ex = sacred.Experiment('fit_tsne')
logger = logging.getLogger('modelfree.interpretation.fit_tsne')


@fit_tsne_ex.config
def base_config():
    activation_dir = 'data/tsne/debug/20190502_203906'  # TODO: cross-platform
    output_root = None
    data_type = 'ff_policy'
    num_components = 2
    num_observations = 1000
    seed = 0
    perplexity = 5
    _ = locals()  # quieten flake8 unused variable warning
    del _


@fit_tsne_ex.named_config
def full_model():
    num_observations = None
    _ = locals()  # quieten flake8 unused variable warning
    del _


@fit_tsne_ex.capture
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


def fit_tsne_helper(activation_paths, output_dir, num_components, num_observations, perplexity):
    all_file_data = []
    all_metadata = []
    for opponent_type, path in activation_paths.items():
        logger.debug(f"Loaded data for {opponent_type} from {path}")
        file_data, metadata = _load_and_reshape_single_file(path, opponent_type)
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
    fit_tsne_ex.add_artifact(metadata_path)

    # Fit t-SNE
    tsne_obj = TSNE(n_components=num_components, verbose=1, perplexity=perplexity)
    logger.info(f"Starting T-SNE fitting, saving to {output_dir}")
    tsne_ids = tsne_obj.fit_transform(sub_data)
    logger.info(f"Completed T-SNE fitting, saved to ")

    # Save weights
    tsne_weights_path = os.path.join(output_dir, 'tsne_weights.pkl')
    with open(tsne_weights_path, "wb") as fp:
        pickle.dump(tsne_obj, fp)
    fit_tsne_ex.add_artifact(tsne_weights_path)

    # Save cluster IDs
    cluster_ids_path = os.path.join(output_dir, 'cluster_ids.npy')
    np.save(cluster_ids_path, tsne_ids)
    fit_tsne_ex.add_artifact(cluster_ids_path)


@fit_tsne_ex.main
def fit_tsne(activation_dir, output_root,
             num_components, num_observations, perplexity):
    # Find activation paths for each environment & victim-path tuple
    stem_pattern = re.compile(r'(.*)_opponent_.*\.npz')
    opponent_pattern = re.compile(r'.*_opponent_([^\s]+)_[^\s]+\.npz')
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

    # Fit t-SNE and save model weights
    for stem, paths in activation_paths.items():
        output_dir = osp.join(output_root, stem)
        os.makedirs(output_dir)
        fit_tsne_helper(paths, output_dir, num_components, num_observations, perplexity)

    # Clean up temporary directory (if needed)
    if tmp_dir is not None:
        tmp_dir.clean()


def main():
    observer = FileStorageObserver.create(osp.join('data', 'sacred', 'fit_tsne'))
    fit_tsne_ex.observers.append(observer)
    fit_tsne_ex.run_commandline()


if __name__ == '__main__':
    main()
