import logging
import os
import pickle
import tempfile

import numpy as np
import pandas as pd
import sacred
from sacred.observers import FileStorageObserver
from sklearn.manifold import TSNE
from swissarmy import logger

tsne_experiment = sacred.Experiment('tsne-base-experiment')
tsne_experiment.observers.append(FileStorageObserver.create(
    '/Users/cody/Data/adversarial_policies/tsne_runs'))
logger_obj = logger.get_logger_object(cl_level=logging.DEBUG)


@tsne_experiment.config
def base_config():
    relative_dirs = ['adversary', 'random', 'zoo']
    base_path = "/Users/cody/Data/adversarial_policies/tsne_save_activations"
    data_type = 'ff_policy'
    num_components = 2
    sacred_dir_ids = None  # otherwise list
    np_file_name = "victim_activations.npz"
    num_observations = 1000
    seed = 0
    perplexity = 5
    _ = locals()  # quieten flake8 unused variable warning
    del _

@tsne_experiment.named_config
def full_model():
    num_observations = None
    _ = locals()  # quieten flake8 unused variable warning
    del _

def _get_latest_sacred_dirs(base_path, rd_list):
    latest_dir_ids = []
    for rd in rd_list:
        sacred_dirs = os.listdir(os.path.join(base_path, rd))
        max_int_dir = 0
        for sd in sacred_dirs:
            try:
                int_dir = int(sd)
                if int_dir > max_int_dir:
                    max_int_dir = int_dir
            except ValueError:
                continue
        latest_dir_ids.append(str(max_int_dir))
    return latest_dir_ids


@tsne_experiment.capture
def _load_and_reshape_single_file(relative_dir, base_path, data_type, np_file_name, sacred_dir):
    traj_data = np.load(os.path.join(base_path, relative_dir, sacred_dir, np_file_name), allow_pickle=True)
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

    concated_data = np.concatenate(episode_list)
    opponent_id = [relative_dir] * len(concated_data)

    metadata_df = pd.DataFrame({'episode_id': episode_id,
                                'observation_index': observation_index,
                                'relative_observation_index': relative_observation_index,
                                'opponent_id': opponent_id})
    return concated_data, metadata_df


@tsne_experiment.automain
def experiment_main(relative_dirs, num_components, base_path, sacred_dir_ids,
                    num_observations, perplexity):
    all_file_data = []
    all_metadata = []
    if sacred_dir_ids is None:
        sacred_dir_ids = _get_latest_sacred_dirs(base_path, relative_dirs)

    for i, rd in enumerate(relative_dirs):
        print("Pulling data out of {}, with sacred run ID {}".format(rd, sacred_dir_ids[i]))
        file_data, metadata = _load_and_reshape_single_file(rd, sacred_dir=sacred_dir_ids[i])
        all_file_data.append(file_data)
        all_metadata.append(metadata)

    merged_file_data = np.concatenate(all_file_data)
    merged_metadata = pd.concat(all_metadata)
    if num_observations is None:
        num_observations = len(merged_metadata)
    with tempfile.TemporaryDirectory() as dirname:
        metadata_path = os.path.join(dirname, 'metadata.csv')
        merged_metadata[0:num_observations].to_csv(metadata_path)
        tsne_experiment.add_artifact(metadata_path)

        tsne_obj = TSNE(n_components=num_components, verbose=1, perplexity=perplexity)
        logger_obj.debug("Starting T-SNE fitting")
        tsne_ids = tsne_obj.fit_transform(merged_file_data[0:num_observations])
        logger_obj.debug("Completed T-SNE fitting")
        print(tsne_ids.shape)
        tsne_weights_path = os.path.join(dirname, 'saved_tsne_weights.pkl')
        with open(tsne_weights_path, "wb") as fp:
            pickle.dump(tsne_obj, fp)
        tsne_experiment.add_artifact(tsne_weights_path)

        cluster_ids_path = os.path.join(dirname, 'cluster_ids.npy')
        np.save(cluster_ids_path, tsne_ids)
        tsne_experiment.add_artifact(cluster_ids_path)
