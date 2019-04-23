import os
import sacred
import pdb
import pickle
import logging
import tempfile

import numpy as np
import pandas as pd
import logging

from sklearn.manifold import TSNE
from sacred.observers import FileStorageObserver
from swissarmy import logger


tsne_experiment = sacred.Experiment('tsne-base-experiment')
tsne_experiment.observers.append(FileStorageObserver.create('/Users/cody/Data/adversarial_policies/tsne_runs'))
logger_obj = logger.get_logger_object(cl_level=logging.DEBUG)

opponent_lookup = {
    'kad-adv': 'Adversarial',
    'kad-rand': 'Random',
    'kad-zoo': 'Zoo'
}
@tsne_experiment.config
def base_config():
    relative_dirs = ['kad-adv', 'kad-rand', 'kad-zoo']
    base_path = "/Users/cody/Data/adversarial_policies"
    data_type = 'ff_policy'
    num_components = 2
    # TODO Try with and without PCA before
    # TODO Try with different numbers of components

@tsne_experiment.capture
def _load_and_reshape_single_file(relative_dir, base_path, data_type):
    traj_data = np.load(os.path.join(base_path, relative_dir, "agent_0.npz"))
    episode_list = traj_data[data_type].tolist()
    episode_lengths = [len(episode) for episode in episode_list]
    episode_id = []
    observation_index = []
    relative_observation_index = []
    for i, episode_length in enumerate(episode_lengths):
        episode_id += [i]*episode_length
        episode_observation_ids = list(range(episode_length))
        observation_index += episode_observation_ids
        relative_observation_index += [el/episode_length for el in episode_observation_ids]

    concated_data = np.concatenate(episode_list)
    opponent_id = [opponent_lookup[relative_dir]]*len(concated_data)

    metadata_df = pd.DataFrame({'episode_id': episode_id,
                                'observation_index': observation_index,
                                'relative_observation_index': relative_observation_index,
                                'opponent_id': opponent_id})
    return concated_data, metadata_df

@tsne_experiment.automain
def experiment_main(relative_dirs, num_components):
    all_file_data = []
    all_metadata = []
    for rd in relative_dirs:
        file_data, metadata = _load_and_reshape_single_file(rd)
        all_file_data.append(file_data)
        all_metadata.append(metadata)

    merged_file_data = np.concatenate(all_file_data)
    merged_metadata = pd.concat(all_metadata)

    with tempfile.TemporaryDirectory() as dirname:
        metadata_path = os.path.join(dirname, 'metadata.csv')
        merged_metadata.to_csv(metadata_path)
        tsne_experiment.add_artifact(metadata_path)

        tsne_obj = TSNE(n_components=num_components)
        logger_obj.debug("Starting T-SNE fitting")
        tsne_ids = tsne_obj.fit_transform(merged_file_data)
        logger_obj.debug("Completed T-SNE fitting")
        print(tsne_ids.shape)
        tsne_weights_path = os.path.join(dirname, 'saved_tsne_weights.pkl')
        with open(tsne_weights_path, "wb") as fp:
            pickle.dump(tsne_obj, fp)
        tsne_experiment.add_artifact(tsne_weights_path)

        cluster_ids_path = os.path.join(dirname, 'cluster_ids.npy')
        np.save(cluster_ids_path, tsne_ids)
        tsne_experiment.add_artifact(cluster_ids_path)






