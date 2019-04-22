import os
import sacred
import pdb
import pickle
import logging

import numpy as np
import pandas as pd
import logging

from sklearn.manifold import TSNE
from sacred.observers import FileStorageObserver
from swissarmy import logger


tsne_experiment = sacred.Experiment('tsne-base-experiment')
tsne_experiment.observers.append(FileStorageObserver.create('/Users/cody/Data/adversarial_policies/tsne_runs'))
logger_obj = logger.get_logger_object(cl_level=logging.DEBUG)

@tsne_experiment.config
def base_config():
    relative_dirs = ['kad-adv', 'kad-rand', 'kad-zoo']
    base_path = "/Users/cody/Data/adversarial_policies"
    data_type = 'ff_policy'

@tsne_experiment.capture
def _load_and_reshape_single_file(relative_dir, base_path, data_type):
    traj_data = np.load(os.path.join(base_path, relative_dir, "agent_0.npz"))
    concated_data = np.concatenate(traj_data[data_type].tolist())
    # Metadata:
    # Opponent type
    # Absolute Timestep
    # Percentage Timestep
    #
    ## TODO: do clever things here around metadata
    return concated_data

@tsne_experiment.automain
def experiment_main(relative_dirs):
    all_file_data = []
    for rd in relative_dirs:
        all_file_data.append(_load_and_reshape_single_file(rd))

    merged_file_data = np.concatenate(all_file_data)

    tsne_obj = TSNE(n_components=2)
    logger_obj.debug("Starting T-SNE fitting")
    tsne_obj.fit(merged_file_data)
    logger_obj.debug("Completed T-SNE fitting")
    f_name = "saved_tsne_weights.pkl"
    with open(f_name, "wb") as fp:
        pickle.dump(tsne_obj, fp)
    tsne_experiment.add_artifact(f_name)

