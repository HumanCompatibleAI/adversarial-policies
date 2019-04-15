import argparse
import itertools
import os

import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity

from modelfree.score_agent import score_ex


class ActivationDensityModeler(object):
    def __init__(self, traj_dataset_path):
        self.data_keys = ('obs', 'ff_policy', 'ff_value')
        self.data = self._load_data(traj_dataset_path)
        self.models = {k: KernelDensity() for k in self.data_keys}
        self.pcas = {k: None for k in self.data_keys}

    def _load_data(self, traj_dataset_path):
        """
        Load data from data saved with TrajectoryRecorder with use_gail_format=False
        Note: only works with policies in which there is exactly one hidden layer
        """
        traj_data = np.load(traj_dataset_path)
        return {k: np.concatenate(traj_data[k].tolist()) for k in self.data_keys}

    def get_density_model(self, data_key, pca_dim=None):
        data = self.data[data_key]
        if pca_dim is not None:
            pca = PCA(n_components=pca_dim)
            data = pca.fit_transform(data)
            self.pcas[data_key] = pca
        self.models[data_key].fit(data)
        return self.models[data_key].get_params()

    def score_samples(self, data_key, samples):
        if self.pcas[data_key] is not None:
            samples = self.pcas[data_key].transform(samples)
        return self.models[data_key].score_samples(samples)

    def score(self, data_key, samples):
        if self.pcas[data_key] is not None:
            samples = self.pcas[data_key].transform(samples)
        return self.models[data_key].score(samples)

    def get_data(self, data_key):
        return self.data[data_key]


def kick_and_defend_ex(pca_dim, episodes, skip_scoring):
    base_config = dict(transparent_params={'ff_policy': False, 'ff_value': False},
                       record_traj_params={'agent_indices': 0},
                       env_name='multicomp/KickAndDefend-v0',
                       num_env=8, record_traj=True, episodes=episodes, render=False)
    dir_names = ('kad-adv', 'kad-rand', 'kad-zoo')
    paths = ('kad1', None, '1')
    types = ('ppo2', 'random', 'zoo')
    if not skip_scoring:
        for dir_name, path, agent_type in zip(dir_names, paths, types):
            config_copy = base_config.copy()
            config_copy['record_traj_params']['save_dir'] = f'data/{dir_name}'
            config_copy.update({'agent_b_type': agent_type, 'agent_b_path': path})
            run = score_ex.run(config_updates=config_copy)
            assert run.status == 'COMPLETED'

    path_str = 'data/{}/agent_0.npz'
    density_modelers = [ActivationDensityModeler(path_str.format(s)) for s in dir_names]
    for i, modeler in enumerate(density_modelers):
        modeler.get_density_model('ff_policy', pca_dim=pca_dim)
        modeler.get_density_model('ff_value', pca_dim=pca_dim)
        print(f'fit model {i}')

    results = np.zeros((2, 3, 3))
    keys = ('ff_policy', 'ff_value')

    os.makedirs('data/densities', exist_ok=True)
    for i, j, k in itertools.product(range(2), range(3), range(3)):
        fit_model = density_modelers[j]
        data_model = density_modelers[k]
        samples = data_model.get_data(keys[i])
        score = fit_model.score(keys[i], samples)
        individual_score = fit_model.score_samples(keys[i], samples)
        np.save(f'data/densities/{i}-{j}-{k}', individual_score)
        results[i, j, k] = score
        print(i, j, k, score)
    print(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pca-dim', type=int, default=None)
    parser.add_argument('--episodes', type=int, default=200)
    parser.add_argument('--skip-scoring', action='store_true')
    args = parser.parse_args()
    kick_and_defend_ex(args.pca_dim, args.episodes, args.skip_scoring)

