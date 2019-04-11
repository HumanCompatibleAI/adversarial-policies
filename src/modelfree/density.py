import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity

from modelfree.score_agent import score_ex


class ActivationDensityModeler(object):
    def __init__(self, traj_dataset_path):
        self.data_keys = ('obs', 'ff_policy', 'ff_value')
        self.data = self._load_data(traj_dataset_path)
        self.model = KernelDensity()

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
        self.model.fit(data)
        return self.model.get_params()

    def score_samples(self, samples):
        return self.model.score_samples(samples)


def kick_and_defend_ex():
    base_config = dict(transparent_params={'ff_policy': False, 'ff_value': False},
                       record_traj_params={'agent_indices': 0},
                       num_env=8, record_traj=True, episodes=500, render=False)
    dir_names = ('kad-zoo', 'kad-rand', 'kad-adv')
    # TODO: get path for kad-adv
    paths = ('1', None, None)
    types = ('zoo', 'random', 'ppo2')
    for dir_name, path, agent_type in zip(dir_names, paths, types):
        config_copy = base_config.copy()
        config_copy['record_traj_params']['save_dir'] = f'data/{dir_name}'
        config_copy.update({'agent_a_type': agent_type, 'agent_a_path': path})
        run = score_ex.run(config_updates=config_copy)
        assert run.status == 'COMPLETED'

    path_str = 'data/{}/agent_0.npz'
    density_modelers = [ActivationDensityModeler(path_str.format(s)) for s in dir_names]
    for modeler in density_modelers:
        modeler.get_density_model('ff_policy')








