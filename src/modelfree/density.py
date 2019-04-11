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









