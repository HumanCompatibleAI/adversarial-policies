from pipeline import density_ex
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity


class PCAPreDensity(object):
    def __init__(self, density_class, pca_components, **kwargs):
        super(PCAPreDensity, self).__init__()
        self.density_class = density_class
        self.num_components = pca_components
        self.kwargs = kwargs
        self.density_obj = self.density_class(**self.kwargs)
        self.pca_obj = PCA(n_components=self.num_components)

    def fit(self, X):
        reduced_representation = self.pca_obj.fit_transform(X)
        self.density_obj.fit(reduced_representation)

    def score_samples(self, X):
        reduced_test_representation = self.pca_obj.transform(X)
        return self.density_obj.score_samples(reduced_test_representation)


@density_ex.named_config
def kde(fit_density_model):
    fit_density_model = dict(fit_density_model)
    fit_density_model['model_class'] = KernelDensity
    _ = locals()  # quieten flake8 unused variable warning
    del _


@density_ex.named_config
def gmm(fit_density_model):
    fit_density_model = dict(fit_density_model)
    fit_density_model['model_class'] = GaussianMixture
    _ = locals()  # quieten flake8 unused variable warning
    del _


@density_ex.named_config
def pca_kde(fit_density_model):
    fit_density_model = dict(fit_density_model)
    fit_density_model['model_class'] = PCAPreDensity
    fit_density_model['model_kwargs'] = {'density_class': KernelDensity}


@density_ex.named_config
def pca_gmm(fit_density_model):
    fit_density_model = dict(fit_density_model)
    fit_density_model['model_class'] = PCAPreDensity
    fit_density_model['model_kwargs'] = {'density_class': GaussianMixture}
