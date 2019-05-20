#!/usr/bin/env bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"


python ${DIR}/pipeline.py with gmm high_accuracy fit_density_model.model_kwargs.n_components=1 fit_density_model.model_kwargs.covariance_type=full
python ${DIR}/pipeline.py with gmm high_accuracy fit_density_model.model_kwargs.n_components=2 fit_density_model.model_kwargs.covariance_type=full
python ${DIR}/pipeline.py with gmm high_accuracy fit_density_model.model_kwargs.n_components=5 fit_density_model.model_kwargs.covariance_type=full
python ${DIR}/pipeline.py with gmm high_accuracy fit_density_model.model_kwargs.n_components=10 fit_density_model.model_kwargs.covariance_type=full

#python ${DIR}/pipeline.py with gmm fit_density_model.model_kwargs.n_components=1 fit_density_model.model_kwargs.covariance_type=diag
#python ${DIR}/pipeline.py with gmm fit_density_model.model_kwargs.n_components=2 fit_density_model.model_kwargs.covariance_type=diag
#python ${DIR}/pipeline.py with gmm fit_density_model.model_kwargs.n_components=5 fit_density_model.model_kwargs.covariance_type=diag
#python ${DIR}/pipeline.py with gmm fit_density_model.model_kwargs.n_components=10 fit_density_model.model_kwargs.covariance_type=diag
#
#python ${DIR}/pipeline.py with gmm fit_density_model.model_kwargs.n_components=1 fit_density_model.model_kwargs.covariance_type=spherical
#python ${DIR}/pipeline.py with gmm fit_density_model.model_kwargs.n_components=2 fit_density_model.model_kwargs.covariance_type=spherical
#python ${DIR}/pipeline.py with gmm fit_density_model.model_kwargs.n_components=5 fit_density_model.model_kwargs.covariance_type=spherical
#python ${DIR}/pipeline.py with gmm fit_density_model.model_kwargs.n_components=10 fit_density_model.model_kwargs.covariance_type=spherical