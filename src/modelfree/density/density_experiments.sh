#!/usr/bin/env bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
#ADV_PATH="/home/ubuntu/data/best_adversaries.json"
ADV_PATH="data/aws/score_agents/2019-05-05T18:12:24+00:00/best_adversaries.json"

#python ${DIR}/pipeline.py with gmm fit_density_model.model_kwargs.n_components=1 fit_density_model.model_kwargs.covariance_type=full tsne_activations.adversary_path=${ADV_PATH}
#python ${DIR}/pipeline.py with gmm fit_density_model.model_kwargs.n_components=2 fit_density_model.model_kwargs.covariance_type=full tsne_activations.adversary_path=${ADV_PATH}
python ${DIR}/pipeline.py with gmm fit_density_model.model_kwargs.n_components=3 fit_density_model.model_kwargs.covariance_type=full tsne_activations.adversary_path=${ADV_PATH}
python ${DIR}/pipeline.py with gmm fit_density_model.model_kwargs.n_components=5 fit_density_model.model_kwargs.covariance_type=full tsne_activations.adversary_path=${ADV_PATH}
python ${DIR}/pipeline.py with gmm fit_density_model.model_kwargs.n_components=10 fit_density_model.model_kwargs.covariance_type=full tsne_activations.adversary_path=${ADV_PATH}

python ${DIR}/pipeline.py with gmm fit_density_model.model_kwargs.n_components=1 fit_density_model.model_kwargs.covariance_type=diag tsne_activations.adversary_path=${ADV_PATH}
python ${DIR}/pipeline.py with gmm fit_density_model.model_kwargs.n_components=2 fit_density_model.model_kwargs.covariance_type=diag tsne_activations.adversary_path=${ADV_PATH}
python ${DIR}/pipeline.py with gmm fit_density_model.model_kwargs.n_components=3 fit_density_model.model_kwargs.covariance_type=diag tsne_activations.adversary_path=${ADV_PATH}
python ${DIR}/pipeline.py with gmm fit_density_model.model_kwargs.n_components=5 fit_density_model.model_kwargs.covariance_type=diag tsne_activations.adversary_path=${ADV_PATH}
python ${DIR}/pipeline.py with gmm fit_density_model.model_kwargs.n_components=10 fit_density_model.model_kwargs.covariance_type=diag tsne_activations.adversary_path=${ADV_PATH}

python ${DIR}/pipeline.py with gmm fit_density_model.model_kwargs.n_components=1 fit_density_model.model_kwargs.covariance_type=spherical tsne_activations.adversary_path=${ADV_PATH}
python ${DIR}/pipeline.py with gmm fit_density_model.model_kwargs.n_components=2 fit_density_model.model_kwargs.covariance_type=spherical tsne_activations.adversary_path=${ADV_PATH}
python ${DIR}/pipeline.py with gmm fit_density_model.model_kwargs.n_components=3 fit_density_model.model_kwargs.covariance_type=spherical tsne_activations.adversary_path=${ADV_PATH}
python ${DIR}/pipeline.py with gmm fit_density_model.model_kwargs.n_components=5 fit_density_model.model_kwargs.covariance_type=spherical tsne_activations.adversary_path=${ADV_PATH}
python ${DIR}/pipeline.py with gmm fit_density_model.model_kwargs.n_components=10 fit_density_model.model_kwargs.covariance_type=spherical tsne_activations.adversary_path=${ADV_PATH}