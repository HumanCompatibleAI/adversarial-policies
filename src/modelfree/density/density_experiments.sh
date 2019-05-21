#!/usr/bin/env bash
ADV_PATH="data/aws/score_agents/2019-05-05T18:12:24+00:00/best_adversaries.json"

python -m modelfree.density.pipeline with gmm fit_density_model.model_kwargs.n_components=1 fit_density_model.model_kwargs.covariance_type=full generate_activations.adversary_path=${ADV_PATH}
python -m modelfree.density.pipeline with gmm fit_density_model.model_kwargs.n_components=2 fit_density_model.model_kwargs.covariance_type=full generate_activations.adversary_path=${ADV_PATH}
python -m modelfree.density.pipeline with gmm fit_density_model.model_kwargs.n_components=3 fit_density_model.model_kwargs.covariance_type=full generate_activations.adversary_path=${ADV_PATH}
python -m modelfree.density.pipeline with gmm fit_density_model.model_kwargs.n_components=5 fit_density_model.model_kwargs.covariance_type=full generate_activations.adversary_path=${ADV_PATH}
python -m modelfree.density.pipeline with gmm fit_density_model.model_kwargs.n_components=10 fit_density_model.model_kwargs.covariance_type=full generate_activations.adversary_path=${ADV_PATH}

python -m modelfree.density.pipeline with gmm fit_density_model.model_kwargs.n_components=1 fit_density_model.model_kwargs.covariance_type=diag generate_activations.adversary_path=${ADV_PATH}
python -m modelfree.density.pipeline with gmm fit_density_model.model_kwargs.n_components=2 fit_density_model.model_kwargs.covariance_type=diag generate_activations.adversary_path=${ADV_PATH}
python -m modelfree.density.pipeline with gmm fit_density_model.model_kwargs.n_components=3 fit_density_model.model_kwargs.covariance_type=diag generate_activations.adversary_path=${ADV_PATH}
python -m modelfree.density.pipeline with gmm fit_density_model.model_kwargs.n_components=5 fit_density_model.model_kwargs.covariance_type=diag generate_activations.adversary_path=${ADV_PATH}
python -m modelfree.density.pipeline with gmm fit_density_model.model_kwargs.n_components=10 fit_density_model.model_kwargs.covariance_type=diag generate_activations.adversary_path=${ADV_PATH}

python -m modelfree.density.pipeline with gmm fit_density_model.model_kwargs.n_components=1 fit_density_model.model_kwargs.covariance_type=spherical generate_activations.adversary_path=${ADV_PATH}
python -m modelfree.density.pipeline with gmm fit_density_model.model_kwargs.n_components=2 fit_density_model.model_kwargs.covariance_type=spherical generate_activations.adversary_path=${ADV_PATH}
python -m modelfree.density.pipeline with gmm fit_density_model.model_kwargs.n_components=3 fit_density_model.model_kwargs.covariance_type=spherical generate_activations.adversary_path=${ADV_PATH}
python -m modelfree.density.pipeline with gmm fit_density_model.model_kwargs.n_components=5 fit_density_model.model_kwargs.covariance_type=spherical generate_activations.adversary_path=${ADV_PATH}
python -m modelfree.density.pipeline with gmm fit_density_model.model_kwargs.n_components=10 fit_density_model.model_kwargs.covariance_type=spherical generate_activations.adversary_path=${ADV_PATH}
