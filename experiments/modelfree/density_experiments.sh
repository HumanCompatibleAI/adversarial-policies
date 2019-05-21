#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

. ${DIR}/common.sh

ADV_PATH="data/aws/score_agents/normal/2019-05-05T18:12:24+00:00/best_adversaries.json"

for components in 1 2 3 5 10; do
    for cov_type in full diag spherical; do
        python -m modelfree.density.pipeline with gmm fit_density_model.model_kwargs.n_components=${components} \
                                                      fit_density_model.model_kwargs.covariance_type=${cov_type} \
                                                      generate_activations.adversary_path=${ADV_PATH}&
        wait_proc
    done
done

wait