#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

. ${DIR}/common.sh

TIMESTAMP=`date --iso-8601=seconds`
ACTIVATION_DIR="data/density/activations/${TIMESTAMP}"

python -m modelfree.common.generate_activations with score_update.score.timesteps=40000 \
                                                     out_dir=${ACTIVATION_DIR}

for components in 5 10 20 40 80; do
    for cov_type in full diag; do
        python -m modelfree.density.pipeline with fit_density_model.gmm \
                                                  fit_density_model.model_kwargs.n_components=${components} \
                                                  fit_density_model.model_kwargs.covariance_type=${cov_type} \
                                                  fit_density_model.activation_dir=${ACTIVATION_DIR}
        wait_proc
    done
done

wait