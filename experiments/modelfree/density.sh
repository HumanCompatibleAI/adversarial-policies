#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

. ${DIR}/common.sh

TIMESTAMP=`date --iso-8601=seconds`
ACTIVATION_DIR="data/density/${TIMESTAMP}/activations"

# We fit our density model with 20,000 timesteps, and use 20,000 timesteps for evaluation.
# So we need 40,000 timesteps for the training opponent. The others we only actually need 20,000
# for so this is slightly wasteful.
python -m modelfree.common.generate_activations with score_update.score.timesteps=40000 \
                                                     out_dir=${ACTIVATION_DIR}

for components in 5 10 20 40 80; do
    for cov_type in full diag; do
        python -m modelfree.density.fit_density with gmm \
                                                model_kwargs.n_components=${components} \
                                                model_kwargs.covariance_type=${cov_type} \
                                                activation_dir=${ACTIVATION_DIR}
                                                output_root=data/density/${TIMESTAMP}/fitted
        wait_proc
    done
done

wait