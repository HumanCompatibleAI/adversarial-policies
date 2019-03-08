#!/usr/bin/env bash

NUM_ENVS="1 2 4 8 16"

for num_env in $NUM_ENVS; do
    for rep in 1 2 3; do
        echo "BENCHMARK: ${num_env} environments test ${rep}"
        time python -m modelfree.train with \
                       total_timesteps=50000 num_env=${num_env} \
                       exp_name="vec-env-benchmark-${num_env}-${rep}"
    done
done