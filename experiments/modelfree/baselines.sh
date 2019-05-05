#!/usr/bin/env bash

function wait_proc {
    if [[ -f ~/ray_bootstrap_config.yaml ]]; then
        # Running on a Ray cluster. We want to submit all the jobs in parallel.
        sleep 5  # stagger jobs a bit
    else
        # Running locally. Each job will start a Ray cluster. Submit sequentially.
        wait
    fi
}

CMD="python -m modelfree.multi.score with high_accuracy"
OUT_DIR=data/aws/score_agents

mkdir -p ${OUT_DIR}
for kind in zoo fixed; do
    mkdir -p ${OUT_DIR}/normal
    ${CMD} ${kind}_baseline save_path=${OUT_DIR}/normal/${kind}_baseline.json&
    wait_proc

    mkdir -p ${OUT_DIR}/victim_masked_init
    ${CMD} ${kind}_baseline mask_observations_of_victim \
           save_path=${OUT_DIR}/victim_masked_init/${kind}_baseline.json&
    wait_proc

    mkdir -p ${OUT_DIR}/victim_masked_zero
    ${CMD} ${kind}_baseline mask_observations_of_victim mask_observations_with_zeros \
           save_path=${OUT_DIR}/victim_masked_zero/${kind}_baseline.json&
    wait_proc
done

wait