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

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
OUT_ROOT=data/aws/score_agents
TIMESTAMP=`date --iso-8601=seconds`

MULTI_SCORE_CMD="python -m modelfree.multi.score with medium_accuracy "

# Make a directory for each of the experiments we'll be running, to store results in
for dir in noisy_adversary_actions noisy_victim_actions noisy_victim_obs; do
    mkdir -p ${OUT_ROOT}/${dir}/${TIMESTAMP}
done

export ADVERSARY_PATHS=${OUT_ROOT}/normal/2019-05-05T18:12:24+00:00/best_adversaries.json

${MULTI_SCORE_CMD} zoo_baseline noise_adversary_actions \
    save_path=${OUT_ROOT}/noisy_adversary_actions/${TIMESTAMP}/noisy_zoo_opponent.json&
wait_proc

echo "Zoo baseline noisy actions completed"

${MULTI_SCORE_CMD} adversary_trained noise_adversary_actions \
    save_path=${OUT_ROOT}/noisy_adversary_actions/${TIMESTAMP}/noisy_adversary.json&
wait_proc

echo "Noisy actions completed"

${MULTI_SCORE_CMD} adversary_trained noise_victim_actions \
    save_path=${OUT_ROOT}/noisy_victim_actions/${TIMESTAMP}/noisy_victim.json&
wait_proc

echo "Noisy victim actions completed"

${MULTI_SCORE_CMD} zoo_baseline mask_observations_of_victim mask_observations_with_additive_noise \
    save_path=${OUT_ROOT}/noisy_victim_obs/${TIMESTAMP}/noisy_zoo_observations.json&
wait_proc
echo "Additive noise masking zoo baseline complete"

${MULTI_SCORE_CMD} adversary_trained mask_observations_of_victim mask_observations_with_additive_noise \
    save_path=${OUT_ROOT}/noisy_victim_obs/${TIMESTAMP}/noisy_adversary_observations.json&
wait_proc

${MULTI_SCORE_CMD} zoo_baseline mask_observations_of_victim mask_observations_with_smaller_additive_noise \
    save_path=${OUT_ROOT}/noisy_victim_obs/${TIMESTAMP}/smaller_noise_noisy_zoo_observations.json&
wait_proc
echo "Additive noise masking zoo baseline complete"

${MULTI_SCORE_CMD} adversary_trained mask_observations_of_victim mask_observations_with_smaller_additive_noise \
    save_path=${OUT_ROOT}/noisy_victim_obs/${TIMESTAMP}/smaller_noise_noisy_adversary_observations.json&
wait_proc


wait
echo "Additive noise masking complete"