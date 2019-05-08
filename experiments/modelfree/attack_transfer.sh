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

MULTI_SCORE_CMD="python -m modelfree.multi.score with high_accuracy adversary_transfer"

if [[ $# -eq 0 ]]; then
    echo "usage: $0 <logdir> [logdir ...]"
    exit 1
fi

for dir in normal victim_masked_init victim_masked_zero adversary_masked_init; do
    mkdir -p ${OUT_ROOT}/${dir}/${TIMESTAMP}
done

ADVERSARY_PATHS=${OUT_ROOT}/normal/${TIMESTAMP}/best_adversaries.json
python ${DIR}/highest_win_rate.py ${ADVERSARY_PATHS} --logdir $*

export ADVERSARY_PATHS=${ADVERSARY_PATHS}

${MULTI_SCORE_CMD} save_path=${OUT_ROOT}/normal/${TIMESTAMP}/adversary_transfer.json&
wait_proc

${MULTI_SCORE_CMD} mask_observations_of_victim \
    save_path=${OUT_ROOT}/victim_masked_init/${TIMESTAMP}/adversary_transfer.json&
wait_proc

${MULTI_SCORE_CMD} mask_observations_of_victim mask_observations_with_zeros \
    save_path=${OUT_ROOT}/victim_masked_zero/${TIMESTAMP}/adversary_transfer.json&
wait_proc

${MULTI_SCORE_CMD} mask_observations_of_adversary \
    save_path=${OUT_ROOT}/adversary_masked_init/${TIMESTAMP}/adversary_transfer.json&
wait_proc

wait