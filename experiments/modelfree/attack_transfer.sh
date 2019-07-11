#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
. ${DIR}/common.sh

OUT_ROOT=data/aws/score_agents
TIMESTAMP=`date --iso-8601=seconds`

function multi_score {
  python -m modelfree.multi.score with adversary_transfer "$@" high_accuracy
}

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

multi_score save_path=${OUT_ROOT}/normal/${TIMESTAMP}/adversary_transfer.json&
wait_proc

multi_score mask_observations_of_victim \
    save_path=${OUT_ROOT}/victim_masked_init/${TIMESTAMP}/adversary_transfer.json&
wait_proc

multi_score mask_observations_of_victim mask_observations_with_zeros \
    save_path=${OUT_ROOT}/victim_masked_zero/${TIMESTAMP}/adversary_transfer.json&
wait_proc

multi_score mask_observations_of_adversary \
    save_path=${OUT_ROOT}/adversary_masked_init/${TIMESTAMP}/adversary_transfer.json&
wait_proc

wait