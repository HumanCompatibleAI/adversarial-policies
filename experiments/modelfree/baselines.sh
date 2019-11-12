#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
. ${DIR}/common.sh

function multi_score {
  python -m aprl.multi.score with "$@" high_accuracy
}

OUT_DIR=data/aws/score_agents

mkdir -p ${OUT_DIR}
for kind in zoo fixed; do
    mkdir -p ${OUT_DIR}/normal
    multi_score ${kind}_baseline save_path=${OUT_DIR}/normal/${kind}_baseline.json&
    wait_proc

    mkdir -p ${OUT_DIR}/victim_masked_init
    multi_score ${kind}_baseline mask_observations_of_victim \
                save_path=${OUT_DIR}/victim_masked_init/${kind}_baseline.json&
    wait_proc

    mkdir -p ${OUT_DIR}/victim_masked_zero
    multi_score ${kind}_baseline mask_observations_of_victim mask_observations_with_zeros \
           save_path=${OUT_DIR}/victim_masked_zero/${kind}_baseline.json&
    wait_proc
done

wait
