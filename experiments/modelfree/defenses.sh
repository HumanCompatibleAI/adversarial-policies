#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
. ${DIR}/common.sh

OUT_ROOT=score_agents/defenses
mkdir -p ${OUT_ROOT}

function multi_score {
  python -m aprl.multi.score with "$@" defenses high_accuracy
}

multi_score save_path=${OUT_ROOT}/normal.json&
wait_proc

multi_score mask_observations_of_victim save_path=${OUT_ROOT}/victim_masked_init.json&
wait_proc

wait
