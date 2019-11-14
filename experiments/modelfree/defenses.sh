#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
. ${DIR}/common.sh

OUT_ROOT=data/aws/score_agents/defenses
mkdir -p ${OUT_ROOT}

function multi_score {
  victims='["zoo", "json:data/aws/multi_train/finetune_defense_single_mlp/highest_win_policies_and_rates.json", "json:data/aws/multi_train/finetune_defense_dual_mlp/highest_win_policies_and_rates.json"]'
  opponents='["fixed", "zoo", "json:data/aws/multi_train/paper/highest_win_policies_and_rates.json", "json:data/aws/multi_train/adv_from_scratch_against_finetune_defense_single_mlp/highest_win_policies_and_rates.json", "json:data/aws/multi_train/adv_from_scratch_against_finetune_defense_dual_mlp/highest_win_policies_and_rates.json"]'
  python -m aprl.multi.score with victims="${victims}" opponents="${opponents}" envs='["multicomp/YouShallNotPassHumans-v0"]' "$@" high_accuracy
}

multi_score save_path=${OUT_ROOT}/normal.json&
wait_proc

multi_score mask_observations_of_victim save_path=${OUT_ROOT}/victim_masked_init.json&
wait_proc

wait
