#!/usr/bin/env bash

# Reproduce results of Dec 2018 draft write-up

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
. ${DIR}/../common.sh

ENV_NAMES="multicomp/KickAndDefend-v0 multicomp/SumoAnts-v0"
PRETRAINED="1 2 3"
SEEDS="0 1 2"

OUT_DIR=data/mf-dec2018rep

# Train PPO against victims
python -m modelfree.multi_train with dec2018rep

SCORE_AGENT="modelfree.score_agent with episodes=1000 num_env=16 render=False"
# Baseline: pretrained policy
call_parallel "$*" ${OUT_DIR}/pretrained ${SCORE_AGENT} \
         env_name={env_name} agent_a_path={agent_a_path} agent_b_path={agent_b_path} \
         ::: env_name ${ENV_NAMES} ::: agent_a_path ${PRETRAINED} ::: agent_b_path ${PRETRAINED}

# Baseline: random action and constant zero
call_parallel "$*" ${OUT_DIR}/fixed ${SCORE_AGENT} \
         env_name={env_name} agent_a_type={agent_a_type} agent_b_path={agent_b_path} \
         ::: env_name ${ENV_NAMES} ::: agent_a_type random zero ::: agent_b_path ${PRETRAINED}