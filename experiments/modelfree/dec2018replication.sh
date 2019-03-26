#!/usr/bin/env bash

# Reproduce results of Dec 2018 draft write-up

ENV_NAMES="multicomp/KickAndDefend-v0 multicomp/SumoAnts-v0"
PRETRAINED="1 2 3"
SEEDS="0 1 2"

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
. ${DIR}/../common.sh

OUT_DIR=data/mf-dec2018rep

# Train PPO against victims
call_train_parallel "$*" ${OUT_DIR}/adversarial \
         env_name={env_name} seed={seed} victim_path={victim_path} \
         exp_name="victim{victim_path}-seed{seed}-anneal{anneal_frac}-{env_name}" \
         rew_shape=True rew_shape_params.anneal_frac={anneal_frac} \
         total_timesteps=5000000 batch_size=2048 \
         learning_rate=2.5e-4 rl_args.ent_coef=0.00 \
         ::: env_name ${ENV_NAMES} \
         ::: victim_path ${PRETRAINED} \
         ::: seed ${SEEDS} \
         ::: anneal_frac 0 0.1

SCORE_AGENT="modelfree.score_agent with episodes=1000 num_env=16 render=False"
# Baseline: pretrained policy
call_parallel "$*" ${OUT_DIR}/pretrained ${SCORE_AGENT} \
         env_name={env_name} agent_a_path={agent_a_path} agent_b_path={agent_b_path} \
         ::: env_name ${ENV_NAMES} ::: agent_a_path ${PRETRAINED} ::: agent_b_path ${PRETRAINED}

# Baseline: random action and constant zero
call_parallel "$*" ${OUT_DIR}/fixed ${SCORE_AGENT} \
         env_name={env_name} agent_a_type={agent_a_type} agent_b_path={agent_b_path} \
         ::: env_name ${ENV_NAMES} ::: agent_a_type random zero ::: agent_b_path ${PRETRAINED}