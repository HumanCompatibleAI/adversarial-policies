#!/usr/bin/env bash

# Test if VecNormalize is helpful

ENV_NAMES="multicomp/KickAndDefend-v0 multicomp/SumoAnts-v0"
PRETRAINED="1 2 3"
SEEDS="0 1 2"

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
. ${DIR}/common.sh

OUT_DIR=data/mf-normalize

# Train PPO against victims
call_parallel "$*" modelfree.train with \
         env_name={env_name} seed={seed} victim_path={victim_path} normalize={normalize} \
         root_dir=${OUT_DIR}/parallel \
         exp_name="victim{victim_path}-seed{seed}-norm{normalize}-{env_name}" \
         rew_shape=True rew_shape_params.anneal_frac=0 \
         total_timesteps=5000000 batch_size=2048 \
         learning_rate=2.5e-4 rl_args.ent_coef=0.00 \
         ::: env_name ${ENV_NAMES} \
         ::: victim_path ${PRETRAINED} \
         ::: normalize True False \
         ::: seed ${SEEDS}