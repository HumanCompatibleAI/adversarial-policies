#!/usr/bin/env bash

# Simple hyperparameter grid-search
# Does not test some relevant parameters (e.g. learning rate, entropy coefficient) at all.

ENV_NAMES="multicomp/KickAndDefend-v0 multicomp/SumoHumans-v0"
PRETRAINED="2"
SEEDS="0 1"
BATCH_SIZES="16384 2048"
EPOCHS="10 4"  # zipped with below (BATCHES) not cartesian product
BATCHES="32 4"

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
. ${DIR}/../common.sh

OUT_DIR=data/mf-hyper

# Train PPO against victims
call_train_parallel "$*" ${OUT_DIR} \
         env_name={env_name} victim_path={victim_path} seed={seed} batch_size={batch_size} \
         rl_args.ent_coef={ent_coef} rl_args.nminibatches={minibatches} rl_args.noptepochs={optim_epochs} \
         exp_name="victim{victim_path}-seed{seed}-batch{batch_size}-mini{minibatches}-optim{optim_epochs}-env{env_name}" \
         rew_shape=True rew_shape_params.anneal_frac=0 normalize=True num_env=8 \
         total_timesteps=3000000 learning_rate=3e-4 rl_args.ent_coef=0.00 \
         ::: env_name ${ENV_NAMES} \
         ::: victim_path ${PRETRAINED} \
         ::: batch_size ${BATCH_SIZES} \
         ::: optim_epochs ${EPOCHS} :::+ minibatches ${BATCHES} \
         ::: seed ${SEEDS}
