#!/usr/bin/env bash

# Check we can learns on some simple (single-agent!) Gym environments

ENV_NAMES="Reacher-v1 Hopper-v1 Ant-v1 Humanoid-v1"
SEEDS="0 1 2"

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
. ${DIR}/common.sh

OUT_DIR=data/mf-single-agent

call_parallel "$*" ${OUT_DIR}/parallel modelfree.train with \
         env_name={env_name} seed={seed} \
         root_dir=${OUT_DIR}/baseline \
         exp_name="{env_name}-seed{seed}" \
         victim_type=none rew_shape=True rew_shape_params.anneal_frac=0 \
         total_timesteps=5000000 batch_size=2048 \
         ::: env_name ${ENV_NAMES} \
         ::: seed ${SEEDS} \
