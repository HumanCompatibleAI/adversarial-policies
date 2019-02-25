#!/usr/bin/env bash

# Reproduce results of Dec 2018 draft write-up

# TODO: try different victims?
# TODO: try more environments
parallel --header : --results data/parallel python -m modelfree.ppo_baseline ppo_baseline with \
         env_name={env_name} seed={seed} total_timesteps=5000000 batch_size=16384 \
         exp_name="mfrep-{env_name}-{seed}" \
         ::: env_name multicomp/KickAndDefend-v0 multicomp/SumoAnts-v0 \
         ::: seed 0 1 2
