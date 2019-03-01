#!/usr/bin/env bash

# Reproduce results of Dec 2018 draft write-up
parallel $* --header : --results data/parallel python -m modelfree.ppo_baseline ppo_baseline with \
         env_name={env_name} seed={seed} victim_path={victim_path} \
         exp_name="mfrep-victim{victim_path}-seed{seed}-{env_name}" \
         rew_shape=True rew_shape_params.anneal_frac={anneal_frac} \
         total_timesteps=5000000 batch_size=2048 \
         learning_rate=2.5e-4 rl_args.ent_coef=0.00 \
         ::: env_name multicomp/KickAndDefend-v0 multicomp/SumoAnts-v0 \
         ::: victim_path 1 2 3 \
         ::: seed 0 1 2 \
         ::: anneal_frac 0 0.1
