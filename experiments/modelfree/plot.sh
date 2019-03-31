#!/usr/bin/env bash

# https://github.com/mrahtz/tbplot
TBPLOT="$HOME/dev/tbplot/tbplot"
ENV_NAMES="KickAndDefend-v0 SumoHumans-v0 SumoAnts-v0 \
          SumoHumansAutoContact-v0 SumoAntsAutoContact-v0 \
          RunToGoalHumans-v0 RunToGoalAnts-v0 \
          YouShallNotPassHumans-v0"
VICTIMS="1 2 3 4"

if [[ $# -neq 2 ]]; then
    echo "usage: $0 <TB data dir> <PNG output dir>"
    exit 1
fi

DATA_DIR="$1"
OUT_DIR="$2"

parallel -j 8 --header : \
         ${TBPLOT} --step --smoothing 0.9 \
         --out ${OUT_DIR}/{env_name}_{victim}.png \
         "${DATA_DIR}/train_rl_*_env_name:victim_path=\[*{env_name}*,\ {victim}\]*/data/baselines/*/rl/tb" \
         ::: env_name ${ENV_NAMES} \
         ::: victim ${VICTIMS}
