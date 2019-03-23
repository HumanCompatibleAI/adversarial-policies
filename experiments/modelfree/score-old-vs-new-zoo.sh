#!/usr/bin/env bash

# Score the same gym_compete agents under the new and old code
# Should get the same results. Differences suggest some implementation bug.

# Choose mix of environments with mlp and lstm policies
ENV_NAMES="multicomp/KickAndDefend-v0 multicomp/SumoHumans-v0 \
           multicomp/RunToGoalHumans-v0 multicomp/YouShallNotPassHumans-v0"
AGENT_TYPES="zoo zoo_old"

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
. ${DIR}/../common.sh

OUT_DIR=data/score-old-vs-new

# Train PPO against victims
call_parallel "$*" ${OUT_DIR} modelfree.score_agent with \
         env_name={env_name} agent_a_type={a_type} agent_b_type={b_type} \
         agent_a_path=1 agent_b_path=1 episodes=1000 num_env=4 render=False \
         ::: env_name ${ENV_NAMES} \
         ::: a_type ${AGENT_TYPES} \
         ::: b_type ${AGENT_TYPES}
