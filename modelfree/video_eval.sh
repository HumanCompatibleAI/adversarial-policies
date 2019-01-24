#!/bin/bash

#AGENTS="pretrainedv1 mlp_train_default_shape mlp_train_no_shape out_lstm_rand out_random_const"
AGENTS="pretrainedv1"
declare -A AGENT_TYPES
AGENT_TYPES=( ["pretrainedv1"]="zoo" ["mlp_train_default_shape"]="our_mlp" ["mlp_train_no_shape"]="our_mlp" ["out_lstm_rand"]="lstm" ["out_random_const"]="const" )

for agent in ${AGENTS}; do 
    python gather_statistics.py --samples 100 --no_visuals True --save-video videos/${agent} --agent_to_eval results_for_adam/${agent}.pkl --agent_type "${AGENT_TYPES[$agent]}" >logs/${agent} 2>&1&
done

wait
