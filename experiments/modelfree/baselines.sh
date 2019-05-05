#!/usr/bin/env bash

CMD="python -m modelfree.multi.score with high_accuracy"
OUT_DIR=data/aws/score_agents

mkdir -p ${OUT_DIR}
for kind in zoo fixed; do
    mkdir -p ${OUT_DIR}/normal
    ${CMD} ${kind}_baseline save_path=${OUT_DIR}/normal/${kind}_baseline.json
    mkdir -p ${OUT_DIR}/victim_masked_init
    ${CMD} ${kind}_baseline mask_observations_of_victim \
           save_path=${OUT_DIR}/victim_masked_init/${kind}_baseline.json
    mkdir -p ${OUT_DIR}/victim_masked_zero
    ${CMD} ${kind}_baseline mask_observations_of_victim \
           score.mask_agent_kwargs.masking_type=zeros \
           save_path=${OUT_DIR}/victim_masked_zero/${kind}_baseline.json
done
