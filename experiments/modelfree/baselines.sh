#!/usr/bin/env bash

OUT_DIR=data/aws/score_agents

mkdir -p ${OUT_DIR}
python -m modelfree.multi.score with zoo_baseline save_path=${OUT_DIR}/zoo_baseline.json
python -m modelfree.multi.score with zoo_baseline mask_observations_of_victim \
                                     save_path=${OUT_DIR}/zoo_baseline_victim_masked_init.json
python -m modelfree.multi.score with zoo_baseline mask_observations_of_victim \
                                     score.mask_agent_kwargs.masking_type=zero \
                                     save_path=${OUT_DIR}/zoo_baseline_victim_masked_zero.json
python -m modelfree.multi.score with fixed_baseline save_path=${OUT_DIR}/fixed_baseline.json
python -m modelfree.multi.score with fixed_baseline mask_observations_of_victim \
                                     save_path=${OUT_DIR}/fixed_baseline_victim_masked_init.json
python -m modelfree.multi.score with fixed_baseline mask_observations_of_victim \
                                     score.mask_agent_kwargs.masking_type=zero \
                                     save_path=${OUT_DIR}/fixed_baseline_victim_masked_zero.json
