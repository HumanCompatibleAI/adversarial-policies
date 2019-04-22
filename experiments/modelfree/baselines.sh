#!/usr/bin/env bash

OUT_DIR=data/score_agents

mkdir -p ${OUT_DIR}
python -m modelfree.multi.score with zoo_baseline save_path=${OUT_DIR}/zoo_baseline.json
python -m modelfree.multi.score with fixed_baseline save_path=${OUT_DIR}/fixed_baseline.json