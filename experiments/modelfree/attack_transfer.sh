#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
OUT_DIR=data/aws/score_agents/
TIMESTAMP=`date --iso-8601=seconds`
ADVERSARY_PATHS=${OUT_DIR}/${TIMESTAMP}_best_adversaries.json
OUT_PREFIX=${OUT_DIR}/${TIMESTAMP}_adversary_transfer
MULTI_SCORE_CMD="python -m modelfree.multi.score with adversary_transfer"

if [[ $# -eq 0 ]]; then
    echo "usage: $0 <logdir> [logdir ...]"
    exit 1
fi

mkdir -p ${OUT_DIR}
python ${DIR}/highest_win_rate.py ${ADVERSARY_PATHS} --logdir $*
ADVERSARY_PATHS=${ADVERSARY_PATHS} ${MULTI_SCORE_CMD} save_path=${OUT_PREFIX}.json
ADVERSARY_PATHS=${ADVERSARY_PATHS} ${MULTI_SCORE_CMD} mask_observations_of_victim \
                                                      save_path=${OUT_PREFIX}_victim_masked_init.json
ADVERSARY_PATHS=${ADVERSARY_PATHS} ${MULTI_SCORE_CMD} mask_observations_of_victim \
                                                      score.mask_agent_kwargs.masking_type=zero \
                                                      save_path=${OUT_PREFIX}_victim_masked_zero.json
ADVERSARY_PATHS=${ADVERSARY_PATHS} ${MULTI_SCORE_CMD} mask_observations_of_adversary \
                                                      save_path=${OUT_PREFIX}_adversary_masked.json
