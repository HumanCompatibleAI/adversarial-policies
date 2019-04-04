#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
OUT_DIR=data/score_agents/
TIMESTAMP=`date --iso-8601=seconds`
ADVERSARY_PATHS=${OUT_DIR}/${TIMESTAMP}_best_adversaries.json
OUT_PATH=${OUT_DIR}/${TIMESTAMP}_adversary_transfer.json

if [[ $# -eq 0 ]]; then
    echo "usage: $0 <logdir> [logdir ...]"
    exit 1
fi

mkdir -p ${OUT_DIR}
python ${DIR}/highest_win_rate.py ${ADVERSARY_PATHS} --logdir=$*
ADVERSARY_PATHS=${ADVERSARY_PATHS} python -m modelfree.multi.score with adversary_transfer \
                                             save_path=${OUT_PATH}