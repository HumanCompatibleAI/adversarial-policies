#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
. ${DIR}/common.sh

REMOTE_HOST=""
LOCAL_DATA="${DIR}/../data"
REMOTE_WORK_DIR="/scratch/${USER}/aprl"
TB_PORT=6006
EXTRA_ARGS=""


while [[ $# -gt 0 ]]
do
key="$1"
case $key in
    -c|--cmd)
    CMD="$2"
    shift
    shift
    ;;
    -h|--host)
    REMOTE_HOST="$2"
    shift
    shift
    ;;
    -l|--listen)
    TB_PORT="$2"
    shift
    shift
    ;;
    -n|--name)
    NAME="$2"
    shift
    shift
    ;;
    -o|--output-dir)
    LOCAL_DATA="$2"
    shift
    shift
    ;;
    -w|--work-dir)
    REMOTE_WORK_DIR="$2"
    shift
    shift
    ;;
    *)
    EXTRA_ARGS="${EXTRA_ARGS} $1"
    shift
    ;;
esac
done

if [[ ${MUJOCO_KEY} == "" ]]; then
    echo "Set MUJOCO_KEY file to a URL with your key"
    exit 1
fi

if [[ ${REMOTE_HOST} == "" ]]; then
    echo "Missing mandatory argument -h <host>"
    exit 1
fi

set -o xtrace  # print commands
set -e  # exit immediately on any error

echo "Starting experiment"
ssh -t -L ${TB_PORT}:localhost:${TB_PORT} ${REMOTE_HOST} \
     "export MUJOCO_KEY='${MUJOCO_KEY}' && \
      git clone ${GIT_REPO} ${REMOTE_WORK_DIR}/${NAME} || (cd ${REMOTE_WORK_DIR}/${NAME} && git fetch) && \
      ${REMOTE_WORK_DIR}/${NAME}/experiments/build_and_run.sh \
          --no-copy -w ${REMOTE_WORK_DIR} -n ${NAME} -l ${TB_PORT} -c \"${CMD}\" ${EXTRA_ARGS}"

echo "Experiment completed, copying data"
rsync -rlptv --exclude=sacred ${REMOTE_HOST}:${REMOTE_WORK_DIR}/${NAME}/data/ ${LOCAL_DATA}/
rsync -rlptv ${REMOTE_HOST}:${REMOTE_WORK_DIR}/${NAME}/data/sacred/ ${LOCAL_DATA}/sacred/${REMOTE_HOST}
