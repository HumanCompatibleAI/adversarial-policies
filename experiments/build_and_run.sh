#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
. ${DIR}/common.sh

CMD="bash"
COPY="True"
DETACH="False"
WORK_DIR="$HOME/aprl"
NAME="adversarial-policies"
TREEISH="master"
TB_PORT=6006
RUN_DOCKER_ARGS=""

while [[ $# -gt 0 ]]
do
key="$1"
case $key in
    -c|--cmd)
    CMD="$2"
    shift
    shift
    ;;
    -d|--detach)
    DETACH="True"
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
    --no-copy)
    COPY="False"
    shift
    ;;
    -r|--revision)
    TREEISH="$2"
    shift
    shift
    ;;
    -w|--work-dir)
    WORK_DIR="$2"
    shift
    shift
    ;;
    --run-docker-args)
    RUN_DOCKER_ARGS="$2"
    shift
    shift
    ;;
    *)
    echo "Unrecognized option '${key}'"
    exit 1
esac
done

if [[ ${MUJOCO_KEY} == "" ]]; then
    echo "Set MUJOCO_KEY file to a URL with your key"
    exit 1
fi

set -e  # exit immediately on any error

if [[ ${COPY} == "True" ]]; then
    git clone ${GIT_REPO} ${WORK_DIR}/${NAME}
fi

cd ${WORK_DIR}/${NAME}
git checkout ${TREEISH}
docker build --cache-from ${DOCKER_REPO}:${NAME} \
             --build-arg MUJOCO_KEY=${MUJOCO_KEY} \
             -t ${DOCKER_REPO}:${NAME} .

mkdir -p data
tmux new-session -d -s ${NAME} \
     "export MUJOCO_KEY=${MUJOCO_KEY} &&
     ./experiments/run_docker.sh -t ${NAME} -l ${TB_PORT}:6006 \
                                 -n ${NAME} -c ${CMD} ${RUN_DOCKER_ARGS}; \
     echo 'Finished; press Ctrl-D to exit'; cat /dev/stdin"
ATTEMPTS=0
while [[ `docker inspect -f {{.State.Running}} ${NAME}` != "true" ]]; do
    echo "Waiting for Docker container to start"
    sleep 2
    ATTEMPTS=$((ATTEMPTS + 1))
    if [[ $ATTEMPTS -gt 5 ]]; then
        echo "Could not start Docker container. Dieing. Look in tmux session '${NAME}' for logs."
        exit 1
    fi
done
tmux new-window -t ${NAME} -d \
    "docker exec ${NAME} bash -c \"env=aprl . ci/prepare_env.sh && tensorboard --port 6006 --logdir data/\""

if [[ ${DETACH} == "True" ]]; then
    echo "Experiment '${NAME}' running in eponymous tmux session, \
          cwd '${WORK_DIR}/${NAME}' and TensorBoard running on port '${TB_PORT}'"
else
    tmux attach-session -t ${NAME}
fi