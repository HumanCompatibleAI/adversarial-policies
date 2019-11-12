#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
. ${DIR}/common.sh

CMD="bash"
NAME="adversarial-policies"
TAG="latest"
RM="--rm"
FLAGS=""

while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    -c|--cmd)
    CMD="$2"
    shift
    shift
    ;;
    -l|--listen)
    FLAGS="${FLAGS} -p $2"
    shift
    shift
    ;;
    -n|--name)
    NAME="$2"
    shift
    shift
    ;;
    -p|--persist)
    RM=""
    shift
    ;;
    -t|--tag)
    TAG="$2"
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

docker run \
       ${FLAGS} \
       ${RM} \
       -it \
       --env MUJOCO_KEY=${MUJOCO_KEY} \
       --name ${NAME} \
       --mount type=bind,source="$(pwd)"/data,target=/adversarial-policies/data \
       ${DOCKER_REPO}:${TAG} \
       bash -c ". ci/prepare_env.sh && ${CMD}"
