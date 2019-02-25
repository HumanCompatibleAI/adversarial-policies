#!/usr/bin/env bash

CMD="bash"
NAME="adversarial-policies"
VENV="modelfree"
TAG="latest"
RM="--rm"

while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    -n|--name)
    NAME="$2"
    shift
    shift
    ;;
    -c|--cmd)
    CMD="$2"
    shift
    shift
    ;;
    -p|--persist)
    RM=""
    shift
    ;;
    -v|--venv)
    VENV="$2"
    shift
    shift
    ;;
    -t|--tag)
    TAG="$2"
    shift
    shift
    ;;
esac
done

docker run \
       ${RM} \
       -it \
       --env MUJOCO_KEY=${MUJOCO_KEY} \
       --name ${NAME} \
       --mount type=bind,source="$(pwd)"/data,target=/adversarial-policies/data \
       humancompatibleai/adversarial_policies:${TAG} \
       bash -c ". ${VENV}venv/bin/activate && pip install . && ${CMD}"
