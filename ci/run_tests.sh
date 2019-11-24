#!/usr/bin/env bash

set -e  # exit immediately on any error

echo "Downloading MuJoCo Key"
curl -o /root/.mujoco/mjkey.txt ${MUJOCO_KEY}

set -o xtrace  # print commands

num_cpus=$2
if [[ ${num_cpus} == "" ]]; then
  num_cpus=$(nproc --all)
  num_cpus=$((${num_cpus} / 2))
fi

export LD_LIBRARY_PATH=/root/.mujoco/mujoco200/bin:${LD_LIBRARY_PATH}
COV_FLAGS="--cov=tests --cov=/venv/lib/python3.7/site-packages/aprl"
pytest -vv -n ${num_cpus} ${COV_FLAGS} tests/

mv .coverage .coverage.tmp
coverage combine  # rewrite paths from virtualenv to src/
codecov
