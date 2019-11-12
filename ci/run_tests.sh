#!/usr/bin/env bash

set -e  # exit immediately on any error

. ci/prepare_env.sh

set -o xtrace  # print commands

num_cpus=$2
if [[ ${num_cpus} == "" ]]; then
  num_cpus=$(nproc --all)
  num_cpus=$((${num_cpus} / 2))
fi

export LD_LIBRARY_PATH=/root/.mujoco/mujoco200/bin:${LD_LIBRARY_PATH}
COV_PACKAGES="aprl modelfree"
COV_FLAGS="--cov=tests"
for package in $COV_PACKAGES; do
    COV_FLAGS="$COV_FLAGS --cov=${venv}/lib/python3.7/site-packages/${package}"
done
pytest -vv -n ${num_cpus} ${COV_FLAGS} tests/

mv .coverage .coverage.${env}
coverage combine  # rewrite paths from virtualenv to src/
codecov
