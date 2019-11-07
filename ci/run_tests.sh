#!/usr/bin/env bash

env=$1
case $env in
aprl)
    export LD_LIBRARY_PATH=/root/.mujoco/mujoco200/bin:${LD_LIBRARY_PATH}
    ;;

modelfree)
    ;;
*)
    echo "Unrecognized environment '${env}'"
    exit 1
    ;;
esac

num_cpus=$2
if [[ ${num_cpus} == "" ]]; then
  num_cpus="auto"
fi

set -e  # exit immediately on any error

. ci/prepare_env.sh

set -o xtrace  # print commands

COV_PACKAGES="aprl modelfree"
COV_FLAGS=""
for package in $COV_PACKAGES; do
    COV_FLAGS="$COV_FLAGS --cov=${venv}/lib/python3.7/site-packages/${package}"
done
pytest -vv -n ${num_cpus} ${COV_FLAGS} tests/${env}

mv .coverage .coverage.${env}
coverage combine  # rewrite paths from virtualenv to src/
codecov --flags ${env}
