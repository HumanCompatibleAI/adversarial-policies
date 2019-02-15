#!/usr/bin/env bash

env=$1
case $env in
aprl)
    export LD_LIBRARY_PATH=/root/.mujoco/mjpro150/bin:${LD_LIBRARY_PATH}
    ;;

modelfree)
    ;;
*)
    echo "Unrecognized environment '${env}'"
    exit 1
    ;;
esac

set -e  # exit immediately on any error

venv=${env}venv
source ${venv}/bin/activate


echo "Downloading MuJoCo Key"
curl -o /root/.mujoco/mjkey.txt ${MUJOCO_KEY}

set -o xtrace  # print commands
pip install .

COV_PACKAGES="aprl modelfree"
COV_FLAGS=""
for package in $COV_PACKAGES; do
    COV_FLAGS="$COV_FLAGS --cov=${venv}/lib/python3.6/site-packages/${package}"
done
pytest -vv $COV_FLAGS tests/${env}

mv .coverage .coverage.${env}
coverage combine  # rewrite paths from virtualenv to src/
codecov --flags ${env}
