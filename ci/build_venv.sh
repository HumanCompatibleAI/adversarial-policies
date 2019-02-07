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

venv=${env}venv
virtualenv -p python3.6 ${venv}
source ${venv}/bin/activate

pip install -r requirements-build.txt
pip install -r requirements.txt
pip install -r ${env}/requirements.txt