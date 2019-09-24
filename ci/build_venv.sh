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

venv=${env}venv
virtualenv -p python3.7 ${venv}  && \
source ${venv}/bin/activate && \
pip install -r requirements-build.txt && \
pip install -r requirements.txt && \
pip install -r requirements-${env}.txt

if [[ $USE_MPI == "True" ]]; then
  pip install mpi4py
fi
