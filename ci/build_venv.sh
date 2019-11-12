#!/usr/bin/env bash

set -e  # exit immediately on any error

venv=venv
virtualenv -p python3.7 ${venv}
source ${venv}/bin/activate
pip install -r requirements-build.txt
pip install -r requirements.txt

if [[ $USE_MPI == "True" ]]; then
  pip install mpi4py
fi
