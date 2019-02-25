#!/usr/bin/env bash

venv=${env}venv
source ${venv}/bin/activate

echo "Downloading MuJoCo Key"
curl -o /root/.mujoco/mjkey.txt ${MUJOCO_KEY}

echo "Installing our code"
pip install .
