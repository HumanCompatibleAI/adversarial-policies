#!/usr/bin/env bash

venv=venv
. ${venv}/bin/activate

echo "Downloading MuJoCo Key"
curl -o /root/.mujoco/mjkey.txt ${MUJOCO_KEY}
