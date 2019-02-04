#!/usr/bin/env bash

curl -o /root/.mujoco/mjkey.txt ${MUJOCO_KEY}
pytest
