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
source ${venv}/bin/activate
curl -o /root/.mujoco/mjkey.txt ${MUJOCO_KEY}
python setup.py install
pytest tests/${env} 
