#!/usr/bin/env bash

if [[ ${MUJOCO_KEY} == "" ]]; then
    echo "Set MUJOCO_KEY file to a URL with your key"
    exit 1
fi

# Run the same CI tests that Travis will run on local machine.
docker build --cache-from humancompatibleai/adversarial_policies:local-test \
             -t humancompatibleai/adversarial_policies:local-test .
if [[ $? -ne 0 ]]; then
    echo "Docker build failed"
    exit 1
fi

docker run --rm --env MUJOCO_KEY=${MUJOCO_KEY} --env CODECOV_TOKEN=${CODECOV_TOKEN} \
                      humancompatibleai/adversarial_policies:local-test \
                      ci/run_tests.sh
