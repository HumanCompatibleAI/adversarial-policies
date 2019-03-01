#!/usr/bin/env bash

TEST_SUITES="aprl modelfree"

if [[ ${MUJOCO_KEY} == "" ]]; then
    echo "Set MUJOCO_KEY file to a URL with your key"
    exit 1
fi

# Run the same CI tests that Travis will run on local machine.
docker build --cache-from humancompatibleai/adversarial_policies:local-test \
             --build-arg MUJOCO_KEY=${MUJOCO_KEY} \
             -t humancompatibleai/adversarial_policies:local-test .
if [[ $? -ne 0 ]]; then
    echo "Docker build failed"
    exit 1
fi

RET=0
for suite in ${TEST_SUITES}; do
    docker run --rm --env MUJOCO_KEY=${MUJOCO_KEY} \
                          humancompatibleai/adversarial_policies:local-test \
                          ci/run_tests.sh ${suite}
    RET=$(($RET + $?))
done

if [[ $RET -eq 0 ]]; then
    echo "All tests passed"
else
    echo "Test failed"
fi
exit $RET
