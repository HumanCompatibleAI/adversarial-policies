DOCKER_REPO="humancompatibleai/adversarial_policies"
GIT_REPO="https://github.com/HumanCompatibleAI/adversarial-policies.git"

call_parallel() {
    PARALLEL_FLAGS=$1
    shift
    OUT_DIR=$1
    shift
    MODULE_NAME=$1
    shift
    EXTRA_ARGS=$*
    parallel ${PARALLEL_FLAGS} --header : --results ${OUT_DIR}/parallel \
              python -m ${MODULE_NAME} ${EXTRA_ARGS}
}
