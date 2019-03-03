call_parallel() {
    PARALLEL_FLAGS=$1
    shift
    PARALLEL_OUT_DIR=$1
    shift
    MODULE_NAME=$1
    shift
    PARALLEL_ARGS=$*
    parallel ${PARALLEL_FLAGS} --header : --results ${PARALLEL_OUT_DIR} \
             python -m ${MODULE_NAME} ${PARALLEL_ARGS}
}