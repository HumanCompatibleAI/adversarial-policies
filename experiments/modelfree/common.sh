#!/usr/bin/env bash

function wait_proc {
    if [[ -f ~/ray_bootstrap_config.yaml ]]; then
        # Running on a Ray cluster. We want to submit all the jobs in parallel.
        sleep 5  # stagger jobs a bit
    else
        # Running locally. Each job will start a Ray cluster. Submit sequentially.
        wait
    fi
}
