#!/usr/bin/env bash

aws --no-sign-request s3 sync \
    --exclude='*/checkpoint/*' --exclude='*/datasets/*' --exclude='videos/*' \
    s3://adversarial-policies-public/ data/aws-public/
