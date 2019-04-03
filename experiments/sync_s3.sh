#!/usr/bin/env bash

aws s3 sync --exclude='*/checkpoint/*' --exclude='*/datasets/*' s3://adversarial-policies/ data/aws/