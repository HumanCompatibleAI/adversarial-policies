#!/usr/bin/env bash

aws s3 sync --exclude='*/checkpoint/*' --exclude='*/datasets/*' s3://adversarial-policies/ data/aws/
aws s3 sync --exclude='*/checkpoint/*' --exclude='*/datasets/*' s3://adversarial-policies-public/ data/aws-public/
