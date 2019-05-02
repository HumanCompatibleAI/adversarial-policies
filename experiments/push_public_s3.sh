#!/usr/bin/env bash

aws s3 sync --acl public-read --delete data/aws-public/ s3://adversarial-policies-public/
