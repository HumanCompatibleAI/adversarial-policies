#!/usr/bin/env bash

S3_SYNC_CMD="aws s3 sync --exclude='*/checkpoint/*' --exclude='*/datasets/*'"

${S3_SYNC_CMD} s3://adversarial-policies/ data/aws/
${S3_SYNC_CMD} s3://adversarial-policies-public/ data/aws-public/
