#!/usr/bin/env bash

# Local directories
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
PROJECT_DIR=$( dirname "${SCRIPT_DIR}" )
PUBLIC_AWS=${PROJECT_DIR}/data/aws-public

# S3 Repos and commands
PRIVATE_S3_REPO=s3://adversarial-policies
PUBLIC_S3_REPO=s3://adversarial-policies-public
S3_SYNC_CMD="aws s3 sync --exclude='*/checkpoint/*' --exclude='*/datasets/*' --acl public-read --delete"

# Copy subset of data from private AWS to public view
echo "Syncing from private bucket ${PRIVATE_S3_REPO} to public bucket ${PUBLIC_S3_REPO}"

REMOTE_COPY="multi_train/paper/20190429_011349 score_agents"
for path in ${REMOTE_COPY}; do
  echo "Syncing ${path}"
  ${S3_SYNC_CMD} ${PRIVATE_S3_REPO}/${path} ${PUBLIC_S3_REPO}/${path}
done

echo "Syncing from local machine ${PUBLIC_AWS} to public bucket ${PUBLIC_S3_REPO}"
LOCAL_COPY="videos"
for path in ${LOCAL_COPY}; do
  echo "Syncing ${path}"
  ${S3_SYNC_CMD} ${PUBLIC_AWS}/${path} ${PUBLIC_S3_REPO}/${path}
done
