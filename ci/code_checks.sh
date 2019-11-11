#!/usr/bin/env bash

SOURCE_DIRS="src experiments tests"

RET=0

echo "flake8 --version"
flake8 --version

echo "Linting code"
flake8 ${SOURCE_DIRS}
RET=$(($RET + $?))

echo "isort --version-number"
isort --version-number

echo "Checking import order using isort"
isort --recursive --diff ${SOURCE_DIRS}
isort --recursive --check-only ${SOURCE_DIRS}
RET=$(($RET + $?))

echo "Type checking"
pytype ${SOURCE_DIRS}
RET=$(($RET + $?))

if [ $RET -ne 0 ]; then
    echo "Linting failed."
fi
exit $RET
