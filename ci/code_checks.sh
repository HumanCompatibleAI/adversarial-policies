#!/usr/bin/env bash

# If you change these, also change .circle/config.yml.
SRC_FILES="src/ tests/ setup.py"
TYPECHECK_FILES="src/ tests/ setup.py"

set -x  # echo commands
set -e  # quit immediately on error

flake8 ${SRC_FILES}
black --check ${SRC_FILES}
codespell -I .codespell.skip --skip='*.pyc' ${SRC_FILES}

if [ -x "`which circleci`" ]; then
    circleci config validate
fi

if [ "$skipexpensive" != "true" ]; then
    pytype ${TYPECHECK_FILES}
fi
