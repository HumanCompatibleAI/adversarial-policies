#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
ROOT_DIR="$( dirname "${SCRIPT_DIR}" )"

OPTIONS="-v -z -r -lpt"
EXCLUDES="LICENSE README.md setup.py scripts/doubleblind.sh ci/local_tests.sh .travis.yml experiments/common.sh experiments/planning 
          src/modelfree/configs/ray/ .git doubleblinded_src.zip *.pkl requirements*.txt"

# Refuse to compile if we find any of these words in non-excluded sources
BLACKLISTED="Adam Gleave Michael Dennis Cody Neel Kant Sergey Levine Stuart Russell berkeley humancompatibleai humancompatible"

TMPDIR=`mktemp --tmpdir -d doubleblinded.XXXXXXXX`

SYNC_CMD="rsync ${OPTIONS} --exclude-from=.gitignore"
for exclude in ${EXCLUDES}; do
  SYNC_CMD="${SYNC_CMD} --exclude=${exclude}"
done

${SYNC_CMD} ${ROOT_DIR} ${TMPDIR}
pushd ${TMPDIR}

GREP_TERMS=""
for pattern in ${BLACKLISTED}; do
  GREP_TERMS="${GREP_TERMS} -e ${pattern}"
done
grep -r . -i -F ${GREP_TERMS}
if [[ $? -ne 1 ]]; then
  echo "Found blacklisted word. Dieing."
  exit 1
fi

rm ${ROOT_DIR}/doubleblinded_src.zip
zip -r ${ROOT_DIR}/doubleblinded_src.zip .
popd
