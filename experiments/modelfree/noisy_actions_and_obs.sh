#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
. ${DIR}/common.sh

AWS_ROOT=data/aws
OUT_ROOT=${AWS_ROOT}/score_agents
TIMESTAMP=`date --iso-8601=seconds`

# Format: multi_score <opponent_type> <noise_type> ["extra_config1 ..."]
# opponent_type: one of zoo_baseline or adversary_trained
# noise_type: one of ${NOISE_TYPES}
# extra_config: a string with a list of space-separated named configs for modelfree.multi.score
# Saves to ${noise_type}/${TIMESTMAP}/${opponent_type}.json
function multi_score {
    opponent_type=$1
    noise_type=$2
    extra_configs=$3

    python -m modelfree.multi.score with ${opponent_type} ${noise_type} ${extra_configs} \
              medium_accuracy save_path=${OUT_ROOT}/${noise_type}/${TIMESTAMP}/${opponent_type}.json
    wait_proc
}

# Sanity check we have the data
if [[ ! -d ${OUT_ROOT} || ! -d ${AWS_ROOT}/multi_train ]]; then
  echo "Could not find some required data dierctories."
  echo "Consider running these commands (if using Ray, add to {head,worker}_start_ray_commands):"
  echo "aws s3 sync s3://adversarial-policies/score_agents/ /adversarial-policies/data/aws/score_agents/ &&"
  echo "aws s3 sync --exclude='*/checkpoint/*' --exclude='*/datasets/*' \
        s3://adversarial-policies/multi_train/paper/20190429_011349/ \
        /adversarial-policies/data/aws/multi_train/paper/20190429_011349/"
  exit 1
fi

# Make a directory for each of the noise types we're using, to store results in
NOISE_TYPES="noise_adversary_actions noise_victim_actions mask_observations_with_additive_noise \
            mask_observations_with_smaller_additive_noise"
for dir in ${NOISE_TYPES}; do
    mkdir -p ${OUT_ROOT}/${dir}/${TIMESTAMP}
done

export ADVERSARY_PATHS=${OUT_ROOT}/normal/2019-05-05T18:12:24+00:00/best_adversaries.json

multi_score zoo_baseline noise_adversary_actions
echo "Zoo baseline noisy actions completed"

multi_score adversary_trained noise_adversary_actions
echo "Noisy actions completed"

multi_score adversary_trained noise_victim_actions
echo "Noisy victim actions completed"

multi_score zoo_baseline mask_observations_with_additive_noise mask_observations_of_victim
multi_score adversary_trained mask_observations_with_additive_noise mask_observations_of_victim
echo "Additive noise masking baseline complete"

multi_score zoo_baseline mask_observations_with_smaller_additive_noise mask_observations_of_victim
multi_score adversary_trained mask_observations_with_smaller_additive_noise mask_observations_of_victim
echo "Additive noise masking baseline complete"

wait
echo "Additive noise masking complete"