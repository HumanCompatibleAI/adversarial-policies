[![Build Status](https://travis-ci.com/HumanCompatibleAI/adversarial-policies.svg?branch=master)](https://travis-ci.com/HumanCompatibleAI/adversarial-policies)

Preliminary research investigating adversarial policies: given a victim policy 
in a multi-agent system, find a policy which will break the victim.

# Setup

This codebase assumes Python 3.6. Install the requirements in 
`requirements-build.txt` before those in `requirements.txt`.
Anaconda users can install directly with `conda env create -f environment.yml`.

Then install *one of* `requirements-aprl.txt` or `requirements-modelfree.txt`
depending on which experiments you need to run (unfortunately these codebases
depend on different MuJoCo and Gym versions.)

Finally, `pip install -e .` to install an editable version of the package.

# Contributions

Please run the `ci/code_checks.sh` before committing. This runs several linting steps.
These are also run as a continuous integration check.

I like to use Git commit hooks to prevent bad commits from happening in the first place:
```bash
ln -s ../../ci/code_checks.sh .git/hooks/pre-commit
```
