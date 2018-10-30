# aerl

This is an preliminary test bed for experiements involving adversarial examples for multi-agent RL.  Specifically, we are looking for adversairial examples which lie within the strategy space for the opponent.

## Work So Far

I have included the repo for multiagent-competition, this includes both a bunch of mujoco environments that can be used as test beds and also a bunch of pre-trained agents that were trained with self-play.  The ant environment here seems to be a good place to start, since it is simple enough that we should be able to train somewhat effectivly but complex enough that there are likely adversarial examples.  

It also has the benefit that it was the quickest thing I could get working.  I have writen most of the code to be independent of the decision about which environment we are using so we can adjust later if it is not easy/hard enough.  

Notes on setup:
* You should follow their installation instructions in their readme 
    * Should run pip on their requirements.txt
    * Should run pip `install -e .` in the gym-compete directory
* I had to change the version of gym to 0.9.1. I changed this in their requirements.txt
* You also need a different version of mujocu, I have mjpro131
* If you have everything installed correctly `bash demo_tasks.sh` should run and should show all of the environments running with agents out of the zoo.

After this I took their main file and refactored it to make it easier to run tests with.  I also added a `simulation_utils` file, some of the functions in there are unused and untested, I took the code from another one of my repos and modified it to work with multi-agent environments.



## Baselines

Next I began working on the baselines that we should have for finding adversarial examples to RL policies.  The two that
come to mind are the random search baseline and the black-box rl baseline.  I am currently working on implementing these
two in that order.  If there is nothing else in this file that means I have not yet finished.