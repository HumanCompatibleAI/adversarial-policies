# aerl

This is an preliminary test bed for experiements involving adversarial examples for multi-agent RL.  Specifically, we 
are looking for adversairial examples which lie within the strategy space for the opponent.

## Work So Far

I have included the repo for multiagent-competition, this includes both a bunch of mujoco environments that can be used 
as test beds and also a bunch of pre-trained agents that were trained with self-play.  The ant environment here seems 
to be a good place to start, since it is simple enough that we should be able to train somewhat effectivly but complex 
enough that there are likely adversarial examples.  

It also has the benefit that it was the quickest thing I could get working.  I have writen most of the code to be 
independent of the decision about which environment we are using so we can adjust later if it is not easy/hard enough.  

Notes on setup:
* You should follow their installation instructions in their readme 
    * Should run pip on their requirements.txt
    * Should run `pip install -e .` in the gym-compete directory
* I had to change the version of gym to 0.9.1. I changed this in their requirements.txt
* You also need a different version of mujocu, I have mjpro131
* If you have everything installed correctly `bash demo_tasks.sh` should run and should show all of the environments 
running with agents out of the zoo.

After this I took their main file and refactored it to make it easier to run tests with.  I also added a 
`simulation_utils` file, some of the functions in there are unused and untested, I took the code from another one of
 my repos and modified it to work with multi-agent environments.



## Baselines

I have a baseline that does random search.  It currently does it over a pretty stupid space of policies but this can 
easily be changed.  The other baseline that we should have is black-box RL which would fix the opponent and do any 
RL algorithm against it.  We should aim to beat both of these in terms of sample efficiency.

I have also noticed something pretty unfortunate about the environments they used in this paper.  They don't have a very
rich observation space so it probably won't work for adversarial examples.  It turns out that though the only feature 
that one agent knows about the other is how far away they are, which does not allow much room for adversarial 
attacks.  It seems like being one of these ant creatures is a very sad life.  They are blind have an ever-present 
awareness of how long they have left to live and can only hear how far their opponent is away, but not even the 
direction.

