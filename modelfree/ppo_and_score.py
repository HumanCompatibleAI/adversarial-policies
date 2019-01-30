from modelfree.score_agent import *
from modelfree.ppo_baseline import *
from sacred import Experiment

ppo_and_score_ex = Experiment("ppo_and_score")

@ppo_and_score_ex.config
def default_config():
    confs = {
        "ppo": {},
        "score": {}
    }

@ppo_and_score_ex.automain
def ppo_and_score(confs):
    training_results = ppo_baseline_ex.run(config_updates=confs["ppo"])
    confs["score"]["agent_b"] = training_results.result

    return score_agent_ex.run(config_updates=confs["score"])

