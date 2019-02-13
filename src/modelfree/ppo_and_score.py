"""Train an agent against a fixed victim via PPO, then score the resulting agent."""

from sacred import Experiment

from modelfree.ppo_baseline import ppo_baseline_ex
from modelfree.score_agent import score_agent_ex

ppo_and_score_ex = Experiment("ppo_and_score")


@ppo_and_score_ex.config
def default_config():
    config = {  # noqa: F841
        "ppo": {},
        "score": {}
    }


@ppo_and_score_ex.named_config
def ant_score_config():
    config = {  # noqa: F841
        "ppo": {"victim": "/home/neel/multiagent-competition/agent-zoo/" + \
                "sumo/ants/agent_parameters-v1.pkl",
                "total_timesteps": 1e7},
        "score": {"agent_a": "/home/neel/multiagent-competition/agent-zoo/" + \
                  "sumo/ants/agent_parameters-v1.pkl",
                  "watch": False}
    }


@ppo_and_score_ex.automain
def ppo_and_score(config):
    training_results = ppo_baseline_ex.run(config_updates=config["ppo"])
    config["score"]["agent_b"] = training_results.result

    return score_agent_ex.run(config_updates=config["score"])
