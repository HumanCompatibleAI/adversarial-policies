"""Train an agent against a fixed victim via PPO, then score the resulting agent."""

from sacred import Experiment

from modelfree.ppo_baseline import ppo_baseline, ppo_baseline_ex
from modelfree.score_agent import score_agent, score_agent_ex

ppo_and_score_ex = Experiment("ppo_and_score", ingredients=[ppo_baseline_ex, score_agent_ex])


@ppo_and_score_ex.automain
def ppo_and_score():
    model_path = ppo_baseline()
    return score_agent(agent_b_type='mlp', agent_b_path=model_path)
