"""Train an agent against a fixed victim via PPO, then score the resulting agent."""

from sacred import Experiment
from sacred.observers import FileStorageObserver

from modelfree.score_agent import score_agent, score_agent_ex
from modelfree.train import train, train_ex

train_and_score = Experiment("train_and_score", ingredients=[train_ex, score_agent_ex])
train_and_score.observers.append(FileStorageObserver.create("data/sacred"))


@train_and_score.automain
def ppo_and_score():
    model_path = train()
    return score_agent(agent_b_type='ppo2', agent_b_path=model_path)
