"""Load serialized policies of different types."""

from stable_baselines import PPO2

from aprl.envs.multi_agent import FakeSingleSpacesVec
from modelfree.gym_compete_conversion import load_zoo_agent


def load_stable_baselines(cls):
    def f(path, env, env_name, index):
        denv = FakeSingleSpacesVec(env, agent_id=index)
        return cls.load(path, env=denv)
    return f


AGENT_LOADERS = {
    'zoo': load_zoo_agent,
    'ppo2': load_stable_baselines(PPO2),
}


def load_policy(policy_type, policy_path, env, env_name, index):
    agent_loader = AGENT_LOADERS.get(policy_type)
    if agent_loader is None:
        raise ValueError(f"Unrecognized agent type '{policy_type}'")
    return agent_loader(policy_path, env, env_name, index)
