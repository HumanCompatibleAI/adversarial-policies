"""Load serialized policies of different types."""

from baselines.ppo2 import ppo2
import gym

from aprl.envs.multi_agent import CurryVecEnv, FlattenSingletonVecEnv, make_dummy_vec_multi_env
from modelfree.gym_compete_conversion import GymCompeteToOurs, load_zoo_agent
from modelfree.utils import StatefulModel, ZeroPolicy


def load_baselines_mlp(agent_name, env, env_name, _, sess):
    # TODO: Find a way of loading a policy without training for one timestep.
    def make_env():
        multi_env = gym.make(env_name)
        multi_env = GymCompeteToOurs(multi_env)
        return multi_env

    denv = make_dummy_vec_multi_env([make_env] * env.num_envs)
    agent = ZeroPolicy(denv.action_space.spaces[0].shape)
    denv = CurryVecEnv(denv, agent, agent_idx=0)
    denv = FlattenSingletonVecEnv(denv)

    with sess.as_default():
        with sess.graph.as_default():
            model = ppo2.learn(network="mlp", env=denv,
                               total_timesteps=0,
                               seed=0,
                               nminibatches=4,
                               log_interval=1,
                               save_interval=1,
                               load_path=agent_name)

    return StatefulModel(model, sess)


AGENT_LOADERS = {
    "zoo": load_zoo_agent,
    "mlp": load_baselines_mlp,
}


def get_agent_any_type(agent, agent_type, env, env_name, index, sess=None):
    agent_loader = AGENT_LOADERS.get(agent_type)
    if agent_loader is None:
        raise ValueError(f"Unrecognized agent type '{agent_type}'")
    return agent_loader(agent, env, env_name, index, sess=sess)
