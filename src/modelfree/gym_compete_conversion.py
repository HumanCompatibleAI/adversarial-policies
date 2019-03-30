from collections import Counter
import logging
import os
import pickle
import pkgutil

from gym import Wrapper
from gym_compete.policy import LSTMPolicy, MlpPolicyValue
import tensorflow as tf

from aprl.envs.multi_agent import MultiAgentEnv, VecMultiWrapper
from modelfree.utils import PolicyToModel, make_session

pylog = logging.getLogger('modelfree.gym_compete_conversion')

POLICY_STATEFUL = {
    'KickAndDefend-v0': True,
    'RunToGoalHumans-v0': False,
    'RunToGoalAnts-v0': False,
    'YouShallNotPassHumans-v0': False,
    'SumoHumans-v0': True,
    'SumoAnts-v0': True,
}


class GymCompeteToOurs(Wrapper, MultiAgentEnv):
    """This adapts gym_compete.MultiAgentEnv to our eponymous MultiAgentEnv.

       The main differences are that we have a scalar done (episode-based) rather than vector
       (agent-based), and only return one info dict (property of environment not agent)."""
    def __init__(self, env):
        Wrapper.__init__(self, env)
        MultiAgentEnv.__init__(self, num_agents=2)

    def step(self, action_n):
        observations, rewards, dones, infos = self.env.step(action_n)
        done = any(dones)
        infos = {i: v for i, v in enumerate(infos)}
        return observations, rewards, done, infos

    def reset(self):
        return self.env.reset()


def game_outcome(info):
    draw = True
    for i, agent_info in info.items():
        if 'winner' in agent_info:
            return i
    if draw:
        return None


class GameOutcomeMonitor(VecMultiWrapper):
    def __init__(self, venv):
        super().__init__(venv)
        self.outcomes = []

    def reset(self):
        return self.venv.reset()

    def step_wait(self):
        obs, rew, dones, infos = self.venv.step_wait()
        for done, info in zip(dones, infos):
            if done:
                self.outcomes.append(game_outcome(info))
        return obs, rew, dones, infos

    def log_callback(self, logger):
        c = Counter()
        c.update(self.outcomes)
        num_games = len(self.outcomes)
        if num_games > 0:
            for agent in range(self.num_agents):
                logger.logkv(f"game_win{agent}", c.get(agent, 0) / num_games)
            logger.logkv("game_tie", c.get(None, 0) / num_games)
        logger.logkv("game_total", num_games)
        self.outcomes = []


def _env_name_to_canonical(env_name):
    env_aliases = {
        'multicomp/SumoHumansAutoContact-v0': 'multicomp/SumoHumans-v0'
    }
    env_name = env_aliases.get(env_name, env_name)
    env_prefix, env_suffix = env_name.split('/')
    if env_prefix != 'multicomp':
        raise ValueError(f"Unsupported env '{env_name}'; must start with multicomp")
    return env_suffix


def is_stateful(env_name):
    return POLICY_STATEFUL[_env_name_to_canonical(env_name)]


def get_policy_type_for_zoo_agent(env_name):
    """Determines the type of policy gym_complete used in each environment.
    :param env_name: (str) the environment of the policy we want to load
    :return: a tuple (cls, kwargs) -- call cls(**kwargs) to create policy."""
    canonical_env = _env_name_to_canonical(env_name)
    lstm = (LSTMPolicy, {'normalize': True})
    mlp = (MlpPolicyValue, {'normalize': True})
    if canonical_env in POLICY_STATEFUL:
        return lstm if POLICY_STATEFUL[canonical_env] else mlp
    else:
        msg = f"Unsupported Environment: {canonical_env}, choose from {POLICY_STATEFUL.keys()}"
        raise ValueError(msg)


def load_zoo_agent_params(tag, env_name, index):
    """Loads parameters for the gym_compete zoo agent, but does not restore them.
    :param tag: (str) version of the zoo agent (e.g. '1', '2', '3').
    :param env_name: (str) Gym environment ID
    :param index: (int) the player ID of the agent we want to load ('0' or '1')
    :return a NumPy array of policy weights."""
    # Load parameters
    canonical_env = _env_name_to_canonical(env_name)
    dir = os.path.join('agent_zoo', canonical_env)
    asymmetric_fname = f'agent{index + 1}_parameters-v{tag}.pkl'
    symmetric_fname = f'agent_parameters-v{tag}.pkl'
    try:  # asymmetric version, parameters tagged with agent id
        path = os.path.join(dir, asymmetric_fname)
        params_pkl = pkgutil.get_data('gym_compete', path)
    except OSError:  # symmetric version, parameters not associated with a specific agent
        path = os.path.join(dir, symmetric_fname)
        params_pkl = pkgutil.get_data('gym_compete', path)
    pylog.info(f"Loaded zoo parameters from '{path}'")

    return pickle.loads(params_pkl)


def load_zoo_agent(tag, env, env_name, index):
    """Loads a gym_compete zoo agent.
    :param tag: (str) version of the zoo agent (e.g. '1', '2', '3').
    :param env: (gym.Env) the environment
    :param env_name: (str) Gym environment ID
    :param index: (int) the player ID of the agent we want to load ('0' or '1')
    :return a BaseModel, where predict executes the loaded policy."""
    g = tf.Graph()
    sess = make_session(g)

    with g.as_default():
        with sess.as_default():
            # Load parameters (do this first so fail-fast if tag does not exist)
            params = load_zoo_agent_params(tag, env_name, index)

            # Build policy
            scope = f"zoo_policy_{tag}_{index}"
            kwargs = dict(sess=sess, ob_space=env.observation_space.spaces[index],
                          ac_space=env.action_space.spaces[index], n_env=env.num_envs,
                          n_steps=1, n_batch=env.num_envs, scope=scope, reuse=False)
            policy_cls, policy_kwargs = get_policy_type_for_zoo_agent(env_name)
            kwargs.update(policy_kwargs)
            policy = policy_cls(**kwargs)

            # Now restore params
            policy.restore(params)

            return PolicyToModel(policy)
