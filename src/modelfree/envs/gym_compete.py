from collections import Counter, OrderedDict, defaultdict
import logging
import os
import pickle
import pkgutil

from gym import Wrapper
from gym_compete.policy import LSTMPolicy, MlpPolicyValue
import tensorflow as tf

from aprl.envs.multi_agent import MultiAgentEnv, VecMultiWrapper
from modelfree.common.transparent import TransparentPolicy
from modelfree.common.utils import PolicyToModel, make_session

pylog = logging.getLogger('modelfree.envs.gym_compete_conversion')

POLICY_STATEFUL = OrderedDict([
    ('KickAndDefend-v0', True),
    ('RunToGoalAnts-v0', False),
    ('RunToGoalHumans-v0', False),
    ('SumoAnts-v0', True),
    ('SumoHumans-v0', True),
    ('YouShallNotPassHumans-v0', False),
])

NUM_ZOO_POLICIES = defaultdict(lambda: 1)
NUM_ZOO_POLICIES.update({
    'SumoHumans-v0': 3,
    'SumoAnts-v0': 4,
    'KickAndDefend-v0': 3,
})

SYMMETRIC_ENV = OrderedDict([
    ('KickAndDefend-v0', False),
    ('RunToGoalAnts-v0', True),
    ('RunToGoalHumans-v0', True),
    ('SumoAnts-v0', True),
    ('SumoHumans-v0', True),
    ('YouShallNotPassHumans-v0', False),
])


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


class TransparentLSTMPolicy(TransparentPolicy, LSTMPolicy):
    """gym_compete LSTMPolicy which is also transparent."""
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, transparent_params,
                 hiddens=None, scope="input", reuse=False, normalize=False):
        LSTMPolicy.__init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, hiddens,
                            scope, reuse, normalize)
        TransparentPolicy.__init__(self, transparent_params)

    def step_transparent(self, obs, state=None, mask=None, deterministic=False):
        action, value, state, neglogp, ff = self.step(obs, state, mask, deterministic,
                                                      extra_op=self.ff_out)
        # 'hid' is the hidden state of policy which is the last of the four state vectors
        transparency_dict = self._get_default_transparency_dict(obs, ff, hid=state[:, -1, :])
        return action, value, state, neglogp, transparency_dict


class TransparentMLPPolicyValue(TransparentPolicy, MlpPolicyValue):
    """gym_compete MlpPolicyValue which is also transparent."""
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, transparent_params,
                 hiddens=None, scope="input", reuse=False, normalize=False):
        MlpPolicyValue.__init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch,
                                hiddens=hiddens, scope=scope, reuse=reuse, normalize=normalize)
        TransparentPolicy.__init__(self, transparent_params)

    def step_transparent(self, obs, state=None, mask=None, deterministic=False):
        action, value, state, neglogp, ff = self.step(obs, state, mask, deterministic,
                                                      extra_op=self.ff_out)
        transparency_dict = self._get_default_transparency_dict(obs, ff, hid=None)
        return action, value, self.initial_state, neglogp, transparency_dict


def env_name_to_canonical(env_name):
    env_aliases = {
        'multicomp/SumoHumansAutoContact-v0': 'multicomp/SumoHumans-v0',
        'multicomp/SumoAntsAutoContact-v0': 'multicomp/SumoAnts-v0',
    }
    env_name = env_aliases.get(env_name, env_name)
    env_prefix, env_suffix = env_name.split('/')
    if env_prefix != 'multicomp':
        raise ValueError(f"Unsupported env '{env_name}'; must start with multicomp")
    return env_suffix


def is_stateful(env_name):
    return POLICY_STATEFUL[env_name_to_canonical(env_name)]


def num_zoo_policies(env_name):
    return NUM_ZOO_POLICIES[env_name_to_canonical(env_name)]


def is_symmetric(env_name):
    return SYMMETRIC_ENV[env_name_to_canonical(env_name)]


def get_policy_type_for_zoo_agent(env_name, transparent_params):
    """Determines the type of policy gym_complete used in each environment.
    :param env_name: (str) the environment of the policy we want to load
    :return: a tuple (cls, kwargs) -- call cls(**kwargs) to create policy."""
    canonical_env = env_name_to_canonical(env_name)
    transparent_lstm = (TransparentLSTMPolicy, {'normalize': True,
                                                'transparent_params': transparent_params})
    transparent_mlp = (TransparentMLPPolicyValue, {'normalize': True,
                                                   'transparent_params': transparent_params})
    if canonical_env in POLICY_STATEFUL:
        return transparent_lstm if POLICY_STATEFUL[canonical_env] else transparent_mlp
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
    canonical_env = env_name_to_canonical(env_name)
    dir = os.path.join('agent_zoo', canonical_env)

    if is_symmetric(env_name):  # asymmetric version, parameters tagged with agent id
        symmetric_fname = f'agent_parameters-v{tag}.pkl'
        path = os.path.join(dir, symmetric_fname)
        params_pkl = pkgutil.get_data('gym_compete', path)
    else:  # symmetric version, parameters not associated with a specific agent
        asymmetric_fname = f'agent{index + 1}_parameters-v{tag}.pkl'
        path = os.path.join(dir, asymmetric_fname)
        params_pkl = pkgutil.get_data('gym_compete', path)

    pylog.info(f"Loaded zoo parameters from '{path}'")

    return pickle.loads(params_pkl)


def load_zoo_agent(tag, env, env_name, index, transparent_params):
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
            policy_cls, policy_kwargs = get_policy_type_for_zoo_agent(env_name, transparent_params)
            kwargs.update(policy_kwargs)
            policy = policy_cls(**kwargs)

            # Now restore params
            policy.restore(params)

            return PolicyToModel(policy)
