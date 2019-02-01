from sacred import Experiment
import gym
import gym_compete
import pickle
import tensorflow as tf
import numpy as np
import sys

from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.vec_normalize import VecNormalize
from baselines.ppo2 import ppo2
from baselines.a2c import utils

import functools
import itertools

from sacred.observers import FileStorageObserver

from aprl.envs.multi_agent import MultiAgentEnv, FlattenSingletonEnv, CurryEnv

#TODO see if you can get this from the actual repo and not from us copying the file....
from policy import LSTMPolicy, MlpPolicyValue

#TODO Some of the functions in here are copied from "main" in the multi-agent repo, and we have our own copy of policy


class TheirsToOurs(MultiAgentEnv):
    def __init__(self, env):
        super().__init__(2, env.action_space.spaces[0], env.observation_space.spaces[0])
        # TODO ask adam about the action/observation spaces.  I am worried about asymetric envs like goalie

        self.spec = env.spec
        self._env = env

    def step(self, action_n):
        observations, rewards, done, infos = self._env.step(action_n)

        observations = list(observations)
        rewards = list(rewards)
        done = list(done)
        # TODO not at all sketch
        infos = {i: v for i, v in enumerate(infos)}

        return observations, rewards, done, infos

    def reset(self):
        return list(self._env.reset())


def get_emperical_score(_run, env, agents, trials, render=False, silent=False):
    result = {
        "ties": 0,
        "wincounts": [0] * len(agents)
    }

    # This tells sacred about the intermediate computation so it updates the result as the experiment is running
    _run.result = result

    for i in range(trials):
        this_result = new_anounce_winner(simulate(env, agents, render=render), silent=silent)
        if result == -1:
            result["ties"] = result["ties"] + 1
        else:
            result["wincounts"][this_result] = result["wincounts"][this_result] +1
        for agent in agents:
            agent.reset()

    return result


def new_anounce_winner(sim_stream, silent=False):
    for _, _, dones, infos in sim_stream:
        if dones[0]:
            draw = True
            for i in range(len(infos)):
                if 'winner' in infos[i]:
                    draw = False
                    return i
            if draw:
                return -1


def simulate(env, agents, render=False):
    """
    Run Environment env with the agents in agents
    :param env: any enviroment following the openai-gym spec
    :param agents: agents that have get-action functions
    :return: streams information about the simulation
    """
    observations = env.reset()
    dones = [False]

    while not dones[0]:
        if render:
            env.render()
        actions = []
        for agent, observation in zip(agents, observations):
            actions.append(agent.get_action(observation))

        observations, rewards, dones, infos = env.step(actions)

        yield observations, rewards, dones, infos


class Agent(object):

    def __init__(self, action_selector, reseter, values=None, sess = None):
        """
        Takes policies from their format to mine
        :param actable: a policy in the format used by mult-agent-compeitition
        """
        self._action_selector = action_selector
        self._reseter = reseter
        self._values = values
        self._sess = sess

    def get_action(self, observation):
        action = self._action_selector(observation)
        return action

    def reset(self):
        return self._reseter()


#TODO Make an Agent_Wrapper class that takes a function from OxM -> AxM and makes an agent.


#######################################################################################################
###                                     Stuff to load their agents                                  ###
#######################################################################################################

def load_from_file(param_pkl_path):
    with open(param_pkl_path, 'rb') as f:
        params = pickle.load(f)
    return params

def get_env_and_policy_type(env_name):
    envs_policy_types = {
        "kick-and-defend-v0": "lstm",
        "run-to-goal-humans-v0": "mlp",
        "run-to-goal-ants-v0": "mlp",
        "you-shall-not-pass-humans-v0": "mlp",
        "sumo-humans-v0": "lstm",
        "sumo-ants-v0": "lstm"
    }

    if env_name in envs_policy_types:
        return envs_policy_types[env_name]

    raise Exception("Unsupported Environment: {}, choose from {}".format(env_name, envs_policy_types.keys()))


def set_from_flat(var_list, flat_params, sess = None):
    shapes = list(map(lambda x: x.get_shape().as_list(), var_list))
    total_size = np.sum([int(np.prod(shape)) for shape in shapes])
    theta = tf.placeholder(tf.float32, [total_size])
    start = 0
    assigns = []
    for (shape, v) in zip(shapes, var_list):
        size = int(np.prod(shape))
        assigns.append(tf.assign(v, tf.reshape(theta[start:start + size], shape)))
        start += size
    op = tf.group(*assigns)
    if sess is None:
        sess = tf.get_default_session()
    sess.run(op, {theta: flat_params})

def load_zoo_policy(file, policy_type, scope, env, index, sess=None):
    if policy_type == "lstm":
        policy = LSTMPolicy(scope=scope, reuse=False,
                            ob_space=env.observation_space.spaces[index],
                            ac_space=env.action_space.spaces[index],
                            hiddens=[128, 128], normalize=True, sess=sess)
    else:
        policy = MlpPolicyValue(scope=scope, reuse=False,
                                ob_space=env.observation_space.spaces[index],
                                ac_space=env.action_space.spaces[index],
                                hiddens=[64, 64], normalize=True, sess=sess)
    set_from_flat(policy.get_variables(), load_from_file(param_pkl_path=file), sess=sess)
    return policy


def load_zoo_agent(agent, env, env_name, index=1, sess=None):
    policy_type = get_env_and_policy_type(env_name)

    with sess.graph.as_default():
            policy = load_zoo_policy(agent, policy_type, "zoo_policy_{}".format(agent), env, index, sess=sess)

    def get_action(observation):
        with sess.as_default():
            return policy.act(stochastic=True, observation=observation)[0]

    return Agent(get_action, policy.reset)

#######################################################################################################
#######################################################################################################
#######################################################################################################



def load_our_mlp(agent_name, env, env_name, sess):
    # TODO DO ANYTHING BUT THIS.  THIS IS VERY DIRTY AND SAD :(
    def make_env(id):
        # TODO: seed (not currently supported)
        # TODO: VecNormalize? (typically good for MuJoCo)
        # TODO: baselines logger?
        # TODO: we're loading identical policy weights into different
        # variables, this is to work-around design choice of Agent's
        # having state stored inside of them.
        sess_inner = make_session()
        with sess_inner.as_default():
            multi_env = env

            attacked_agent = constant_zero_agent(act_dim=8)

            single_env = FlattenSingletonEnv(CurryEnv(TheirsToOurs(multi_env), attacked_agent))

            # TODO: upgrade Gym so don't have to do thi0s
            single_env.observation_space.dtype = np.dtype(np.float32)
        return single_env
        # TODO: close session?

    # TODO DO NOT EVEN READ THE ABOVE CODE :'(
    denv = SubprocVecEnv([functools.partial(make_env, 0)])
    with sess.as_default():
        with sess.graph.as_default():
            model = ppo2.learn(network="mlp", env=denv,
                               total_timesteps=1,
                               seed=0,
                               nminibatches=4,
                               log_interval=1,
                               save_interval=1,
                               load_path=agent_name)

    stateful_model = StatefulModel(denv, model, sess)
    trained_agent = Agent(action_selector=stateful_model.get_action,
                                reseter=stateful_model.reset)

    return trained_agent

def constant_zero_agent(act_dim=8):
    constant_action = np.zeros((act_dim,))


    class Temp():

        def __init__(self, constants):
            self.constants = constants

        def get_action(self, observation):

            return self.constants

        def reset(self):
            pass

        def save(self, filename):
            with open(filename, "wb") as file:
                pickle.dump(self.constants, file, pickle.HIGHEST_PROTOCOL)

        def load(self, filename):
            with open(filename, "rb") as file:
                self.constants = pickle.load(file)

    return Temp(constant_action)


class Policy(object):
    def reset(self, **kwargs):
        pass

    def act(self, observation):
        # should return act, info
        raise NotImplementedError()

class StatefulModel(Policy):
    def __init__(self, env, model, sess):
        self._sess = sess
        self.nenv = env.num_envs
        self.env = env
        self.dones = [False for _ in range(self.nenv)]
        self.model = model
        self.states = model.initial_state

    def get_action(self, observation):
        with self._sess.as_default():
            actions, values, self.states, neglocpacs = self.model.step([observation] * self.nenv, S=self.states, M=self.dones)
            return actions[0]

    def reset(self):
        self._dones = [True] * self.nenv


#######################################################################################################
#######################################################################################################


def make_session(graph=None):
    tf_config = tf.ConfigProto(
        inter_op_parallelism_threads=1,
        intra_op_parallelism_threads=1)
    tf_config.gpu_options.allow_growth = True
    sess = tf.Session(graph=graph, config=tf_config)
    return sess



#TODO Make loader for our_mlp
def get_agent_any_type(agent, agent_type, env, env_name, sess=None):
    agent_loaders = {
        "zoo": load_zoo_agent,
        "our_mlp": load_our_mlp
    }
    return agent_loaders[agent_type](agent, env, env_name, sess=sess)


score_agent_ex = Experiment('score_agent')
score_agent_ex.observers.append(FileStorageObserver.create('my_runs'))

@score_agent_ex.config
def default_score_config():
    agent_a = "agent-zoo/sumo/ants/agent_parameters-v1.pkl"
    agent_a_type = "zoo"
    env = "sumo-ants-v0"
    agent_b_type = "our_mlp"
    agent_b = "outs/20190128_231014 Dummy Exp Name/model.pkl"
    samples = 50

@score_agent_ex.automain
def score_agent(_run, env, agent_a, agent_b, samples, agent_a_type, agent_b_type):
    env_object = gym.make(env)
    graph_a = tf.Graph()
    sess_a = make_session(graph_a)
    graph_b = tf.Graph()
    sess_b = make_session(graph_b)
    with sess_a:
        with sess_b:
            # TODO seperate tensorflow graphs to get either order of the next two statements to work
            agent_b_object = get_agent_any_type(agent_b, agent_b_type, env_object, env, sess=sess_b)
            agent_a_object = get_agent_any_type(agent_a, agent_a_type, env_object, env, sess=sess_a)


            agents = [agent_a_object, agent_b_object]

            # TODO figure out how to stop the other thread from crashing when I finish
            return get_emperical_score(_run, env_object, agents, samples)

