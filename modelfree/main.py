from sacred import Experiment
import gym
import gym_compete
from policy import LSTMPolicy, MlpPolicyValue
import pickle
import tensorflow as tf
import numpy as np
import sys

#TODO Commit this to a new branch

#TODO Some of the functions in here are copied from "main" in the multi-agent repo, and we have our own copy of policy
#TODO Fix import errors
ex = Experiment('score_agent')


#TODO Needs to be changed to post processing
def get_emperical_score(env, agents, trials, render=False, silent=False):
    tiecount = 0
    wincount = [0] * len(agents)
    for _ in range(trials):
        result = new_anounce_winner(simulate(env, agents, render=render), silent=silent)
        if result == -1:
            tiecount += 1
        else:
            wincount[result] += 1
        for agent in agents:
            agent.reset()
    return tiecount, wincount


def new_anounce_winner(sim_stream, silent=False):
    for _, _, dones, infos in sim_stream:

        if dones[0]:
            draw = True
            for i in range(len(infos)):
                if 'winner' in infos[i]:
                    draw = False
                    if not silent:
                        print("Winner: Agent {}, Scores: {}, Total Episodes: {}".format(i, 1,1))
                    return i
            if draw:
                if not silent:
                    print("Game Tied: Agent {}, Scores: {}, Total Episodes: {}".format(i, 1,1))
                return -1












#TODO Needs to be changed to agents without state
#TODO Needs to add wrappers to log the data using sacred
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

    raise Exception("Unsupported Environment, choose from {}".format(envs_policy_types.key()))


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

    policy = load_zoo_policy(agent, policy_type, "zoo_policy_{}".format(agent), env, index, sess=sess)

    def get_action(observation):
        return policy.act(stochastic=True, observation=observation)[0]

    return Agent(get_action, policy.reset)

#######################################################################################################
#######################################################################################################
#######################################################################################################

def make_session():
    tf_config = tf.ConfigProto(
        inter_op_parallelism_threads=1,
        intra_op_parallelism_threads=1)
    tf_config.gpu_options.allow_growth = True
    sess = tf.Session(config=tf_config)
    return sess

#TODO Make loader for our_mlp
def get_agent_any_type(agent, agent_type, env, env_name):
    agent_loaders = {
        "zoo": load_zoo_agent,
        "our_mlp": None
    }
    return agent_loaders[agent_type](agent, env, env_name)


@ex.config
def default_config():
    agent_a = "agent-zoo/sumo/ants/agent_parameters-v1.pkl"
    agent_a_type = "zoo"
    env = "sumo-ants-v0"
    agent_b_type = "zoo"
    agent_b = "agent-zoo/sumo/ants/agent_parameters-v2.pkl"
    samples = 100

@ex.automain
def score_agent(env, agent_a, agent_b, samples, agent_a_type, agent_b_type):
    env_object = gym.make(env)

    sess = make_session()
    with sess:

        agent_a_object = get_agent_any_type(agent_a, agent_a_type, env_object, env)
        agent_b_object = get_agent_any_type(agent_b, agent_b_type, env_object, env)
        agents = [agent_a_object, agent_b_object]

        #TODO record the intermediate details with sacred
        ties, win_loss = get_emperical_score(env_object, agents, samples)

        #TODO Record the ending scores with sacred
