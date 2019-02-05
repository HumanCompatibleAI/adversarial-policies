from aprl.envs.multi_agent import MultiAgentEnv
from modelfree.simulation_utils import simulate, Agent
import pickle
import numpy as np
import tensorflow as tf
import gym_compete

#TODO see if you can get this from the actual repo and not from us copying the file....
from modelfree.policy import LSTMPolicy, MlpPolicyValue

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


def get_emperical_score(_run, env, agents, trials, render=False):
    result = {
        "ties": 0,
        "wincounts": [0] * len(agents)
    }

    # This tells sacred about the intermediate computation so it updates the result as the experiment is running
    _run.result = result

    for i in range(trials):
        this_result = new_anounce_winner(simulate(env, agents, render=render))
        if result == -1:
            result["ties"] = result["ties"] + 1
        else:
            result["wincounts"][this_result] = result["wincounts"][this_result] +1
        for agent in agents:
            agent.reset()

    return result


def new_anounce_winner(sim_stream):
    for _, _, dones, infos in sim_stream:
        if dones[0]:
            draw = True
            for i in range(len(infos)):
                if 'winner' in infos[i]:
                    draw = False
                    return i
            if draw:
                return -1



def load_from_file(param_pkl_path):
    with open(param_pkl_path, 'rb') as f:
        params = pickle.load(f)
    return params


def get_policy_type_for_agent_zoo(env_name):
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
    policy_type = get_policy_type_for_agent_zoo(env_name)

    with sess.graph.as_default():
            policy = load_zoo_policy(agent, policy_type, "zoo_policy_{}".format(agent), env, index, sess=sess)

    def get_action(observation):
        with sess.as_default():
            return policy.act(stochastic=True, observation=observation)[0]

    return Agent(get_action, policy.reset)