import argparse
import numpy as np
import pickle
import os
import os.path as osp
import functools

from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.ppo2 import ppo2
import gym
from gym.core import Wrapper
from gym.monitoring.video_recorder import VideoRecorder
import tensorflow as tf

import utils
from random_search import constant_agent_sampler
from rl_baseline import StatefulModel
from simulation_utils import simulate
from utils import load_agent, LSTMPolicy, Agent, Gymify, MultiToSingle, CurryEnv
from utils import get_env_and_policy_type, get_trained_sumo_ant_locations, make_session, get_trained_kicker_locations

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

#I copied this over to avoid possible merge conflicts, the other one should be removed T
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

def get_agent_any_type(type_opps, name, policy_type, env):
    if type_opps == "zoo":
        return load_agent(name, policy_type,"zoo_ant_policy_2", env, 1)
    elif type_opps == "const":
        trained_agent = constant_agent_sampler()
        trained_agent.load(name)
        return trained_agent
    elif type_opps == "lstm":
        policy = LSTMPolicy(scope="agent_new", reuse=False,
                            ob_space=env.observation_space.spaces[0],
                            ac_space=env.action_space.spaces[0],
                            hiddens=[128, 128], normalize=True)

        def get_action(observation):
            return policy.act(stochastic=True, observation=observation)[0]

        trained_agent = Agent(get_action, policy.reset)

        with open(name, "rb") as file:
            values_from_save = pickle.load(file)

        for key, value in values_from_save.items():
            var = tf.get_default_graph().get_tensor_by_name(key)
            sess.run(tf.assign(var, value))

        return trained_agent
    elif type_opps == "our_mlp":
        #TODO DO ANYTHING BUT THIS.  THIS IS VERY DIRTY AND SAD :(
        def make_env(id):
            # TODO: seed (not currently supported)
            # TODO: VecNormalize? (typically good for MuJoCo)
            # TODO: baselines logger?
            # TODO: we're loading identical policy weights into different
            # variables, this is to work-around design choice of Agent's
            # having state stored inside of them.
            sess = utils.make_session()
            with sess.as_default():
                multi_env=env

                attacked_agent = constant_agent_sampler(act_dim=8, magnitude = 100)

                single_env = Gymify(MultiToSingle(CurryEnv(multi_env, attacked_agent)))
                single_env.spec = gym.envs.registration.EnvSpec('Dummy-v0')

                # TODO: upgrade Gym so don't have to do thi0s
                single_env.observation_space.dtype = np.dtype(np.float32)
            return single_env
            # TODO: close session?


        #TODO DO NOT EVEN READ THE ABOVE CODE :'(

        denv = SubprocVecEnv([functools.partial(make_env, 0)])

        model = ppo2.learn(network="mlp", env=denv,
                   total_timesteps=1,
                   seed=0,
                   nminibatches=4,
                   log_interval=1,
                   save_interval=1,
                   load_path=name)

        stateful_model = StatefulModel(denv, model)
        trained_agent = utils.Agent(action_selector=stateful_model.get_action,
                                    reseter=stateful_model.reset)

        return trained_agent
    raise(Exception('Agent type unrecognized'))


def evaluate_agent(attacked_agent, type_in, name, policy_type, env, samples, visuals, silent=False):
    trained_agent = get_agent_any_type(type_in, name, policy_type, env)

    agents = [attacked_agent, trained_agent]
    tiecount, wincounts = get_emperical_score(env, agents, samples, render=visuals, silent=silent)

    #print("After {} trials the tiecount was {} and the wincounts were {}".format(samples,
    #                                                                             tiecount, wincounts))
    return tiecount, wincounts


class VideoWrapper(Wrapper):
    def __init__(self, env, directory):
        super(VideoWrapper, self).__init__(env)
        self.directory = osp.abspath(directory)
        os.makedirs(self.directory, exist_ok=True)
        self.episode_id = 0
        self.video_recorder = None

    def _step(self, action):
        obs, rew, done, info = self.env.step(action)
        if all(done):
            winners = [i for i, d in enumerate(info) if 'winner' in d]
            metadata = {'winners': winners}
            self._reset_video_recorder(metadata)
        self.video_recorder.capture_frame()
        return obs, rew, done, info

    def _reset(self):
        self._reset_video_recorder()
        self.episode_id += 1
        return self.env.reset()

    def _reset_video_recorder(self, metadata=None):
        if self.video_recorder:
            if metadata is not None:
                self.video_recorder.metadata.update(metadata)
            self.video_recorder.close()
        self.video_recorder = VideoRecorder(
            env=self.env,
            base_path=osp.join(self.directory, 'video.{:06}'.format(self.episode_id)),
            metadata={'episode_id': self.episode_id},
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Collecting Win/Loss/Tie Statistics against ant_pats[1]")
    p.add_argument("--env", default="sumo-ants", type=str)
    p.add_argument("--samples", default=0, help="max number of matches during visualization", type=int)
    p.add_argument("--agent_to_eval", default=None, help="True if you should run last best", type=str)
    p.add_argument("--agent_type", default="zoo", help="Either zoo, const, lstm or matrix, our_mlp", type=str)
    p.add_argument("--all", default=False, help="run evaluation on all of the default agents", type=bool)
    p.add_argument("--no_visuals", type=bool)
    p.add_argument("--save-video", type=str, default="")
    p.add_argument("--nearly_silent", type=bool, default=False)
    configs = p.parse_args()

    env, policy_type = get_env_and_policy_type(configs.env)
    if configs.save_video:
        env = VideoWrapper(env, configs.save_video)

    pretrained_agent = utils.get_trained_agent(configs.env)

    sess = make_session()
    with sess:

        #TODO Load Agent should be changed to "load_zoo_agent"


        if not configs.all:
            #ties, win_loss = evaluate_agent(attacked_agent, configs.agent_type, configs.agent_to_eval, policy_type, env,configs.samples,
             #              not configs.no_visuals, silent=configs.nearly_silent)

            trained_agent = get_agent_any_type(configs.agent_type, configs.agent_to_eval, policy_type, env)
            attacked_agent = load_agent(pretrained_agent, policy_type, "zoo_ant_policy4", env, 0)

            agents = [attacked_agent, trained_agent]
            ties, win_loss = get_emperical_score(env, agents, configs.samples, render=not configs.no_visuals, silent=configs.nearly_silent)

            # print("After {} trials the tiecount was {} and the wincounts were {}".format(samples,


            print("[MAGIC NUMBER 87623123] In {} trials {} acheived {} Ties and winrates {}".format(configs.samples, configs.agent_to_eval, ties, win_loss))


        else:
            attacked_agent = load_agent(pretrained_agent, policy_type, "zoo_ant_policy", env, 0)
            trained_agents = {"pretrained": {"agent_to_eval": get_trained_sumo_ant_locations()[3],
                                             "agent_type": "zoo"},
                              "random_const": {"agent_to_eval": "out_random_const.pkl",
                                               "agent_type": "const"},
                              "random_lstm": {"agent_to_eval": "out_lstm_rand.pkl",
                                              "agent_type": "lstm"},
                              "trained_mlp": {"agent_to_eval": "20181129_184109 mlp_train_test",
                                              "agent_type": "our_mlp"}
                              }
            results = {}
            for key, value in trained_agents.items():
                results[key] = evaluate_agent(attacked_agent, value["agent_type"], value["agent_to_eval"], policy_type, env,
                               configs.samples,
                               not configs.no_visuals)

            print()
            print()
            print("{} samples were taken for each agent.".format(configs.samples))
            for key in results.keys():
                print("{} acheived {} Ties and winrates {}".format(key, results[key][0], results[key][1]))


