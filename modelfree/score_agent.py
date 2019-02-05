import os
import os.path as osp

import gym
from gym.core import Wrapper
from gym.monitoring.video_recorder import VideoRecorder
from sacred import Experiment
from sacred.observers import FileStorageObserver
import tensorflow as tf

from modelfree.gym_compete_conversion import announce_winner, load_zoo_agent
from modelfree.ppo_baseline import load_our_mlp
from modelfree.simulation_utils import simulate
from modelfree.utils import make_session


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


def get_emperical_score(_run, env, agents, trials, render=False):
    result = {
        "ties": 0,
        "wincounts": [0] * len(agents)
    }

    # This tells sacred about the intermediate computation so it
    # updates the result as the experiment is running
    _run.result = result

    for i in range(trials):
        winner = announce_winner(simulate(env, agents, render=render))
        if winner is None:
            result["ties"] = result["ties"] + 1
        else:
            result["wincounts"][winner] = result["wincounts"][winner] + 1
        for agent in agents:
            agent.reset()

    return result


def get_agent_any_type(agent, agent_type, env, env_name, index, sess=None):
    agent_loaders = {
        "zoo": load_zoo_agent,
        "our_mlp": load_our_mlp
    }
    return agent_loaders[agent_type](agent, env, env_name, index, sess=sess)


score_agent_ex = Experiment('score_agent')
score_agent_ex.observers.append(FileStorageObserver.create('my_runs'))


@score_agent_ex.config
def default_score_config():
    agent_a = "agent-zoo/sumo/ants/agent_parameters-v1.pkl"
    agent_a_type = "zoo"
    env = "sumo-ants-v0"
    agent_b_type = "our_mlp"
    agent_b = "outs/20190131_172524 Dummy Exp Name/model.pkl"
    samples = 50
    watch = True
    videos = False
    video_dir = "videos/"
    return locals()  # not needed by sacred, but supresses unused variable warning


@score_agent_ex.automain
def score_agent(_run, env, agent_a, agent_b, agent_a_type, agent_b_type,
                samples, watch, videos, video_dir):
    env_object = gym.make(env)

    if videos:
        env_object = VideoWrapper(env_object, video_dir)

    agents = [agent_a, agent_b]
    agent_types = [agent_a_type, agent_b_type]
    graphs = [tf.Graph() for _ in agents]
    sessions = [make_session(graph) for graph in graphs]
    with sessions[0], sessions[1]:
        zipped = zip(agents, agent_types, sessions)
        agent_objects = [get_agent_any_type(agent, agent_type, env_object, env, i, sess=sess)
                         for i, (agent, agent_type, sess) in enumerate(zipped)]

        # TODO figure out how to stop the other thread from crashing when I finish
        return get_emperical_score(_run, env_object, agent_objects, samples, render=watch)
