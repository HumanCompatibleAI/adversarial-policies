from sacred import Experiment
import gym
from gym.core import Wrapper
from gym.monitoring.video_recorder import VideoRecorder
import os
import os.path as osp
from modelfree.ppo_baseline import load_our_mlp

from modelfree.utils import make_session
from sacred.observers import FileStorageObserver

from modelfree.gym_compete_conversion import *
from modelfree.simulation_utils import simulate


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

    # This tells sacred about the intermediate computation so it updates the result as the experiment is running
    _run.result = result

    for i in range(trials):
        winner = anounce_winner(simulate(env, agents, render=render))
        if winner is None:
            result["ties"] = result["ties"] + 1
        else:
            result["wincounts"][winner] = result["wincounts"][winner] +1
        for agent in agents:
            agent.reset()

    return result


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
    agent_b = "outs/20190131_172524 Dummy Exp Name/model.pkl"
    samples = 50
    watch =True
    videos = False
    video_dir = "videos/"


@score_agent_ex.automain
def score_agent(_run, env, agent_a, agent_b, samples, agent_a_type, agent_b_type, watch, videos, video_dir):
    env_object = gym.make(env)

    if videos:
        env_object = VideoWrapper(env_object, video_dir)

    graph_a = tf.Graph()
    sess_a = make_session(graph_a)
    graph_b = tf.Graph()
    sess_b = make_session(graph_b)
    with sess_a:
        with sess_b:
            agent_b_object = get_agent_any_type(agent_b, agent_b_type, env_object, env, sess=sess_b)
            agent_a_object = get_agent_any_type(agent_a, agent_a_type, env_object, env, sess=sess_a)

            agents = [agent_a_object, agent_b_object]

            # TODO figure out how to stop the other thread from crashing when I finish
            return get_emperical_score(_run, env_object, agents, samples, render=watch)

