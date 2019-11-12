from collections import defaultdict, namedtuple
import pickle

import gym
import numpy as np
from stable_baselines.common.vec_env import VecEnvWrapper

from aprl.common.mujoco import MujocoState, ResettableEnv
from aprl.envs.multi_agent import (FlattenSingletonVecEnv, MultiWrapper, make_dummy_vec_multi_env,
                                   make_subproc_vec_multi_env)
from aprl.envs.wrappers import make_env
from aprl.policies.loader import load_policy
from aprl.training.victim_envs import EmbedVictimWrapper

LookbackTuple = namedtuple('LookbackTuple', ['venv', 'data'])


class LookbackRewardVecWrapper(VecEnvWrapper):
    """Retains information about episodes and rollouts for use in k-lookback whitebox attacks"""
    def __init__(self, venv, env_name, use_debug, victim_index, victim_path, victim_type,
                 transparent_params, lb_mul, lb_num, lb_path, lb_type):
        super().__init__(venv)
        self.lb_num = lb_num
        self.lb_mul = lb_mul
        if transparent_params is None:
            raise ValueError("LookbackRewardVecWrapper assumes transparent policies and venvs.")
        self.transparent_params = transparent_params
        self.victim_index = victim_index

        self._policy = load_policy(lb_type, lb_path, self.venv.unwrapped, env_name,
                                   1 - victim_index, transparent_params=None)
        self._action = None
        self._obs = None
        self._state = None
        self._new_lb_state = None
        self._dones = [False] * self.num_envs
        self.ep_lens = np.zeros(self.num_envs).astype(int)
        self.lb_tuples = self._create_lb_tuples(env_name, use_debug, victim_index,
                                                victim_path, victim_type)
        self.use_debug = use_debug
        if self.use_debug:
            # create a debug file for this venv and also every lookback venv ordinally
            self.debug_files = [open(f'debug{i}.pkl', 'wb') for i in range(self.lb_num + 1)]
            self.get_debug_venv().set_debug_file(self.debug_files[0])

    def _create_lb_tuples(self, env_name, use_debug, victim_index, victim_path, victim_type):
        """Create lookback data structures which are used to compare our episode rollouts against
        those of an environment where a lookback base policy acted instead.

        params victim_index, victim_path, victim_type are the same as in policy_loader.load_policy
        :param use_debug (bool): Use DummyVecEnv instead of SubprocVecEnv
        :return: (list<LookbackTuple>) lb_tuples
        """
        def env_fn(i):
            return make_env(env_name, 0, i, out_dir='data/lookbacks/',
                            pre_wrappers=[OldMujocoResettableWrapper])
        lb_tuples = []
        for _ in range(self.lb_num):
            make_vec_env = make_dummy_vec_multi_env if use_debug else make_subproc_vec_multi_env
            multi_venv = make_vec_env([lambda: env_fn(i) for i in range(self.num_envs)])
            if use_debug:
                multi_venv = DebugVenv(multi_venv)

            victim = load_policy(policy_path=victim_path, policy_type=victim_type, env=multi_venv,
                                 env_name=env_name, index=victim_index,
                                 transparent_params=self.transparent_params)

            multi_venv = EmbedVictimWrapper(multi_env=multi_venv, victim=victim,
                                            victim_index=victim_index,
                                            transparent=True, deterministic=True)

            single_venv = FlattenSingletonVecEnv(multi_venv)
            data_dict = {'state': None, 'action': None, 'info': defaultdict(dict)}
            lb_tuples.append(LookbackTuple(venv=single_venv, data=data_dict))
        return lb_tuples

    def step_async(self, actions):
        # cycle the lb_tuples. The previously last one will now be a new lookback branch.
        self.lb_tuples = [self.lb_tuples[-1]] + self.lb_tuples[:-1]
        new_lb_tuple = self.lb_tuples[0]

        # reassign debug files based on new lb_tuple ordering
        if self.use_debug:
            for i in range(self.lb_num):
                self.lb_tuples[i].venv.set_debug_file(self.debug_files[i + 1])

        # synchronize the observation of the victim in the new lookback venv.
        # then, synchronize the mujoco state (qpos, qvel, qacc) for each individual env.
        self._sync_curry_venvs(new_lb_tuple, env_idx=None)
        current_states = self.venv.unwrapped.env_method('get_state')
        for env_idx in range(self.num_envs):
            new_lb_tuple.venv.unwrapped.env_method('set_state', current_states[env_idx],
                                                   indices=env_idx, forward=False)

        # the state of the lookback policy in the new lookback venv is the one from having
        # always seen the most recent observation
        lb_action, self._new_lb_state = self._policy.predict(self._obs, state=self._new_lb_state,
                                                             mask=self._dones, deterministic=True)
        new_lb_tuple.data['state'] = self._new_lb_state
        # step_async the new lookback venv's action
        new_lb_tuple.venv.step_async(lb_action)

        # for all of the other lookback venvs, step_async their cached action
        # which was calculated in step_wait -> _process_lb_data in the prior timestep.
        for i, lb_tuple in enumerate(self.lb_tuples[1:]):
            lb_tuple.venv.step_async(lb_tuple.data['action'])

        # finally, step_async our own actions.
        self.venv.step_async(actions)
        self.ep_lens += 1

    def step_wait(self):
        # collect and process our own data
        observations, rewards, self._dones, infos = self.venv.step_wait()
        self._process_own_obs(observations)
        # reset/validate episode lengths
        self.ep_lens *= ~np.array(self._dones)

        # collect and process data for the lookback venvs
        lb_data = [lb_tuple.venv.step_wait() for lb_tuple in self.lb_tuples]
        self._process_lb_data(lb_data)

        for env_idx in range(self.num_envs):
            if self._dones[env_idx]:
                # synchronize this env with self.venv since self.venv was reset
                self._reset_lb_data(observations, env_idx)

            # cannot have more lookback venvs than timesteps taken in episode
            valid_lb_tuples = self.lb_tuples[:self.ep_lens[env_idx]]
            victim_info = infos[env_idx][self.victim_index]
            env_diff_reward = 0

            # lookback comparison loop
            for i, lb_tuple in enumerate(valid_lb_tuples):
                lb_victim_info = lb_tuple.data['info'][env_idx][self.victim_index]
                # reward our agent for producing differences between our venv and lookbacks
                for key in self.transparent_params:
                    diff_ff = victim_info[key] - lb_victim_info[key]  # typically ff_policy
                    env_diff_reward += np.linalg.norm(diff_ff)

            # reward shaping
            rewards[env_idx] += self.lb_mul * env_diff_reward
        return observations, rewards, self._dones, infos

    def reset(self):
        # process our observations and then synchronize lookbacks with our venv.
        observations = self.venv.reset()
        self._process_own_obs(observations)
        self._reset_lb_data(observations)
        return observations

    def _process_own_obs(self, observations):
        """Record action, state and observations of our policy

        :param observations ([float]) observations from self.venv
        :return: None
        """
        self._obs = self._get_truncated_obs(observations)
        self._action, self._state = self._policy.predict(self._obs, state=self._state,
                                                         mask=self._dones, deterministic=True)

    def _process_lb_data(self, lb_data):
        """Record action and state of lookback policy

        :param lb_data: list of (observations, rewards, dones, infos), one for each lb_tuple
        :return: None
        """
        for idx, (lb_obs, lb_reward, _, lb_info) in enumerate(lb_data):
            # prepare observation and state and then get next timestep's action and state
            lb_obs = self._get_truncated_obs(lb_obs)
            lb_tuple = self.lb_tuples[idx]
            input_state = lb_tuple.data['state']
            lb_action, lb_state = self._policy.predict(lb_obs, state=input_state,
                                                       mask=self._dones, deterministic=True)

            # update lb_tuple[idx].data since this data will be used elsewhere
            lb_tuple.data['action'] = lb_action
            lb_tuple.data['state'] = lb_state
            for env_idx in range(self.num_envs):
                lb_tuple.data['info'][env_idx].update(lb_info[env_idx])

    def _reset_lb_data(self, initial_observations, env_idx=None):
        """Reset lb_venv states when self.venv resets. Also reset data for baseline policy.

        :param initial_observations ([float]) observations from freshly reset self.venv
        :return: None
        """
        # get action and state to synchronize caches for lookback agents
        obs = self._get_truncated_obs(initial_observations)
        action, state = self._policy.predict(obs, state=None, mask=None, deterministic=True)

        # data from our own environment which needs to be set elsewhere
        mj_states, sim_data, radii = self._get_all_mujoco_state_data(env_idx)
        for lb_tuple in self.lb_tuples:
            # synchronize lb_tuple data caches
            if env_idx is None:
                # this branch is only called in self.reset()
                lb_tuple.venv.reset()
                lb_tuple.data['action'] = action
                lb_tuple.data['state'] = state
            else:
                # this gets called when an episode ends in one of the environments
                lb_tuple.data['action'][env_idx] = action[env_idx]
                if state is None:
                    lb_tuple.data['state'] = None
                else:
                    lb_tuple.data['state'][env_idx, :, :] = state[env_idx, :, :]

            # synchronize environment states
            self._sync_curry_venvs(lb_tuple, env_idx)
            envs_iter = range(self.num_envs) if env_idx is None else (env_idx,)
            for i, env_to_set in enumerate(envs_iter):
                lb_venv = lb_tuple.venv.unwrapped
                if radii is not None:
                    lb_venv.env_method('set_radius', radii[i], indices=env_to_set)
                lb_venv.env_method('set_state', mj_states[i], indices=env_to_set, forward=False)
                lb_venv.env_method('set_sim_data', sim_data[i], indices=env_to_set)

    def _sync_curry_venvs(self, lb_tuple, env_idx):
        """Synchronize observation of victim in lookback CurryVecEnv to that of our own victim."""
        our_curry_obs = self.get_curry_obs(env_idx=env_idx)
        lb_tuple.venv.set_curry_obs(our_curry_obs, env_idx=env_idx)

    def _get_truncated_obs(self, obs):
        """Truncate the observation given to self._policy if we are using adversarial noise ball"""
        return obs[:, :self._policy.policy.observation_space.shape[0]]

    def _get_all_mujoco_state_data(self, env_idx):
        """Get qpos+qvel state, full MjData dictionary and environment radius if applicable

        :param venv (VecEnvWrapper) environment from where to gather
        :param env_idx (int, None) indices of environments to get data from
        :return: (list<[float]>, list<dict>, list<float> or None) mj_states, sim_data, radii
        """
        env_methods = ['get_state', 'get_sim_data']
        base_venv = self.venv.unwrapped
        all_data = [base_venv.env_method(s, indices=env_idx) for s in env_methods]
        try:
            all_data.append(base_venv.env_method('get_radius', indices=env_idx))
        except AttributeError:
            # this environment doesn't have a radius
            all_data.append(None)

        mj_states, sim_data, radii = all_data
        return mj_states, sim_data, radii


class OldMujocoResettableWrapper(ResettableEnv, MultiWrapper):
    """Converts a MujocoEnv into a ResettableEnv.

    Specifically designed to handle getting and setting states with Mujoco 1.31 / mujoco_py 0.5.7
    Note all MuJoCo environments are resettable."""
    def __init__(self, env):
        """Wraps a MujocoEnv, adding get_state and set_state methods.
        :param env: a MujocoEnv."""
        gym.Wrapper.__init__(self, env)
        self.sim = env.unwrapped.env_scene

    def get_sim_data(self):
        """Get the contents of an MjData object in dict form for the environment

        :return: (dict<str, [float]) state_dict
        """
        state_dict = {}
        for k, v in type(self.sim.data._wrapped.contents).__dict__['_fields_']:
            if k not in ['contact', 'buffer']:
                state_dict[k] = getattr(self.sim.data, k)
        return state_dict

    def get_radius(self):
        """Get the radius of the environment, assuming it has one.

        :return: (float) radius
        """
        if hasattr(self.env.unwrapped, 'RADIUS'):
            return self.env.unwrapped.RADIUS
        else:
            return None

    def get_state(self):
        """Serializes the qpos and qvel state of the MuJoCo emulator.

        :return: ([float]) state
        """
        state = MujocoState.from_mjdata(self.sim.data).flatten()
        return state

    def set_state(self, x, forward=True):
        """Restores qpos and qvel, calling forward() to derive other values.

        :param forward (bool) whether to call forward on the environment.
        """
        state = MujocoState.from_flattened(x, self.sim)
        state.set_mjdata(self.sim.data)
        if forward:
            self.sim.model.forward()  # put mjData in consistent state

    def set_radius(self, radius):
        """Set the radius of this environment.

        :param radius (float) size of environment, if it exists"""
        self.env.unwrapped.RADIUS = radius
        self.env.unwrapped._set_geom_radius()

    def set_sim_data(self, sim_data):
        """Set the fields of the MjData object of this environment

        :param sim_data (dict<str, [float]) dict with MjData fields extracted using get_state"""
        for k, v in sim_data.items():
            setattr(self.sim.data, k, v)

    def reset(self):
        """See base class."""
        return self.env.reset()

    def step(self, a):
        """See base class."""
        return self.env.step(a)


class DebugVenv(VecEnvWrapper):
    """VecEnvWrapper whose purpose is to record trajectory information for debugging purposes

    :param venv (VecEnv) the environment to wrap
    :param dump_mujoco_state (bool) whether to dump all MjData information (memory intensive)
    """
    def __init__(self, venv, dump_mujoco_state=False):
        super().__init__(venv)
        self.num_agents = self.venv.num_agents
        self.dump_mujoco_state = dump_mujoco_state
        self.debug_file = None
        self.debug_dict = {}

    def step_async(self, actions):
        self.debug_dict['actions'] = actions
        if self.dump_mujoco_state:
            state_data = self.unwrapped.envs[0].env.sim.data
            fields = type(state_data._wrapped.contents).__dict__['_fields_']
            keys = [t[0] for t in fields if t[0] != 'contact']
            for k in keys:
                val = getattr(state_data, k)
                if isinstance(val, np.ndarray) and val.size > 0:
                    self.debug_dict[k] = val

        self.venv.step_async(actions)

    def step_wait(self):
        obs, rew, dones, infos = self.venv.step_wait()
        if self.debug_file is not None:
            self.debug_dict.update({'next_obs': obs, 'rewards': rew})
            pickle.dump(self.debug_dict, self.debug_file)
        self.debug_dict = {}
        return obs, rew, dones, infos

    def reset(self):
        observations = self.venv.reset()
        if self.debug_file is not None:
            self.debug_dict['prev_obs'] = observations
        return observations

    def set_debug_file(self, f):
        """Setter for self.debug_file."""
        self.debug_file = f

    def get_debug_venv(self):
        """Helper method to locate self in a stack of nested VecEnvWrappers"""
        return self
