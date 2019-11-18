from collections import deque
from itertools import islice

from stable_baselines.common.vec_env import VecEnvWrapper

from aprl.policies.wrappers import NoisyAgentWrapper
from aprl.training.scheduling import ConditionalAnnealer, ConstantAnnealer, LinearAnnealer

REW_TYPES = set(("sparse", "dense"))


class RewardShapingVecWrapper(VecEnvWrapper):
    """
    A more direct interface for shaping the reward of the attacking agent.
    - shaping_params schema: {'sparse': {k: v}, 'dense': {k: v}, **kwargs}
    """

    def __init__(self, venv, agent_idx, shaping_params, reward_annealer=None):
        super().__init__(venv)
        assert shaping_params.keys() == REW_TYPES
        self.shaping_params = {}
        for rew_type, params in shaping_params.items():
            for rew_term, weight in params.items():
                self.shaping_params[rew_term] = (rew_type, weight)

        self.reward_annealer = reward_annealer
        self.agent_idx = agent_idx
        queue_keys = REW_TYPES.union(["length"])
        self.ep_logs = {k: deque([], maxlen=10000) for k in queue_keys}
        self.ep_logs["total_episodes"] = 0
        self.ep_logs["last_callback_episode"] = 0
        self.step_rew_dict = {
            rew_type: [[] for _ in range(self.num_envs)] for rew_type in REW_TYPES
        }

    def log_callback(self, logger):
        """Logs various metrics. This is given as a callback to PPO2.learn()"""
        num_episodes = self.ep_logs["total_episodes"] - self.ep_logs["last_callback_episode"]
        if num_episodes == 0:
            return

        means = {}
        for rew_type in REW_TYPES:
            if len(self.ep_logs[rew_type]) < num_episodes:
                raise AssertionError(f"Data missing in ep_logs for {rew_type}")
            rews = islice(self.ep_logs[rew_type], num_episodes)
            means[rew_type] = sum(rews) / num_episodes
            logger.logkv(f"shaping/ep{rew_type}mean", means[rew_type])

        overall_mean = _anneal(means, self.reward_annealer)
        logger.logkv("shaping/eprewmean_true", overall_mean)
        c = self.reward_annealer()
        logger.logkv("shaping/rew_anneal_c", c)
        self.ep_logs["last_callback_episode"] = self.ep_logs["total_episodes"]

    def get_logs(self):
        """Interface to access self.ep_logs which contains data about episodes"""
        if self.ep_logs["total_episodes"] == 0:
            return None
        # keys: 'dense', 'sparse', 'length', 'total_episodes', 'last_callback_episode'
        return self.ep_logs

    def reset(self):
        return self.venv.reset()

    def step_wait(self):
        obs, rew, done, infos = self.venv.step_wait()
        for env_num in range(self.num_envs):
            # Compute shaped_reward for each rew_type
            shaped_reward = {k: 0 for k in REW_TYPES}
            for rew_term, rew_value in infos[env_num][self.agent_idx].items():
                if rew_term not in self.shaping_params:
                    continue
                rew_type, weight = self.shaping_params[rew_term]
                shaped_reward[rew_type] += weight * rew_value

            # Compute total shaped reward, optionally annealing
            rew[env_num] = _anneal(shaped_reward, self.reward_annealer)

            # Log the results of an episode into buffers and then pass on the shaped reward
            for rew_type, val in shaped_reward.items():
                self.step_rew_dict[rew_type][env_num].append(val)

            if done[env_num]:
                ep_length = max(len(self.step_rew_dict[k]) for k in self.step_rew_dict.keys())
                self.ep_logs["length"].appendleft(ep_length)  # pytype:disable=attribute-error
                for rew_type in REW_TYPES:
                    rew_type_total = sum(self.step_rew_dict[rew_type][env_num])
                    rew_type_logs = self.ep_logs[rew_type]
                    rew_type_logs.appendleft(rew_type_total)  # pytype:disable=attribute-error
                    self.step_rew_dict[rew_type][env_num] = []
                self.ep_logs["total_episodes"] += 1
        return obs, rew, done, infos


def apply_reward_wrapper(single_env, shaping_params, agent_idx, scheduler):
    if "metric" in shaping_params:
        rew_shape_annealer = ConditionalAnnealer.from_dict(shaping_params, get_logs=None)
        scheduler.set_conditional("rew_shape")
    else:
        anneal_frac = shaping_params.get("anneal_frac")
        if anneal_frac is not None:
            rew_shape_annealer = LinearAnnealer(1, 0, anneal_frac)
        else:
            # In this case, we weight the reward terms as per shaping_params
            # but the ratio of sparse to dense reward remains constant.
            rew_shape_annealer = ConstantAnnealer(0.5)

    scheduler.set_annealer("rew_shape", rew_shape_annealer)
    return RewardShapingVecWrapper(
        single_env,
        agent_idx=agent_idx,
        shaping_params=shaping_params["weights"],
        reward_annealer=scheduler.get_annealer("rew_shape"),
    )


def apply_embedded_agent_wrapper(embedded, noise_params, scheduler):
    if "metric" in noise_params:
        noise_annealer = ConditionalAnnealer.from_dict(noise_params, get_logs=None)
        scheduler.set_conditional("noise")
    else:
        noise_anneal_frac = noise_params.get("anneal_frac", 0)
        noise_param = noise_params.get("param", 0)

        if noise_anneal_frac <= 0:
            msg = "victim_noise_params.anneal_frac must be >0 if using a NoisyAgentWrapper."
            raise ValueError(msg)
        noise_annealer = LinearAnnealer(noise_param, 0, noise_anneal_frac)
    scheduler.set_annealer("noise", noise_annealer)
    return NoisyAgentWrapper(embedded, noise_annealer=scheduler.get_annealer("noise"))


def _anneal(reward_dict, reward_annealer):
    c = reward_annealer()
    assert 0 <= c <= 1
    sparse_weight = 1 - c
    dense_weight = c
    return reward_dict["sparse"] * sparse_weight + reward_dict["dense"] * dense_weight
