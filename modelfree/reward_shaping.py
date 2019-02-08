from gym import Wrapper

class RewardShapingEnv(Wrapper):
    """A more direct interface for shaping the reward of the attacking agent."""

    default_shaping_params = {
        #'center_reward' : 1,
        #'ctrl_cost'     : -1,
        #'contact_cost'  : -1,
        #'survive'       : 1,

        # sparse reward field as per gym_compete/new_envs/multi_agent_env:151
        'reward_remaining' : 1,
        # dense reward field as per gym_compete/new_envs/agents/humanoid_fighter:45
        'reward_move' : 1
    }

    # Annealing schedule from the paper
    default_annealer = lambda s: max(0, 1 - s / 500.)

    def __init__(self, env, shaping_params=default_shaping_params,
                 annealer=default_annealer):
        super().__init__(env)
        self.env = env
        self.shaping_params = shaping_params
        self.annealer = annealer
        self.step_num = 0

    def step(self, actions):
        obs, rew, done, infos = self.env.step(actions)
        # replace rew with differently shaped rew
        # victim is agent 0, attacker is agent 1
        shaped_reward = 0
        for rew_type, rew_value in infos[1].items():
            if rew_type not in self.shaping_params:
                continue
            weighted_reward = self.shaping_params[rew_type] * rew_value

            if self.annealer is not None:
                c = self.get_annealed_exploration_reward()
                if rew_type is 'reward_remaining':
                    weighted_reward *= (1 - c)
                elif rew_type is 'reward_move':
                    weighted_reward *= c

            shaped_reward += weighted_reward
        rew = shaped_reward
        self.step_num += 1
        return obs, rew, done, infos

    def get_annealed_exploration_reward(self):
        """
        Returns c (which will be annealed to zero) s.t. the total reward equals
        c * reward_move + (1 - c) * reward_remaining
        """
        c = self.annealer(self.step_num)
        assert 0 <= c <= 1
        return c

