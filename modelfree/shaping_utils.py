import functools
import numpy as np



#TODO Everything in this file is currently unused.  Was origonally built to shape reward in soccer and ant, re-add this

class NoRewardEnvWrapper(object):

    def __init__(self, env):
        self._env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space


    def step(self, actions):
        observations, _, done, infos = self._env.step(actions)
        return observations, 0, done, infos

    def reset(self):
        return self._env.reset()

    def set_shape_weight(self, n):
        return self._env.set_shape_weight(n)


def get_reward_wrapper(rewards=None):
    if rewards is None or not rewards:
        raise(Exception("Gave no reward function for agent.  It has no purpose :( "))

    if "their_win_loss" in rewards and "their_shape" in rewards:
        raise(Exception("specify either win/loss or shaped (which includes win/loss)"))
    elif "their_win_loss" not in rewards and "their_shaped" not in rewards:
        rewards.append("not_their_win_loss")
    elif "their_shaped" in rewards:
        rewards.remove("their_shaped")
    else:
        rewards.append("not_their_shape")
        rewards.remove("their_win_loss")

    return functools.partial(shape_reward, rewards)


def shape_reward(rewards=None, env=None):
    if env is None:
        raise(Exception("Env was unspecified, other args were:{}".format([rewards])))

    if rewards is None or not rewards:
        return env

    '''
        Note that when we are using their rewards we modify then recurse and when we are using ours we recurse and then 
        modify.  This is because we have to implement their rewards by modifying the environment while we can implement
        ours with wrappers
    '''

    reward_type = rewards.pop()
    if reward_type == "not_their_win_loss":
        env = NoRewardEnvWrapper(env)

        return shape_reward(rewards=rewards, env=env)
    elif reward_type == "not_their_shape":
        #Note that for this to work, this has to be the last reward_type in the list... (which it is by construction...)
        env.set_shape_weight(0)
        return shape_reward(rewards=rewards, env=env)

    else:
        env = shape_reward(rewards=rewards, env=env)
        args = reward_type.split("~")
        if len(args) != 4:
            raise (Exception("Unknown reward type {}".format(reward_type)))

        name = args[0]
        shape_style = args[1]
        const = float(args[2])
        cutoff = args[3]

        shapeing_functions = {
            "me_mag": me_mag,
            "me_pos_mag": me_pos_mag,
            "opp_mag": opp_mag,
            "opp_pos_mag": opp_pos_mag,
            "me_pos": me_pos_shape,
            "you_pos": you_pos_shape,
            "opp_goalie_pos_mag": opp_goalie_pos_mag,
            "opp_goalie_mag": opp_goalie_mag,
            "opp_mag_human_sumo": opp_mag_human_sumo
        }

        if name not in shapeing_functions:
            raise (Exception("Unknown reward type {}".format(reward_type)))

        return ShapeingWrapper(env, shapeing_functions[name], shape_style, const, cutoff)


class ShapeingWrapper(object):

    def __init__(self, env, shapeing_fun, shape_style, const, cutoff):
        self._env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.shapeing_fun = shapeing_fun
        self.const = const
        self.shape_style = shape_style
        self.last_obs = None
        self.cutoff = cutoff

    def step(self, actions):
        observations, rewards, done, infos = self._env.step(actions)

        delta = self.shapeing_fun(observations, self.last_obs)
        points = np.linalg.norm(delta, ord=float(self.shape_style))

        if self.cutoff == "smooth":
            rewards += points * self.const
        else:
            cutoff_float = float(self.cutoff)
            if points > cutoff_float:
                rewards += self.const

        self.last_obs = observations

        return observations, rewards, done, infos

    def reset(self):
        self.last_obs = None
        observations = self._env.reset()
        return observations


def me_pos_shape(obs, last_obs):
    x_me = obs[0]
    y_me = obs[1]
    z_me = obs[2]

    return [x_me, y_me]


def you_pos_shape(obs, last_obs):

        x_opp = obs[-29]
        y_opp = obs[-28]
        z_opp = obs[-27]

        return [x_opp, y_opp]

def me_pos_mag(obs, last_obs):
    if last_obs is not None:

        last_me_pos = last_obs[0:3]
        cur_me_pos = obs[0:3]
        me_delta = cur_me_pos - last_me_pos

        #multiply by 2 to get units in body diameter
        return me_delta * 2
    return [0]

def me_pos_mag(obs, last_obs):
    if last_obs is not None:

        last_me_pos = last_obs[0:3]
        cur_me_pos = obs[0:3]
        me_delta = cur_me_pos - last_me_pos

        #multiply by 2 to get units in body diameter
        return me_delta * 2
    return [0]

def me_mag(obs, last_obs):
    if last_obs is not None:

        last_me_pos = last_obs[0:30]
        cur_me_pos = obs[0:30]
        me_delta = cur_me_pos - last_me_pos

        return me_delta
    return [0]

#TODO THis probably isnt right?  Shoud be -30:-27?
def opp_pos_mag(obs, last_obs):
    if last_obs is not None:
        last_opp_pos = last_obs[-3:]
        cur_opp_pos = obs[-3:]
        opp_delta = cur_opp_pos - last_opp_pos

        # multiply by 2 to get units in body diameter
        return opp_delta * 2
    return [0]


def opp_mag(obs, last_obs):
    if last_obs is not None:
        last_opp_pos = last_obs[-30:]
        cur_opp_pos = obs[-30:]
        opp_delta = cur_opp_pos - last_opp_pos

        return opp_delta
    return [0]

def opp_mag_human_sumo(obs, last_obs):
    if last_obs is not None:
        last_opp_pos = last_obs[-35:]
        cur_opp_pos = obs[-35:]
        opp_delta = cur_opp_pos - last_opp_pos

        return opp_delta
    return [0]

def opp_goalie_mag(obs, last_obs):
    if last_obs is not None:
        last_opp_pos = last_obs[-24:]
        cur_opp_pos = obs[-24:]
        opp_delta = cur_opp_pos - last_opp_pos

        return opp_delta
    return [0]

def opp_goalie_pos_mag(obs, last_obs):
    if last_obs is not None:
        last_opp_pos = last_obs[-24:-22]
        cur_opp_pos = obs[-24:-22]
        opp_delta = cur_opp_pos - last_opp_pos

        return opp_delta
    return [0]


#TODO this is a hack to get around the wrappers and still be able to change the shape weight of the origonal env
class ShapeWeightHack(object):
    def __init__(self, env):
        """
        """
        self._env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space

    def step(self, action):
        return self._env.step(action)

    def reset(self):
        return self._env.reset()

    def render(self):
        self._env.render()

    def set_shape_weight(self, n):
        self._env.move_reward_weight = 0
