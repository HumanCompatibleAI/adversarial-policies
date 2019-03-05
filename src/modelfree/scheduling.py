from abc import ABC, abstractmethod
import collections
import functools
import operator
from stable_baselines import logger


class Scheduler(object):
    """Keep track of frac_remaining and return time-dependent values"""
    def __init__(self, annealer_dict=None):
        if annealer_dict is None:
            annealer_dict = {}
        self.annealer_dict = annealer_dict
        self.func_dict = {}
        for a in annealer_dict:
            self.func_dict[a] = a.get_value

        self.conditionals = collections.defaultdict(lambda: False)
        self.frac_remaining = 1  # frac_remaining goes from 1 to 0

    @staticmethod
    def _validate_func_type(func_type):
        allowed_func_types = frozenset(['lr', 'rew_shape', 'noise'])
        if func_type not in allowed_func_types:
            raise KeyError("func_type not in allowed_func_types")

    def _update_frac_remaining(self, frac_remaining=None):
        if frac_remaining is not None:
            self.frac_remaining = frac_remaining

    def set_conditional(self, func_type):
        """Set the value of key func_type to be True in the dictionary self.conditionals"""
        self._validate_func_type(func_type)
        self.conditionals[func_type] = True

    def get_conditional(self, func_type):
        """Interface with the dictionary self.conditionals"""
        self._validate_func_type(func_type)
        return self.conditionals[func_type]

    def set_annealer_and_func(self, func_type, annealer):
        """
        Common interface for attaching a function to this Scheduler.
        :param func_type: str, must be in self.allowed_func_types
        :param annealer: Annealer, must have a get_value method
        :return: None
        """
        self._validate_func_type(func_type)
        if not isinstance(annealer, Annealer) and annealer is not None:
            raise TypeError('set_func requires an Annealer as input')
        self.annealer_dict[func_type] = annealer
        self.func_dict[func_type] = annealer.get_value

    def get_val(self, val_type, frac_remaining=None):
        """
        Common interface for getting values from functions attached to this Scheduler.
        :param val_type: str, must be in self.allowed_func_types
        :param frac_remaining: optional arg, if given will update the state of Scheduler
        :return: scalar float
        """
        self._validate_func_type(val_type)
        self._update_frac_remaining(frac_remaining)
        return self.func_dict[val_type](self.frac_remaining)

    def get_func(self, func_type):
        """
        Interface for accessing the functions in self.func_dict
        :param func_type: str, must be in self.allowed_func_types
        :return: callable which takes one optional argument - frac_remaining
        """
        self._validate_func_type(func_type)
        if self.func_dict.get(func_type) is None:
            return None
        else:
            return functools.partial(self.get_val, func_type)

    def set_annealer_shaping_env(self, func_type, env):
        """Interface for updating properties of Annealers. Used for setting env for an Annealer"""
        self._validate_func_type(func_type)
        self.annealer_dict[func_type].set_shaping_env(env)


# Annealers

class Annealer(ABC):
    """Abstract class for implementing Annealers."""
    def __init__(self, start_val, end_val):
        self.start_val = start_val
        self.end_val = end_val

    @abstractmethod
    def get_value(self, step):
        raise NotImplementedError()


class ConstantAnnealer(Annealer):
    """Returns a constant value"""
    def __init__(self, const_val, shaping_env=None):
        self.const_val = const_val
        self.shaping_env = shaping_env
        super().__init__(const_val, const_val)

    def get_value(self, frac_remaining):
        return self.const_val


class LinearAnnealer(Annealer):
    """Linearly anneals from start_val to end_val over end_frac fraction of training"""
    def __init__(self, start_val, end_val, end_frac, shaping_env=None):
        super().__init__(start_val, end_val)
        assert 0 <= end_frac <= 1, "Invalid end_frac for LinearAnnealer"
        self.end_frac = end_frac
        self.shaping_env = shaping_env

    def get_value(self, frac_remaining):
        anneal_progress = min(1.0, (1 - frac_remaining) / self.end_frac)
        return (1 - anneal_progress) * self.start_val + anneal_progress * self.end_val


class ConditionalAnnealer(Annealer):
    """Anneal the value depending on some condition."""
    def __init__(self, start_val, end_val, decay_factor, thresh, window_size,
                 min_wait, operator, metric, shaping_env):
        super().__init__(start_val, end_val)
        self.decay_factor = decay_factor
        self.thresh = thresh
        self.window_size = window_size
        self.min_wait = min_wait
        self.operator = operator
        self.metric = metric
        self.shaping_env = shaping_env

        self.current_param_val = start_val
        self.last_num_episodes = 0

    def set_shaping_env(self, shaping_env):
        """Set the environment attribute since passing into constructor may not be possible"""
        self.shaping_env = shaping_env

    @classmethod
    def from_dict(cls, cond_config, shaping_env=None):
        constructor_config = {
            'decay_factor': 0.98,
            'thresh': 0,
            'start_val': 2,
            'end_val': 0,
            'window_size': 1,
            'min_wait': 1,
            'operator': operator.lt,
            'metric': 'epsparsemean',
        }
        if 'operator' in cond_config:
            cond_config['operator'] = getattr(operator, cond_config['operator'])
        cond_config.pop('victim_noise_anneal_frac')
        cond_config.pop('victim_noise_param')
        constructor_config.update(cond_config)
        return cls(**constructor_config, shaping_env=shaping_env)

    def get_value(self, frac_remaining):
        if self.shaping_env is None:
            raise ValueError("ConditionalAnnealer requires a RewardShapingVecWrapper to be set")

        current_data = self.shaping_env.get_log_buffer_data()
        current_metric_data = current_data[self.metric][-self.window_size:]
        current_wait = self.last_num_episodes - current_data['num_episodes']
        if current_wait < self.min_wait:
            return self.current_param_val

        avg_metric_val = float(sum(current_metric_data)) / self.window_size
        if self.operator(avg_metric_val, self.thresh):
            self.current_param_val *= self.decay_factor
            self.last_num_episodes = current_data['num_episodes']
        return self.current_param_val


DEFAULT_ANNEALERS = {
    # Schedule used in the multiagent competition paper for reward shaping.
    'default_reward': LinearAnnealer(1, 0, 0.5),
    # Default baselines.ppo2 learning rate
    'default_lr': ConstantAnnealer(3e-4),
}
