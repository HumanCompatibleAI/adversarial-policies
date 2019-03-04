from abc import ABC, abstractmethod
import collections
import functools
import operator
from stable_baselines import logger


class Scheduler(object):
    """Keep track of frac_remaining and return time-dependent values"""
    def __init__(self, func_dict=None):
        if func_dict is None:
            func_dict = {}
        assert isinstance(func_dict, dict)
        self.func_dict = func_dict
        self.allowed_func_types = ['lr', 'rew_shape', 'noise']
        self.frac_remaining = 1  # frac_remaining goes from 1 to 0

    def _update_frac_remaining(self, frac_remaining=None):
        if frac_remaining is not None:
            self.frac_remaining = frac_remaining

    def set_func(self, func_type, func):
        """
        Common interface for attaching a function to this Scheduler.
        :param func_type: str, must be in self.allowed_func_types
        :param func: callable which takes one optional argument - frac_remaining
        :return: None
        """
        assert func_type in self.allowed_func_types
        assert callable(func) or func is None
        self.func_dict[func_type] = func

    def get_val(self, val_type, frac_remaining=None):
        """
        Common interface for getting values from functions attached to this Scheduler.
        :param val_type: str, must be in self.allowed_func_types
        :param frac_remaining: optional arg, if given will update the state of Scheduler
        :return: scalar float
        """
        assert val_type in self.allowed_func_types
        self._update_frac_remaining(frac_remaining)
        return self.func_dict[val_type](self.frac_remaining)

    def get_func(self, func_type):
        """
        Interface for accessing the functions in self.func_dict
        :param func_type: str, must be in self.allowed_func_types
        :return: callable which takes one optional argument - frac_remaining
        """
        assert func_type in self.allowed_func_types
        if self.func_dict.get(func_type) is None:
            return None
        else:
            return functools.partial(self.get_val, func_type)


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
    def __init__(self, const_val, logger=None):
        self.const_val = const_val
        self.logger = logger
        super().__init__(const_val, const_val)

    def get_value(self, frac_remaining):
        return self.const_val


class LinearAnnealer(Annealer):
    """Linearly anneals from start_val to end_val over end_frac fraction of training"""
    def __init__(self, start_val, end_val, end_frac, logger=None):
        super().__init__(start_val, end_val)
        assert 0 <= end_frac <= 1, "Invalid end_frac for LinearAnnealer"
        self.end_frac = end_frac
        self.logger = logger

    def get_value(self, frac_remaining):
        anneal_progress = min(1.0, (1 - frac_remaining) / self.end_frac)
        return (1 - anneal_progress) * self.start_val + anneal_progress * self.end_val


class ConditionalAnnealer(Annealer):
    """Anneal the value depending on some condition."""
    def __init__(self, start_val, end_val, decay_factor, thresh, window_size,
                 min_wait, operator, metric, logger):
        super().__init__(start_val, end_val)
        self.decay_factor = decay_factor
        self.thresh = thresh
        self.window_size = window_size
        self.min_wait = min_wait
        self.operator = operator
        self.metric = metric
        self.logger = logger

        self.current_param_val = start_val
        self.current_wait = 0
        self.window_vals = collections.deque([0], maxlen=self.window_size)

    @classmethod
    def from_dict(cls, cond_config, logger):
        constructor_config = {
            'decay_factor': 0.98,
            'thresh': 0,
            'start_val': 2,
            'end_val': 0,
            'window_size': 1,
            'min_wait': 1,
            'operator': operator.lt,
            'metric': 'z-epsparsemean',
        }
        if 'operator' in cond_config:
            cond_config['operator'] = getattr(operator, cond_config['operator'])
        cond_config.pop('victim_noise_anneal_frac')
        cond_config.pop('victim_noise_param')
        constructor_config.update(cond_config)
        return cls(**constructor_config, logger=logger)

    def get_value(self, frac_remaining):
        current_metric_val = self.logger.getkvs()[self.metric]
        if current_metric_val != self.window_vals[-1]:  # only update if new episode mean
            self.window_vals.append(current_metric_val)  # deque pops other side automatically
            self.current_wait += 1

        avg_metric_val = float(sum(self.window_vals)) / self.window_size
        if self.current_wait >= self.min_wait and self.operator(avg_metric_val, self.thresh):
            self.current_param_val *= self.decay_factor
            self.current_wait = 0
        return self.current_param_val


DEFAULT_ANNEALERS = {
    # Schedule used in the multiagent competition paper for reward shaping.
    'default_reward': LinearAnnealer(1, 0, 0.5),
    # Default baselines.ppo2 learning rate
    'default_lr': ConstantAnnealer(3e-4),
}
