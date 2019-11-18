from abc import ABC, abstractmethod
import collections
import functools
import itertools
import operator


def _validate_func_type(func_type):
    allowed_func_types = frozenset(["lr", "rew_shape", "noise"])
    if func_type not in allowed_func_types:
        raise KeyError("func_type not in allowed_func_types")


class Scheduler(object):
    """Keep track of frac_remaining and return time-dependent values"""

    def __init__(self, annealer_dict=None):
        if annealer_dict is None:
            annealer_dict = {}
        self.annealer_dict = annealer_dict
        self.conditionals = collections.defaultdict(lambda: False)
        self.frac_remaining = 1  # frac_remaining goes from 1 to 0

    def _update_frac_remaining(self, frac_remaining=None):
        if frac_remaining is not None:
            self.frac_remaining = frac_remaining

    def set_conditional(self, func_type):
        """Registers that the annealer for func_type is a ConditionalAnnealer"""
        _validate_func_type(func_type)
        self.conditionals[func_type] = True

    def is_conditional(self, func_type):
        """Interface to see whether the annealer for func_type is a ConditionalAnnealer"""
        _validate_func_type(func_type)
        return self.conditionals[func_type]

    def set_annealer(self, func_type, annealer):
        """
        Common interface for attaching a function to this Scheduler.
        :param func_type: str, must be in self.allowed_func_types
        :param annealer: Annealer, must have a get_value method
        :return: None
        """
        _validate_func_type(func_type)
        if not isinstance(annealer, Annealer):
            raise TypeError("set_annealer_and_func requires an Annealer as input")
        self.annealer_dict[func_type] = annealer

    def get_val(self, val_type, frac_remaining=None):
        """
        Common interface for getting values from functions attached to this Scheduler.
        :param val_type: str, must be in self.allowed_func_types
        :param frac_remaining: optional arg, if given will update the state of Scheduler
        :return: scalar float
        """
        _validate_func_type(val_type)
        self._update_frac_remaining(frac_remaining)
        return self.annealer_dict[val_type].get_value(self.frac_remaining)

    def get_annealer(self, func_type):
        """
        Interface for accessing the functions in self.func_dict
        :param func_type: str, must be in self.allowed_func_types
        :return: callable which takes one optional argument - frac_remaining
        """
        _validate_func_type(func_type)
        if func_type not in self.annealer_dict:
            return None
        else:
            return functools.partial(self.get_val, func_type)

    def set_annealer_get_logs(self, func_type, get_logs):
        """Interface for setting the get_logs function for an Annealer"""
        _validate_func_type(func_type)
        self.annealer_dict[func_type].set_get_logs(get_logs)


class Annealer(ABC):
    """Abstract class for implementing Annealers."""

    def __init__(self, start_val, end_val, get_logs=None):
        """
        :param start_val: (int or float) starting value
        :param end_val: (int or float) ending value
        :param get_logs: (RewardShapingVecWrapper.get_logs) function which returns
        a dict containing data on rewards and lengths of episodes.
        """
        self.start_val = start_val
        self.end_val = end_val
        self.get_logs = get_logs

    def __getstate__(self):
        """Custom pickler.
        Omits self.get_logs which may involve non-picklable objects such as tf.Session."""
        state = self.__dict__.copy()
        state["get_logs"] = None
        return state

    def set_get_logs(self, get_logs):
        """Set the get_logs attribute since passing into constructor may not be possible."""
        self.get_logs = get_logs

    @abstractmethod
    def get_value(self, step):
        raise NotImplementedError()


class ConstantAnnealer(Annealer):
    """Returns a constant value"""

    def __init__(self, const_val):
        self.const_val = const_val
        super().__init__(const_val, const_val)

    def get_value(self, frac_remaining):
        return self.const_val


class LinearAnnealer(Annealer):
    """Linearly anneals from start_val to end_val over end_frac fraction of training"""

    def __init__(self, start_val, end_val, end_frac):
        super().__init__(start_val, end_val)
        if not 0 <= end_frac <= 1:
            raise ValueError(f"Invalid end_frac of {end_frac} for LinearAnnealer")
        self.end_frac = end_frac

    def get_value(self, frac_remaining):
        if self.end_frac == 0:
            anneal_progress = 1.0
        else:
            anneal_progress = min(1.0, (1 - frac_remaining) / self.end_frac)
        return (1 - anneal_progress) * self.start_val + anneal_progress * self.end_val


class ConditionalAnnealer(Annealer):
    """Anneal the value depending on some condition."""

    def __init__(
        self,
        start_val,
        end_val,
        decay_factor,
        thresh,
        window_size,
        min_wait,
        max_wait,
        operator,
        metric,
        get_logs,
    ):
        super().__init__(start_val, end_val, get_logs)
        self.decay_factor = decay_factor
        self.thresh = thresh
        self.window_size = window_size
        self.min_wait = min_wait
        self.max_wait = max_wait
        self.operator = operator
        self.metric = metric

        self.current_param_val = start_val
        self.last_total_episodes = 0

    @classmethod
    def from_dict(cls, cond_config, get_logs=None):
        # provided to help keep track of arguments
        constructor_config = {
            "decay_factor": 0.98,
            "thresh": 0,
            "start_val": 1,
            "end_val": 0,
            "window_size": 1,
            "min_wait": 1,
            "max_wait": 10000,
            "operator": operator.gt,
            "metric": "sparse",  # dense, length
        }
        if "operator" in cond_config:
            cond_config["operator"] = getattr(operator, cond_config["operator"])

        trimmed_config = {k: v for k, v in cond_config.items() if k in constructor_config}
        constructor_config.update(trimmed_config)
        return cls(**constructor_config, get_logs=get_logs)

    def get_value(self, frac_remaining):
        if self.get_logs is None:
            raise ValueError("ConditionalAnnealer requires a get_logs function to be set")

        current_data = self.get_logs()
        if current_data is None:  # if we have zero episodes thus far
            return self.current_param_val
        current_wait = current_data["total_episodes"] - self.last_total_episodes
        if current_wait < self.min_wait:
            return self.current_param_val

        metric_data = current_data[self.metric]
        current_metric_data = list(itertools.islice(metric_data, self.window_size))
        avg_metric_val = float(sum(current_metric_data)) / len(current_metric_data)
        if self.operator(avg_metric_val, self.thresh) or current_wait > self.max_wait:
            self.current_param_val *= self.decay_factor
            self.last_total_episodes = current_data["total_episodes"]
        return self.current_param_val
