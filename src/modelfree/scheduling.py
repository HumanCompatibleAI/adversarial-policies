from abc import ABC, abstractmethod
import functools


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
        assert callable(func)
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
    def __init__(self, const_val):
        self.const_val = const_val
        super().__init__(const_val, const_val)

    def get_value(self, frac_remaining):
        return self.const_val


class LinearAnnealer(Annealer):
    """Linearly anneals from start_val to end_val over end_frac fraction of training"""
    def __init__(self, start_val, end_val, end_frac):
        super().__init__(start_val, end_val)
        assert 0 <= end_frac <= 1, "Invalid end_frac for LinearAnnealer"
        self.end_frac = end_frac

    def get_value(self, frac_remaining):
        if self.end_frac == 0:
            anneal_progress = 1.0
        else:
            anneal_progress = min(1.0, (1 - frac_remaining) / self.end_frac)
        return (1 - anneal_progress) * self.start_val + anneal_progress * self.end_val
