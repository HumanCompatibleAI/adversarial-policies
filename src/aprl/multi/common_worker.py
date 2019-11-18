import collections

import sacred


def flatten_config(config):
    """Take dict with ':'-separated keys and values or tuples of values,
       flattening to single key-value pairs.

       Example: _flatten_config({'a:b': (1, 2), 'c': 3}) -> {'a: 1, 'b': 2, 'c': 3}."""
    new_config = {}
    for ks, vs in config.items():
        ks = ks.split(":")
        if len(ks) == 1:
            vs = (vs,)

        for k, v in zip(ks, vs):
            assert k not in new_config, f"duplicate key '{k}'"
            new_config[k] = v

    return new_config


def update(d, u):
    """Recursive dictionary update."""
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def fix_sacred_capture():
    """Workaround for Sacred stdout capture issue #195 and Ray issue #5718."""
    # TODO(adam): remove once Sacred issue #195 is closed
    sacred.SETTINGS.CAPTURE_MODE = "sys"
