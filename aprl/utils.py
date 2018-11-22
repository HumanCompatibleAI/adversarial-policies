def getattr_unwrapped(env, attr):
    """Get attribute attr from env, or one of the nested environments.
    Args:
        - env(gym.Wrapper or gym.Env): a (possibly wrapped) environment.
        - attr: name of the attribute
    Returns:
        env.attr, if present, otherwise env.unwrapped.attr and so on recursively.
    """
    try:
        return getattr(env, attr)
    except AttributeError:
        if env.env == env:
            raise
        else:
            return getattr_unwrapped(env.env, attr)