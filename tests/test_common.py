import os
import tempfile

import gym

from aprl.common.multi_monitor import MultiMonitor
import aprl.envs  # noqa: F401


def test_multi_monitor():
    """Smoke test for MultiMonitor."""
    env = gym.make("aprl/IteratedMatchingPennies-v0")
    env.seed(42)
    with tempfile.TemporaryDirectory(prefix="test_multi_mon") as d:
        env = MultiMonitor(env, filename=os.path.join(d, "test"))
        for eps in range(5):
            env.reset()
            done = False
            while not done:
                a = env.action_space.sample()
                _, _, done, info = env.step(a)
            epinfo = info["episode"]
            assert set(epinfo.keys()) == {"r", "r0", "r1", "l", "t"}
