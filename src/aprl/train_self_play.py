import argparse
import datetime
import functools
import os

import gym
from stable_baselines import logger

from aprl.beta.ppo_self_play import PPOSelfPlay
from aprl.common.multi_monitor import MultiMonitor
from aprl.envs import make_dummy_vec_multi_env

ISO_TIMESTAMP = "%Y%m%d_%H%M%S"


def parse():
    parser = argparse.ArgumentParser(description='Train via self-play.')
    parser.add_argument('--vec-env', type=int, default=8,
                        help='number of environments to run in parallel.')
    parser.add_argument('--population-size', type=int, default=4,
                        help='number of agents to training via self-play')
    parser.add_argument('--network', type=str, default='mlp',
                        help='network architecture for policy')
    parser.add_argument('--total-timesteps', type=int, default=100000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output-dir', type=str, default='data')
    parser.add_argument('env', type=str, help='environment')
    parser.add_argument('exp_name', type=str, help='name of experiment')
    return parser.parse_args()


def main():
    args = parse()
    timestamp = datetime.datetime.now().strftime(ISO_TIMESTAMP)
    out_dir = '{}-{}'.format(timestamp, args.exp_name)
    out_path = os.path.join(args.output_dir, out_dir)
    os.makedirs(out_path)
    logger.configure(folder=os.path.join(out_path, 'baselines'))

    # Construct environments
    def make_env(i):
        env = gym.make(args.env)
        env = MultiMonitor(env, os.path.join(out_path, 'mon{:d}'.format(i)),
                           allow_early_resets=True)
        env.seed(args.seed + 10*i)
        return env
    env_fns = [functools.partial(make_env, i) for i in range(args.vec_env)]
    env = make_dummy_vec_multi_env(env_fns)

    # Perform self-play
    self_play = PPOSelfPlay(population_size=args.population_size,
                            training_type='best',
                            env=env,
                            network=args.network)
    self_play.learn(total_timesteps=args.total_timesteps)
    env.close()


if __name__ == '__main__':
    main()
