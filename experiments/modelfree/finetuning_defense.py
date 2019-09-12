import os.path as osp

from sacred.observers import FileStorageObserver

from modelfree.score_agent import score_ex
from modelfree.train import train_ex


@train_ex.named_config
def finetuning_defense():
    # After 10e6, win rate of adversary down to 7%
    # adversary is index 0, training agent is index 1. Results at sacred/train/13, and also
    # at 20190911-124616-default
    rew_shape = True
    rew_shape_params = {'anneal_frac': 0}
    load_policy = {
        'path': '1',
        'type': 'zoo',
    }
    normalize = False
    total_timesteps = int(10e6)
    batch_size = 32768
    learning_rate = 1e-4
    rl_args = {
        'ent_coef': 0.0,
        'nminibatches': 4,
        'noptepochs': 4,
    }
    env_name = 'multicomp/YouShallNotPassHumans-v0'
    victim_type = "ppo2"
    victim_path = "data/aws-public/multi_train/paper/20190429_011349/" \
                  "train_rl-7086bd7945d8a380b53e797f3932c739_10_env_name:" \
                  "victim_path=['multicomp_YouShallNotPassHumans-v0', 1],seed=0," \
                  "victim_index=1_2019-04-29_01-13-49dzng78qx/data/baselines" \
                  "/20190429_011353-default-env_name=multicomp" \
                  "_YouShallNotPassHumans-v0-victim_path=1-seed=0-victim_index=1/final_model"
    victim_index = 0

    _ = locals()
    del _


@train_ex.named_config
def test_retraining_adversary():
    # hyperparams stolen from best_guess_train
    rew_shape = True
    rew_shape_params = {'anneal_frac': 0}
    load_policy = {
        'path': None,
        'type': 'ppo2',
    }
    normalize = True
    total_timesteps = int(10e6)
    batch_size = 16384
    learning_rate = 3e-4
    rl_args = {
        'ent_coef': 0.0,
        'nminibatches': 4,
        'noptepochs': 4,
    }
    env_name = 'multicomp/YouShallNotPassHumans-v0'
    victim_type = "ppo2"
    victim_path = "data/baselines/20190911_124616-default/final_model"
    victim_index = 1

    _ = locals()
    del _


@score_ex.named_config
def test_adversary_trained_model():
    # Other agent wins 70% of the time relative to 30% for fine-tuned agent
    env_name = 'multicomp/YouShallNotPassHumans-v0'
    agent_a_type = "zoo"
    agent_a_path = "1"
    agent_b_type = "ppo2"
    agent_b_path = "data/baselines/20190911_124616-default/final_model"
    episodes = 100
    _ = locals()
    del _


if __name__ == "__main__":
    # observer = FileStorageObserver.create(osp.join('data', 'sacred', 'score_agent'))
    # score_ex.observers.append(observer)
    # score_ex.run(named_configs=["test_adversary_trained_model"])
    observer = FileStorageObserver.create(osp.join('data', 'sacred', 'train'))
    train_ex.observers.append(observer)
    train_ex.run(named_configs=["test_retraining_adversary"])
    # train_ex.run(named_configs=["finetuning_defense"])
