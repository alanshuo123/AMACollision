#!/bin/env python
'''
cnn model :mnih15, not mnih15_shared_weights
"vf_share_layers": False,
rewardFunction: advrs
除了使用 LSTM，您还可以使用 "preprocessor_pref": "deepmind" 来启用帧堆叠。
'''
import gym
import macad_gym  # noqa F401
import argparse
import os
from pprint import pprint

import cv2
import ray
import ray.tune as tune
from gym.spaces import Box, Discrete, Tuple, Dict
# from env_wrappers import wrap_deepmind
from models import register_mnih15_net, register_mnih15_shared_weights_net, register_carla_model, register_carla_imitation_model

from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy  # 0.8.5
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.preprocessors import Preprocessor
from ray.tune import register_env
import time
# import tensorflow as tf
# from tensorboardX import SummaryWriter

import datetime

start_time = datetime.datetime.now()
print("start-time", start_time)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--env",
    default="Town03_roundabout-v0",
    help="Name Gym env. Used only in debug mode. Default=PongNoFrameskip-v4")
parser.add_argument(
    "--disable-comet",
    action="store_true",
    help="Disables comet logging. Used for local smoke tests")
parser.add_argument(
    "--num-workers",
    default=1,  # 2 #fix
    type=int,
    help="Num workers (CPU cores) to use")
parser.add_argument(
    "--num-gpus", default=0, type=int, help="Number of gpus to use. Default=2")
parser.add_argument(
    "--sample-bs-per-worker",  # one iteration
    default=150,
    type=int,
    help="Number of samples in a batch per worker. Default=50")
parser.add_argument(
    "--train-bs",
    default=300,
    type=int,
    help="Train batch size. Use as per available GPU mem. Default=500")
parser.add_argument(
    "--envs-per-worker",
    default=1,
    type=int,
    help="Number of env instances per worker. Default=10")
parser.add_argument(
    "--notes",
    default=None,
    help="Custom experiment description to be added to comet logs")
parser.add_argument(
    "--model-arch",
    default="carla_imitation_model",
    help="Model architecture to use. Default=carla")
parser.add_argument(
    "--num-steps",
    default=1000000,
    type=int,
    help="Number of steps to train. Default=20M")
parser.add_argument(
    "--num-iters",
    default=1024,
    type=int,
    help="Number of training iterations. Default=20")
parser.add_argument(
    "--log-graph",
    action="store_true",
    help="Write TF graph on Tensorboard for debugging", default=True)
parser.add_argument(
    "--num-framestack",
    type=int,
    default=4,
    help="Number of obs frames to stack")
parser.add_argument(
    "--debug", action="store_true", help="Run in debug-friendly mode", default=False)
parser.add_argument(
    "--redis-address",
    default=None,
    help="Address of ray head node. Be sure to start ray with"
         "ray start --redis-address <...> --num-gpus<.> before running this script")
parser.add_argument(
    "--use-lstm", action="store_true", help="Append a LSTM cell to the model", default=True)

args = parser.parse_args()

model_name = "carla_imitation_model"
register_carla_imitation_model()
# Used only in debug mode
env_name = 'Town03_roundabout-v0'
env = gym.make(env_name)

env_actor_configs = env.configs
num_framestack = args.num_framestack


# env_config["env"]["render"] = False


def env_creator(env_config):
    import macad_gym
    env = gym.make("Town03_roundabout-v0")

    # Apply wrappers to: convert to Grayscale, resize to 84 x 84,
    # stack frames & some more op
    # env = wrap_deepmind(env, dim=84, num_framestack=num_framestack)
    return env


register_env(env_name, lambda config: env_creator(config))

config = {
    # Model and preprocessor options.
    "model": {
        "custom_model": model_name,
        "custom_options": {
            # Custom notes for the experiment
            "notes": {
                "args": vars(args)
            },
        },
        # NOTE:Wrappers are applied by RLlib if custom_preproc is NOT specified
        # "custom_preprocessor": "sq_im_84",
        "dim": 88,
        "free_log_std": False,  # if args.discrete_actions else True,
        "grayscale": True,
    },

    "env_config": env_actor_configs
}


def default_policy():
    env_actor_configs["env"]["render"] = True
    config = {
        # Model and preprocessor options.
        "model": {
            "custom_model": model_name,
            "custom_options": {
                # Custom notes for the experiment
                "notes": {
                    "args": vars(args)
                },
            },
            # NOTE:Wrappers are applied by RLlib if custom_preproc is NOT specified
            # "custom_preprocessor": "sq_im_84",
            "dim": 84,
            "free_log_std": False,  # if args.discrete_actions else True,
            "grayscale": True,
            # conv_filters to be used with the custom CNN model.
            # "conv_filters": [[16, [4, 4], 2], [32, [3, 3], 2], [16, [3, 3], 2]]
            #use LSTM
            "use_lstm": False,
            # "lstm_cell_size": 64,
            # "lstm_use_prev_action_reward": True,
        },
        # Should use a critic as a baseline (otherwise don't use value baseline;
        # required for using GAE).
        "use_critic": True,
        # If true, use the Generalized Advantage Estimator (GAE)
        # with a value function, see https://arxiv.org/pdf/1506.02438.pdf.
        "use_gae": True,
        # The GAE(lambda) parameter.
        "lambda": 1.0,
        # Initial coefficient for KL divergence.
        "kl_coeff": 0.3,

        "rollout_fragment_length": args.sample_bs_per_worker,
        "sgd_minibatch_size": 18,
        # Whether to shuffle sequences in the batch when training (recommended).
        "shuffle_sequences": True,
        # Number of SGD iterations in each outer loop (i.e., number of epochs to
        # execute per train batch).
        "num_sgd_iter": 4,
        # Stepsize of SGD.
        # tune.grid_search([0.01, 0.001, 0.0001])
        "lr": 0.0001,
        # Learning rate schedule.
        # "lr_schedule": None,
        # Share layers for value function. If you set this to True, it's important
        # to tune vf_loss_coeff.
        "vf_share_layers": True,
        # Coefficient of the value function loss. IMPORTANT: you must tune this if
        # you set vf_share_layers: True.
        "vf_loss_coeff": 1.0,
        # Coefficient of the entropy regularizer.
        "entropy_coeff": 0.1,
        # Decay schedule for the entropy regularizer.
        "entropy_coeff_schedule": None,
        # PPO clip parameter.
        "clip_param": 0.3,
        # Clip param for the value function. Note that this is sensitive to the
        # scale of the rewards. If your expected V is large, increase this.
        "vf_clip_param": 10.0,
        # If specified, clip the global norm of use_lstmgradients by this amount.
        "grad_clip": None,
        # Target value for KL divergence.
        "kl_target": 0.03,
        # Whether to rollout "complete_episodes" or "truncate_episodes".
        "batch_mode": "complete_episodes",
        # Which observation filter to apply to the observation.
        "observation_filter": "NoFilter",
        # Uses the sync samples optimizer instead of the multi-gpu one. This is
        # usually slower, but you might want to try it if you run into issues with
        # the default optimizer.
        "simple_optimizer": False,
        # Use PyTorch as framework?
        "use_pytorch": False,

        # Discount factor of the MDP.
        "gamma": 0.99,
        # Number of steps after which the episode is forced to terminate. Defaults
        # to `env.spec.max_episode_steps` (if present) for Gym envs.
        "horizon": 512,
        "sample_batch_size":
            args.sample_bs_per_worker,
        "train_batch_size":
            args.train_bs,
        # "rollout_fragment_length": 128,
        "num_workers":
            args.num_workers,
        # Number of environments to evaluate vectorwise per worker.
        "num_envs_per_worker":
            args.envs_per_worker,
        # "num_cpus_per_worker":
        #     1,
        "num_gpus_per_worker":
            0,
        "env_config": env_actor_configs
    }

    # pprint (config)
    return (PPOTFPolicy,
            Tuple(
                [
                    Box(0.0,1.0,shape=(512,)),  # image feature
                    Box(-20.0, 20.0, shape=(11,)),  # forward_speed, dist to ego, direction to ego(x/y axis),
                ]
            ),
            Discrete(9),
            config)


# pprint (args.checkpoint_path)
# pprint(os.path.isfile(args.checkpoint_path))

checkpoint_path="/home/tianshuo/ray_results/exp2.5_imitation_run/PPO_Town03_roundabout-v0_0_2023-12-27_00-32-48a7p2zuaq/checkpoint_2550/checkpoint-2550"
# ray.init(num_cpus=30,memory=60*1024*1024*1024)

analysis = tune.run(
    "PPO",
    name="round_road",
    stop={
        "training_iteration": 20000,
        "timesteps_total": 8000000,
        "episodes_total": 100000,
    },
    config={
        "env":env_name,
        "log_level": "DEBUG",
        # "num_sgd_iter": 10,  # Enables Experience Replay
        "multiagent": {
            "policies": {
                id: default_policy()
                for id in env_actor_configs["actors"].keys() #if id != "ego"
            },
            "policy_mapping_fn":
                tune.function(lambda agent_id: agent_id ),
            "policies_to_train": ["car1", "car2"],
        },
        "env_config": env_actor_configs,
        "num_workers": 1,
        "num_gpus": 0,#args.num_gpus,
        "num_envs_per_worker": args.envs_per_worker,
        "sample_batch_size": args.sample_bs_per_worker,
        "train_batch_size": args.train_bs,
        # "horizon": 512,
        "lr": 0.0001
        # "lr": tune.grid_search([0.01, 0.001, 0.0001]),
        # "horizon": 512, #yet to be fixed
        # "framework": "tf"
    },
    checkpoint_freq=50,
    checkpoint_at_end=True,
    restore=checkpoint_path,
)

end_time = datetime.datetime.now()
print("endtime:", end_time)

time_delta = end_time - start_time
hours = time_delta.total_seconds() / 3600
print("training time:", hours)

ray.shutdown()