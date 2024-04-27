'''
cnn model :mnih15, not mnih15_shared_weights
"vf_share_layers": False,
rewardFunction: advrs
'''
import math

import gym
import numpy as np

import macad_gym  # noqa F401
import argparse
import os
from pprint import pprint
from collections import defaultdict
import cv2
import ray
import ray.tune as tune
from gym.spaces import Box, Discrete, Tuple, Dict
from env_wrappers import wrap_deepmind
from models import register_mnih15_net, register_mnih15_shared_weights_net, register_carla_model, register_carla_imitation_model

from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy  # 0.8.5
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.preprocessors import Preprocessor
from ray.tune import register_env
import time
import tensorflow as tf
from tensorboardX import SummaryWriter
import ray.rllib.agents.ppo as ppo
import datetime

start_time = datetime.datetime.now()
print("start-time", start_time)
parser = argparse.ArgumentParser()
parser.add_argument(
    "--env",
    default="Town03I3C2_measure_continuous-v0",
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
env_name = 'Town03I3C2_measure_continuous-v0'
env = gym.make(env_name)

env_actor_configs = env.configs
num_framestack = args.num_framestack


# env_config["env"]["render"] = False


def env_creator(env_config):
    import macad_gym
    env = gym.make("Town03I3C2_measure_continuous-v0")

    # Apply wrappers to: convert to Grayscale, resize to 84 x 84,
    # stack frames & some more op
    # env = wrap_deepmind(env, dim=84, num_framestack=num_framestack)
    return env


register_env(env_name, lambda config: env_creator(config))


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



ray.init()

config = {
    "log_level": "DEBUG",
    # "num_sgd_iter": 10,  # Enables Experience Replay
    "multiagent": {
        "policies": {
            id: default_policy()
            for id in env_actor_configs["actors"].keys()
        },
        "policy_mapping_fn": tune.function(lambda agent_id: agent_id),
        # "policies_to_train": ["car1","car2", "car3"],
    },
    "env_config": env_actor_configs,
    "num_workers": args.num_workers,
    "num_gpus": args.num_gpus,
}

# ray.init(num_cpus=30,memory=60*1024*1024*1024)
checkpoint_path="/home/tianshuo/ray_results/exp2.5_imitation_run/PPO_Town03I3C2_measure_continuous-v0_0_2023-12-24_01-46-28gwfqjps7/checkpoint_20000/checkpoint-20000"

print("***Training completed. Restoring new Trainer for action inference.***")


trainer = ppo.PPOTrainer(
    env=env_name,
    config=config)
trainer.restore(checkpoint_path=checkpoint_path)
configs = env.configs


velocity = np.arange(1, 11)  #速度限制数据

collision_rate_speed = defaultdict(list)
goal_rate_speed = defaultdict(list)
avg_ttfc_speed = defaultdict(list)
avg_ttg_speed = defaultdict(list)
avg_offlane_speed =defaultdict(list)
avg_collision_to_ego_speed =defaultdict(list)
avg_collision_to_vehicles_speed =defaultdict(list)
# avg_collision_to_egos_speed = defaultdict(list)
avg_collision_to_others_speed = defaultdict(list)
# egocollision的总次数占 全部collision次数的比例
ego_collision_rate_speed = defaultdict(list)
# 发生碰撞的episode中，ego碰撞占总碰撞的比例
avg_ego_collision_rate_speed = defaultdict(list)
avg_reward_speed = defaultdict(list)
lateral_distance_speed = defaultdict(list)
actor_ids = ["ego", "car1", "car2","car3"]
for v in velocity:
    print("velocity: "+str(v)+" m/s")
    time_to_first_collision = defaultdict(list)
    time_to_goal = defaultdict(list)
    final_reward = defaultdict(list)
    collision_times = defaultdict(list)
    offlane_times = defaultdict(list)
    collision_to_ego_times = defaultdict(list)
    collision_to_vehicles = defaultdict(list)
    collision_to_others = defaultdict(list)
    lateral_distance_rmse = defaultdict(list)

    for ep in range(200):
        obs = env.reset()

        total_reward_dict = {}
        action_dict = {}

        ep_reward = defaultdict(list)

        env_config = configs["env"]
        actor_configs = configs["actors"]
        for actor_id in actor_configs.keys():
            env._behavior_agents[actor_id]._behavior.max_speed = v*3.6
            total_reward_dict[actor_id] = 0
            if env._discrete_actions:
                action_dict[actor_id] = 3  # Forward
            else:
                action_dict[actor_id] = [1, 0]  # test values

        start = time.time()
        i = 0
        done = {"__all__": False}

        ego_flag=False
        while not done["__all__"]:
            # while i < 20:  # TEST
            i += 1
            action_dict = {}
            for actor_id in actor_configs.keys():
                # if actor_id == "ego":
                #     action_dict["ego"] = 0
                # else :
                action_dict[actor_id] = trainer.compute_action(policy_id=actor_id, observation=obs[actor_id])

            obs, reward, done, info = env.step(action_dict)

            for actor_id in total_reward_dict.keys():
                total_reward_dict[actor_id] += reward[actor_id]
                ep_reward[actor_id].append(total_reward_dict[actor_id])

            # print("ego velocity:",env._actors["ego"].get_velocity().x,env._actors["ego"].get_velocity().y,env._actors["ego"].get_velocity().z)
            # print("ego location:", env._actors["ego"].get_location().x, env._actors["ego"].get_location().y,
            #       env._actors["ego"].get_location().z)
            # print(":{}\n\t".join(["Step#", "rew", "ep_rew",
            #                       "done{}"]).format(i, reward,
            #                                         total_reward_dict, done))
        print("episodes_",ep)
        print("_lateral_list:", env._lateral_distance)
        if (env._collisions["ego"].collision_vehicles > 0):
            print(env._collisions["car1"].collision_to_ego, env._collisions["car2"].collision_to_ego, env._collisions["car3"].collision_to_ego)
            print(ep,"_reward: ",ep_reward)
        for actor_id in total_reward_dict.keys():
            if env._time_to_firstcollision[actor_id] is None:
                time_to_first_collision[actor_id].append(0)
            else:
                time_to_first_collision[actor_id].append(env._time_to_firstcollision[actor_id])
            if env._time_to_goal[actor_id] is None:
                time_to_goal[actor_id].append(0)
            else:
                time_to_goal[actor_id].append(env._time_to_goal[actor_id])
            final_reward[actor_id].append(total_reward_dict[actor_id])
            collision_times[actor_id].append(
                env._collisions[actor_id].collision_vehicles + env._collisions[actor_id].collision_other +
                env._collisions[actor_id].collision_to_ego)
            collision_to_ego_times[actor_id].append(env._collisions[actor_id].collision_to_ego)
            collision_to_vehicles[actor_id].append(env._collisions[actor_id].collision_vehicles)
            collision_to_others[actor_id].append(env._collisions[actor_id].collision_other)
            offlane_times[actor_id].append(env._lane_invasions[actor_id].offlane)
            lateral = sum([num ** 2 for num in env._lateral_distance[actor_id]]) / len(env._lateral_distance[actor_id])
            lateral_distance_rmse[actor_id].append(math.sqrt(lateral))
        # for actor_id in actor_configs.keys():
        #     print(actor_id,":",env._collisions[actor_id].collision_vehicles)
        print(ep,"{} fps".format(i / (time.time() - start)))
    # print("time_to_goal",time_to_goal)
    # print("time_to_first_collision",time_to_first_collision)
    # print("final_reward",final_reward)
    # print("collision_to_ego_times",collision_to_ego_times)
    # print("collision_to_others",collision_to_others)
    # print("collision_to_vehicles",collision_to_vehicles)
    # print("offlane_time",offlane_times)


    collision_rate = {}
    goal_rate = {}
    avg_ttfc = {}
    avg_ttg = {}
    avg_offlane = {}
    avg_collision_to_vehicles = {}
    avg_collision_to_egos = {}
    avg_collision_to_ego = {}
    avg_collision_to_others = {}
    # egocollision的总次数占 全部collision次数的比例
    ego_collision_rate = {}
    # 发生碰撞的episode中，ego碰撞占总碰撞的比例
    avg_ego_collision_rate = {}
    avg_reward = {}
    
    for actor_id in actor_ids:
        print(actor_id+":")
        LDE=sum(lateral_distance_rmse[actor_id])/len(lateral_distance_rmse[actor_id])
        print("lateral_distance: ", LDE, end=", ")
        collision_rate[actor_id] = np.count_nonzero(time_to_first_collision[actor_id]) / len(
            time_to_first_collision[actor_id])
        print("collision_rate:", collision_rate[actor_id], end=", ")
        goal_rate[actor_id] = np.count_nonzero(time_to_goal[actor_id]) / len(time_to_goal[actor_id])
        print("goal_rate:", goal_rate[actor_id], end=", ")
        avg_ttfc[actor_id] = (np.sum(time_to_first_collision[actor_id]) + (
                len(time_to_first_collision[actor_id]) - np.count_nonzero(
            time_to_first_collision[actor_id])) * 200) / len(time_to_first_collision[actor_id])
        print("avg_ttfc:", avg_ttfc[actor_id], end=", ")
        avg_ttg[actor_id] = (np.sum(time_to_goal[actor_id]) + (
                len(time_to_goal[actor_id]) - np.count_nonzero(time_to_goal[actor_id])) * 200) / len(
            time_to_goal[actor_id])
        print("avg_ttg:", avg_ttg[actor_id], end=", ")
        avg_offlane[actor_id] = np.sum(offlane_times[actor_id]) / len(offlane_times[actor_id])
        print("avg_offlane:", avg_offlane[actor_id], end=", ")
        avg_collision_to_ego[actor_id] = np.sum(collision_to_ego_times[actor_id]) / len(
            collision_to_ego_times[actor_id])
        print("avg_collision_to_ego:", avg_collision_to_ego[actor_id], end=", ")
        avg_collision_to_vehicles[actor_id] = np.sum(collision_to_vehicles[actor_id]) / len(
            collision_to_vehicles[actor_id])
        print("avg_collision_to_vehicles:", avg_collision_to_vehicles[actor_id], end=", ")
        avg_collision_to_others[actor_id] = np.sum(collision_to_others[actor_id]) / len(collision_to_others[actor_id])
        print("avg_collision_to_others:", avg_collision_to_others[actor_id], end=", ")
        avg_reward[actor_id] = np.sum(final_reward[actor_id]) / len(final_reward[actor_id])
        print("avg_reward:", avg_reward[actor_id], end=", ")

        total_collision = np.add(collision_to_vehicles[actor_id],
                                np.add(collision_to_others[actor_id], collision_to_ego_times[actor_id]))
        collision_times =  np.sum(total_collision)  if np.sum(total_collision) > 0 else 0.1
        ego_collision_rate[actor_id] = np.sum(collision_to_ego_times[actor_id]) / collision_times
        print("ego_collision_rate:", ego_collision_rate[actor_id], end=", ") #与ego车碰撞的次数，占所有总碰撞次数的比例

        count = 0
        avg_ego_collision_rate[actor_id] = 0
        for i, value in enumerate(total_collision):
            if value > 0:
                count += 1
                avg_ego_collision_rate[actor_id] += (collision_to_ego_times[actor_id][i] / total_collision[i])
        if count > 0:
            avg_ego_collision_rate[actor_id] /= count
        print("avg_ego_collision_rate:", avg_ego_collision_rate[actor_id])

        collision_rate_speed[actor_id].append(collision_rate[actor_id])
        goal_rate_speed[actor_id].append(goal_rate[actor_id])
        avg_ttfc_speed[actor_id].append(avg_ttfc[actor_id])
        avg_ttg_speed[actor_id].append(avg_ttg[actor_id])
        avg_offlane_speed[actor_id].append(avg_offlane[actor_id])
        avg_collision_to_ego_speed[actor_id].append(avg_collision_to_ego[actor_id])
        avg_collision_to_vehicles_speed[actor_id].append(avg_collision_to_vehicles[actor_id])
        avg_collision_to_others_speed[actor_id].append(avg_collision_to_others[actor_id])
        # egocollision的总次数占 全部collision次数的比例
        ego_collision_rate_speed[actor_id].append(ego_collision_rate[actor_id])
        # 发生碰撞的episode中，ego碰撞占总碰撞的比例
        avg_ego_collision_rate_speed[actor_id].append(avg_ego_collision_rate[actor_id])
        avg_reward_speed[actor_id].append(avg_reward[actor_id])
        lateral_distance_speed[actor_id].append(LDE)

for actor_id in actor_ids:
    print(actor_id+":")
    print("lateral_distance_speed: ", lateral_distance_speed[actor_id])
    print("collision_rate_speed: ", collision_rate_speed[actor_id])
    print("avg_ttfc_speed: ", avg_ttfc_speed[actor_id])
    print("avg_ttg_speed: ", avg_ttg_speed[actor_id])
    print("avg_offlane_speed: ", avg_offlane_speed[actor_id])
    print("avg_collision_to_ego_speed: ", avg_collision_to_ego_speed[actor_id])
    print("avg_collision_to_vehicles_speed: ", avg_collision_to_vehicles_speed[actor_id])
    print("avg_collision_to_others_speed: ", avg_collision_to_others_speed[actor_id])
    print("ego_collision_rate_speed: ", ego_collision_rate_speed[actor_id])
    print("avg_ego_collision_rate_speed: ", avg_ego_collision_rate_speed[actor_id])
    print("avg_reward_speed: ", avg_reward_speed[actor_id])
ray.shutdown()