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
import ray
import ray.tune as tune
from gym.spaces import Box, Discrete, Tuple, Dict
from models import register_mnih15_net, register_mnih15_shared_weights_net, register_carla_model, register_carla_imitation_model

from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy  # 0.8.5
from ray.tune import register_env
import time
import ray.rllib.agents.ppo as ppo
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

# checkpoint_path="/home/tianshuo/ray_results/exp2.5_imitation_run/PPO_Town03_roundabout-v0_0_2023-11-30_17-23-42lx402gw9/checkpoint_7000/checkpoint-7000"
# ray.init(num_cpus=30,memory=60*1024*1024*1024)
checkpoint_path="/home/tianshuo/ray_results/round_road/PPO_Town03_roundabout-v0_0_2023-12-27_23-02-58gk6o9tah/checkpoint_6950/checkpoint-6950"

print("***Training completed. Restoring new Trainer for action inference.***")


trainer = ppo.PPOTrainer(
    env=env_name,
    config=config)
trainer.restore(checkpoint_path=checkpoint_path)
configs = env.configs

time_to_first_collision = defaultdict(list)
time_to_goal = defaultdict(list)
final_reward = defaultdict(list)
collision_times = defaultdict(list)
offlane_times = defaultdict(list)
collision_to_ego_times = defaultdict(list)
collision_to_vehicles = defaultdict(list)
collision_to_others = defaultdict(list)
lateral_distance_rmse = defaultdict(list)

for ep in range(100):
    obs = env.reset()

    total_reward_dict = {}
    action_dict = {}

    ep_reward = defaultdict(list)

    env_config = configs["env"]
    actor_configs = configs["actors"]
    for actor_id in actor_configs.keys():
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
        print(env._collisions["car1"].collision_to_ego, env._collisions["car2"].collision_to_ego)
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
print("time_to_goal",time_to_goal)
print("time_to_first_collision",time_to_first_collision)
print("final_reward",final_reward)
print("collision_to_ego_times",collision_to_ego_times)
print("collision_to_others",collision_to_others)
print("collision_to_vehicles",collision_to_vehicles)
print("offlane_time",offlane_times)
print("lateral_distance_rmse",lateral_distance_rmse)

collision_rate = {}
goal_rate = {}
avg_ttfc = {}
avg_ttg = {}
avg_offlane = {}
avg_collision_to_ego = {}
avg_collision_to_vehicles = {}
avg_collision_to_egos = {}
avg_collision_to_others = {}
# egocollision的总次数占 全部collision次数的比例
ego_collision_rate = {}
# 发生碰撞的episode中，ego碰撞占总碰撞的比例
avg_ego_collision_rate = {}
avg_reward = {}
actor_ids = ["ego", "car1", "car2"]
for actor_id in actor_ids:
    print(actor_id+":")
    print("lateral_distance: ", sum(lateral_distance_rmse[actor_id])/len(lateral_distance_rmse[actor_id]))
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
    collision_times =  np.sum(total_collision)  if np.sum(total_collision) > 0 else 0.1;
    ego_collision_rate[actor_id] = np.sum(collision_to_ego_times[actor_id]) / collision_times
    print("ego_collision_rate:", ego_collision_rate[actor_id], end=", ")

    count = 0
    avg_ego_collision_rate[actor_id] = 0
    for i, value in enumerate(total_collision):
        if value > 0:
            count += 1
            avg_ego_collision_rate[actor_id] += (collision_to_ego_times[actor_id][i] / total_collision[i])
    if count > 0:
        avg_ego_collision_rate[actor_id] /= count
    print("avg_ego_collision_rate:", avg_ego_collision_rate[actor_id])

ray.shutdown()

'''
frameskip:true
time_to_goal defaultdict(<class 'list'>, {'ego': [171, 0, 0, 186, 0, 171, 174, 0, 0, 194, 171, 0, 173, 191, 197, 0, 0, 172, 188, 0, 0, 195, 171, 0, 172, 171, 0, 0, 171, 0, 0, 171, 170, 171, 171, 194, 201, 172, 0, 172, 171, 201, 0, 196, 171, 171, 0, 171, 172, 170, 0, 0, 171, 0, 188, 0, 0, 172, 171, 172, 188, 0, 171, 170, 201, 191, 187, 171, 171, 188, 0, 0, 171, 0, 199, 0, 171, 0, 171, 0, 0, 194, 0, 196, 195, 0, 199, 0, 198, 0, 0, 170, 193, 183, 172, 171, 171, 0, 172, 187], 'car1': [0, 153, 0, 147, 0, 194, 0, 0, 0, 158, 0, 0, 0, 156, 165, 0, 183, 0, 152, 166, 0, 159, 0, 166, 0, 0, 0, 0, 0, 181, 0, 0, 0, 0, 0, 0, 162, 0, 0, 0, 0, 169, 0, 167, 199, 0, 0, 0, 0, 0, 162, 0, 0, 0, 152, 0, 0, 0, 0, 0, 152, 0, 0, 0, 166, 158, 150, 0, 0, 152, 0, 171, 0, 170, 168, 0, 0, 159, 0, 0, 157, 154, 0, 162, 159, 179, 0, 169, 166, 0, 182, 0, 157, 148, 0, 200, 0, 0, 0, 150], 'car2': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]})
time_to_first_collision defaultdict(<class 'list'>, {'ego': [35, 31, 25, 0, 37, 0, 0, 36, 37, 0, 0, 82, 81, 34, 0, 34, 86, 83, 36, 29, 86, 96, 0, 80, 33, 0, 25, 36, 0, 86, 35, 37, 0, 0, 0, 0, 0, 86, 93, 81, 0, 81, 35, 67, 0, 0, 32, 0, 33, 0, 84, 38, 0, 26, 0, 35, 36, 66, 38, 0, 0, 25, 0, 37, 98, 109, 0, 35, 0, 0, 88, 30, 37, 29, 85, 32, 0, 27, 0, 30, 0, 0, 37, 0, 0, 28, 83, 0, 82, 36, 78, 97, 0, 0, 37, 85, 0, 30, 31, 0], 'car1': [0, 0, 24, 0, 83, 0, 107, 81, 100, 98, 0, 82, 81, 0, 0, 104, 85, 82, 0, 0, 85, 96, 0, 79, 0, 111, 26, 85, 0, 86, 116, 0, 0, 0, 0, 0, 0, 86, 93, 81, 0, 80, 110, 67, 0, 0, 83, 111, 0, 0, 83, 92, 0, 0, 0, 99, 118, 66, 0, 146, 0, 25, 0, 0, 98, 0, 0, 0, 155, 0, 88, 0, 194, 0, 85, 83, 0, 0, 0, 0, 0, 0, 117, 0, 0, 0, 82, 0, 82, 93, 77, 97, 0, 0, 0, 85, 117, 102, 0, 0], 'car2': [34, 31, 29, 0, 37, 0, 0, 35, 36, 0, 0, 120, 0, 33, 0, 34, 0, 0, 36, 28, 0, 0, 123, 0, 33, 111, 25, 36, 0, 0, 34, 36, 0, 0, 0, 105, 0, 0, 134, 175, 0, 0, 34, 0, 0, 0, 32, 111, 33, 0, 172, 37, 0, 25, 0, 34, 36, 0, 37, 0, 0, 54, 0, 36, 0, 109, 0, 35, 0, 0, 126, 29, 36, 29, 0, 32, 0, 26, 0, 30, 0, 0, 36, 0, 0, 28, 0, 178, 0, 35, 0, 197, 0, 0, 36, 0, 117, 30, 31, 0]})
final_reward defaultdict(<class 'list'>, {'ego': [10.199999999999976, 9.699999999999976, -7.000000000000001, 20.200000000000017, -1.600000000000004, 19.70000000000001, 19.70000000000001, -1.8000000000000052, 0.09999999999999631, 20.200000000000017, 20.200000000000017, 1.5999999999999879, 10.199999999999969, 10.199999999999976, 20.200000000000017, -4.135580766728708e-15, 9.699999999999969, 10.199999999999967, 10.199999999999976, 10.199999999999974, 9.699999999999969, 10.199999999999962, 19.70000000000001, 10.199999999999969, 10.199999999999976, 20.200000000000017, -17.3, -1.4000000000000048, 20.200000000000017, 10.199999999999966, 1.699999999999996, 9.69999999999998, 19.70000000000001, 20.200000000000017, 20.200000000000017, 20.200000000000017, 20.200000000000017, 10.199999999999966, 3.4999999999999845, 7.599999999999977, 20.200000000000017, 9.69999999999997, 1.0999999999999954, 9.699999999999976, 20.200000000000017, 20.200000000000017, -11.600000000000007, 20.200000000000017, 10.199999999999976, 20.200000000000017, 6.799999999999979, -1.2000000000000024, 20.200000000000017, 10.199999999999973, 19.70000000000001, -4.579669976578771e-15, 1.3999999999999961, 10.199999999999974, 9.69999999999998, 19.70000000000001, 19.70000000000001, -4.50000000000001, 20.200000000000017, 10.199999999999978, 10.199999999999962, 10.199999999999958, 19.70000000000001, 10.199999999999976, 20.200000000000017, 20.200000000000017, 2.1999999999999855, 10.199999999999974, 9.49999999999998, 10.199999999999974, 10.199999999999967, -11.600000000000007, 20.200000000000017, 10.199999999999973, 19.70000000000001, 10.199999999999974, 19.70000000000001, 20.200000000000017, 1.799999999999997, 20.200000000000017, 20.200000000000017, 10.199999999999973, 10.199999999999967, 20.200000000000017, 10.199999999999967, -0.6000000000000045, 9.699999999999973, 9.799999999999963, 20.200000000000017, 19.70000000000001, 10.199999999999978, 10.199999999999967, 19.70000000000001, 0.29999999999999366, 10.199999999999974, 20.200000000000017], 'car1': [49.12776417095603, 13.809163638383598, 108.82042003946748, 10.41500509068808, 125.38182007518401, 36.93014094040034, -48.056841817117665, 130.40147532640424, 20.236917795985214, 10.095730728194017, 48.462578377151125, 113.10277434237393, 142.82756778592145, 22.065706787351164, 25.106593608287728, 9.684279622656618, 140.6234187556783, 141.2108017143372, 12.148777418571417, 24.328656191765187, 81.20866079919244, 125.16727838244947, 43.0241273394608, 129.12535925621586, 38.744933033893666, 36.59386802947855, 105.22559931500314, 126.6007767154648, 27.486268345519072, 140.15259995540512, 6.43992037105054, 27.103440052450594, 45.74215133567051, 41.770687324128225, 45.79778002195203, 30.377086660458318, 21.580657941900984, 144.38545126854385, 119.48076122858566, 139.0996885830971, 43.11440668005272, 136.0449687809441, 1.1970204262963051, 131.31852279946952, 37.437624647919705, 39.355822894500335, 128.27378691429436, 11.320186967031157, 45.91016427872821, 45.732832542965525, 142.70462477827087, 135.41327879149435, 37.28552727124014, 29.904846708383968, 20.91213877030334, 1.0214747879691988, 9.43408950569496, 134.54728849242275, 27.53652710042385, -15.734800951428786, 16.445765674042576, 100.93920971127304, 25.890401629022985, 43.86382793978659, 127.91866033565925, 22.21664264252058, 17.13907069273681, 39.224180187980856, 33.79455515682648, 16.542299173587494, 116.28601148791326, 29.153465275894064, 15.101867210889061, 30.14146042641366, 134.79952685522215, 126.8481651800923, 46.75798936737179, 26.434161436118398, 47.780176101110094, 50.08540227853599, 20.59587168180749, 15.503408433148017, 13.45445153795876, 21.428032895541993, 21.428560988377253, 31.570218534219613, 124.58816449612613, 29.730791156642084, 132.0581350936153, 7.264796698682389, 133.76609196616695, 143.3660386737812, 24.923576645053267, 9.408425286849361, 44.92891024128653, 142.6089841159692, -35.706795040866055, 6.041279889486354, 36.41866505173225, 14.788291719442071], 'car2': [14.713588936585044, 73.16032929464185, 88.88513306092565, -33.44197605517981, 62.01889348241874, -50.56412497727794, -46.2867701769645, 61.96485040102839, 56.88094480834299, -41.75140488912831, -54.83245254311033, -56.36526529442152, -55.72631204796847, 52.79122319709358, -51.001267417945925, 48.18130330403767, -51.2077676272136, -50.42118065117913, -19.549322759276247, -0.5979522492306821, -73.23788625556702, -38.7513425765741, -41.55734403902317, -44.226639640454636, 23.169706773590516, -71.58035885988531, 90.58300588212498, 60.83773345379302, -92.28266732517275, -54.84538303766956, 43.47720331264475, 7.048131718296716, -68.41762784700087, -45.887904586477354, -57.59487280086375, -122.44064934400724, -47.667435490595025, -53.09882022070513, -60.7194047124752, -83.20960185852103, -53.73479647999702, -38.81028623437116, 48.76682837997214, -38.31377130162953, -55.397780373557985, -69.75321997910143, 60.56835014659513, -41.016974764729774, 25.504367373150107, -49.72013632995318, -69.59864604446912, 63.88929336575077, -50.57876133880467, 7.046452154719261, -42.33006192530003, 44.12465371155197, 43.483131237654604, -54.740600962564244, -28.750035076363783, -65.39241731915186, -43.209926403617345, 81.17079477716166, -90.11484469456995, 51.58669770965143, -50.462797202289345, 50.33718657306569, -35.186741206402026, 32.146980492119866, -65.49876802662914, -42.10488861918614, 64.85290922585565, 7.021332875786506, -4.123816651734803, 11.891740372654567, -57.69285301878065, 62.04900993552252, -55.99791191040636, 5.642741265179049, -57.673939401408695, 4.3128978010319114, -16.34590154826126, -45.83945007508232, 47.79246725699548, -38.289043564908724, -44.280664623207485, 0.41545469110219985, -46.14292800517504, -81.27649321079596, -58.02703522124666, 47.741861185153944, -57.145948498467725, -83.57771129365503, -41.612126419412874, -26.768011923465085, 48.13820516568483, -50.15196136938961, -63.369786697470126, 44.638838412780004, -21.37298788126269, -50.593880894507315]})
collision_to_ego_times defaultdict(<class 'list'>, {'ego': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'car1': [0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0], 'car2': [1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0]})
collision_to_others defaultdict(<class 'list'>, {'ego': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'car1': [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], 'car2': [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0]})
collision_to_vehicles defaultdict(<class 'list'>, {'ego': [1, 1, 2, 0, 2, 0, 0, 2, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 2, 2, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 2, 0, 1, 0, 1, 2, 0, 1, 0, 1, 1, 1, 1, 0, 0, 2, 0, 1, 1, 1, 0, 1, 0, 0, 2, 1, 1, 1, 1, 2, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0], 'car1': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0], 'car2': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]})
offlane_time defaultdict(<class 'list'>, {'ego': [0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0], 'car1': [0, 1, 0, 0, 0, 0, 1, 0, 1, 2, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1, 0, 0, 1, 0, 2, 0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 1, 0], 'car2': [1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1, 0]})
wholeExpTime: 0.6530023708608416
ego
collision_rate: 0.62, goal_rate: 0.63, avg_ttfc: 108.97, avg_ttg: 187.38, avg_offlane: 0.26, avg_collision_to_ego: 0.0, avg_collision_to_vehicles: 0.72, avg_collision_to_others: 0.0, avg_reward: 11.237999999999994, ego_collision_rate: 0.0, avg_ego_collision_rate: 0.0
car1
collision_rate: 0.49, goal_rate: 0.39, avg_ttfc: 147.06, avg_ttg: 186.2, avg_offlane: 0.25, avg_collision_to_ego: 0.32, avg_collision_to_vehicles: 0.14, avg_collision_to_others: 0.11, avg_reward: 57.60088917567555, ego_collision_rate: 0.5614035087719298, avg_ego_collision_rate: 0.5952380952380952
car2
collision_rate: 0.51, goal_rate: 0.0, avg_ttfc: 128.46, avg_ttg: 200.0, avg_offlane: 0.13, avg_collision_to_ego: 0.4, avg_collision_to_vehicles: 0.15, avg_collision_to_others: 0.15, avg_reward: -18.154274000405515, ego_collision_rate: 0.5714285714285714, avg_ego_collision_rate: 0.6045751633986928

'''