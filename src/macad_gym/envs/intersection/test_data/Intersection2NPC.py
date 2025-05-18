#continuous action, use-lstm, ppo, shared-layer fixed-delta-tick
#!/usr/bin/env python
'''
2 vehicle_4w,
early_terminate_on_collision: false
"send_measurements": True,
'''
import math
import time
import numpy as np
import matplotlib.pyplot as plt
import csv
from macad_gym.carla.multi_env import MultiCarlaEnv
from collections import defaultdict
'''
Enum('CameraType', ['rgb',
                'depth_raw',
                'depth',
                'semseg_raw',
                'semseg'])
'''
class Intersection2carTown03Continuous(MultiCarlaEnv):
    """A 4-way signalized intersection Multi-Agent Carla-Gym environment"""
    def __init__(self):
        self.configs = {
            "scenarios": "SSUI3C_TOWN3",
            "env": {
                "server_map": "/Game/Carla/Maps/Town03",
                "render": True,
                "render_x_res": 800,
                "render_y_res": 600,
                "x_res": 300,
                "y_res": 120,
                "framestack": 1,
                "discrete_actions": True,
                "squash_action_logits": False,
                "verbose": False,
                "use_depth_camera": False,
                "use_semantic_segmentation_camera": True,
                "send_measurements": True,
                "enable_planner": False,
                "spectator_loc": [170.5, 60, 30],
                "sync_server": True,
                "fixed_delta_seconds": 0.05,
                "using_imitation_model":False,
            },
            "actors": {
                "ego": {
                    "type": "vehicle_4W",
                    "enable_planner": True,
                    "convert_images_to_video": False,
                    "early_terminate_on_collision": False,
                    "reward_function": "custom",
                    "scenarios": "SSUI3C_TOWN3_CAR1",
                    "manual_control": False,
                    "auto_control": True,
                    "camera_type": "rgb",
                    "collision_sensor": "on",
                    "lane_sensor": "on",
                    "log_images": False,
                    "log_measurements": False,
                    "render": False,
                    "x_res": 300,
                    "y_res": 120,
                    "use_depth_camera": False,
                    "send_measurements": True,
                    "frame_skip": False
                },
                # "car1": {
                #     "type": "vehicle_2W",
                #     "enable_planner": True,
                #     "convert_images_to_video": False,
                #     "early_terminate_on_collision": False,
                #     "reward_function": "advrs",
                #     "scenarios": "SSUI3C_TOWN3_CAR1",
                #     "manual_control": False,
                #     "auto_control": True,
                #     "camera_type": "rgb",
                #     "collision_sensor": "on",
                #     "lane_sensor": "on",
                #     "log_images": False,
                #     "log_measurements": False,
                #     "render": False,
                #     "x_res": 300,
                #     "y_res": 120,
                #     "use_depth_camera": False,
                #     "send_measurements": True,
                #     "frame_skip": True
                # },
                "car2": {
                    "type": "vehicle_4W",
                    "enable_planner": True,
                    "convert_images_to_video": False,
                    "early_terminate_on_collision": False,
                    "reward_function": "advrs",
                    "scenarios": "SSUI3C_TOWN3_CAR2",
                    "manual_control": False,
                    "auto_control": True,
                    "camera_type": "rgb",
                    "collision_sensor": "on",
                    "lane_sensor": "on",
                    "log_images": False,
                    "log_measurements": False,
                    "render": False,
                    "x_res": 300,
                    "y_res": 120,
                    "use_depth_camera": False,
                    "send_measurements": True,
                    "frame_skip": False
                },
                "car3": {
                    "type": "vehicle_4W",
                    "enable_planner": True,
                    "convert_images_to_video": False,
                    "early_terminate_on_collision": False,
                    "reward_function": "advrs",
                    "scenarios": "SSUI3C_TOWN3_CAR3",
                    "manual_control": False,
                    "auto_control": True,
                    "camera_type": "rgb",
                    "collision_sensor": "on",
                    "lane_sensor": "on",
                    "log_images": False,
                    "log_measurements": False,
                    "render": False,
                    "x_res": 300,
                    "y_res": 120,
                    "use_depth_camera": False,
                    "send_measurements": True,
                    "frame_skip": False
                },
            },
        }
        super(Intersection2carTown03Continuous, self).__init__(self.configs)


if __name__ == "__main__":
    env = Intersection2carTown03Continuous()
    configs = env.configs
    speed_list = defaultdict(list)
    time_to_first_collision = defaultdict(list)
    time_to_goal = defaultdict(list)
    collision_times = defaultdict(list)
    collision_to_ego_times = defaultdict(list)

    start_num = 601
    episodes_num = 100

    speed_list = [5, 7.5, 10]
    episode_start_time = time.time()
    for v in speed_list:
        for ep in range(episodes_num):
            obs = env.reset()
            total_reward_dict = {}
            action_dict = {}
            env_config = configs["env"]
            actor_configs = configs["actors"]

            for actor_id in actor_configs.keys():
                env._behavior_agents[actor_id]._behavior.max_speed = v * 3.6
                if env._discrete_actions:
                    action_dict[actor_id] = 0  # Forward
                else:
                    action_dict[actor_id] = [0, 0]  # test values

            start = time.time()
            i = 0
            done = {"__all__": False}
            while not done["__all__"]:
                i += 1
                action_dict = env.action_space.sample()
                obs, reward, done, info = env.step(action_dict)

            print("episode_", ep, " ends")
            print("{} fps".format(i / (time.time() - start)))

            for actor_id in actor_configs.keys():
                if env._time_to_firstcollision[actor_id] is None:
                    time_to_first_collision[actor_id].append(0)
                else:
                    time_to_first_collision[actor_id].append(env._time_to_firstcollision[actor_id])
                if env._time_to_goal[actor_id] is None:
                    time_to_goal[actor_id].append(0)
                else:
                    time_to_goal[actor_id].append(env._time_to_goal[actor_id])
                collision_times[actor_id].append(
                    env._collisions[actor_id].collision_vehicles + env._collisions[actor_id].collision_other +
                    env._collisions[actor_id].collision_to_ego)
                collision_to_ego_times[actor_id].append(env._collisions[actor_id].collision_to_ego)
        total_collision_with_ego = []
        n = len(collision_to_ego_times['ego'])
        for i in range(n):
            total_sum = sum(collision_to_ego_times[vehicle][i] for vehicle in collision_to_ego_times.keys())
            total_collision_with_ego.append(total_sum)
        print("time_to_goal", str(time_to_goal['ego']))
        print("time_to_first_collision", str(time_to_first_collision['ego']))
        print("collision_to_ego_times", str(collision_to_ego_times))
        print("total_collision_to_ego", str(total_collision_with_ego))


        episodes = list(range(start_num,start_num+episodes_num))
        start_num = start_num + episodes_num
        speeds = [v for _ in range(episodes_num)]

        ego_position = [3 for _ in range(episodes_num)]
        # 打开一个新的CSV文件
        with open('Intersection_2NPC.csv', 'a', newline='') as file:
            writer = csv.writer(file)

            # # 写入列名（可选）
            # writer.writerow(["epi", "ego_position", "speed", "total_collision_to_ego", "ttfc", "ttg"])

            # 逐行写入数据
            for row in zip(episodes, ego_position, speeds, total_collision_with_ego, time_to_first_collision['ego'], time_to_goal['ego']):
                writer.writerow(row)

        print("successfully write csv")

    episode_end_time = time.time()
    print("total_time:", str(float(episode_end_time-episode_start_time)/3600))

