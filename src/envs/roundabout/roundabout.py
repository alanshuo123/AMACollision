#continuous action, use-lstm, ppo, shared-layer fixed-delta-tick
#!/usr/bin/env python
'''
Town03_roundabout
'''
import time
import numpy as np
import matplotlib.pyplot as plt
import math
from macad_gym.carla.multi_env import MultiCarlaEnv
from collections import defaultdict
'''
Enum('CameraType', ['rgb',
                'depth_raw',
                'depth',
                'semseg_raw',
                'semseg'])
'''
class Roundabout(MultiCarlaEnv):
    """A 4-way signalized intersection Multi-Agent Carla-Gym environment"""
    def __init__(self):
        self.configs = {
            "scenarios": "ROUND_TOWN3",
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
                "spectator_loc": [0, 0, 50],
                "sync_server": True,
                "fixed_delta_seconds": 0.05,
                "using_imitation_model":True,
            },
            "actors": {
                "ego": {
                    "type": "vehicle_4W",
                    "enable_planner": False,
                    "convert_images_to_video": False,
                    "early_terminate_on_collision": False,
                    "reward_function": "custom",
                    "scenarios": "ROUND_TOWN3_CAR1",
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
                "car1": {
                    "type": "vehicle_4W",
                    "enable_planner": False,
                    "convert_images_to_video": False,
                    "early_terminate_on_collision": False,
                    "reward_function": "advrs",
                    "scenarios": "ROUND_TOWN3_CAR1",
                    "manual_control": False,
                    "auto_control": False,
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
                    "frame_skip": True
                },
                "car2": {
                    "type": "vehicle_4W",
                    "enable_planner": False,
                    "convert_images_to_video": False,
                    "early_terminate_on_collision": False,
                    "reward_function": "advrs",
                    "scenarios": "ROUND_TOWN3_CAR2",
                    "manual_control": False,
                    "auto_control": False,
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
                    "frame_skip": True
                },
                # "car3": {
                #     "type": "vehicle_4W",
                #     "enable_planner": False,
                #     "convert_images_to_video": False,
                #     "early_terminate_on_collision": False,
                #     "reward_function": "advrs",
                #     "scenarios": "ROUND_TOWN3_CAR3",
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
                #     "frame_skip": False
                # },
            },
        }
        super(Roundabout, self).__init__(self.configs)


if __name__ == "__main__":
    env = Roundabout()
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
        env_config = configs["env"]
        actor_configs = configs["actors"]

        for actor_id in actor_configs.keys():
            total_reward_dict[actor_id] = 0
            if env._discrete_actions:
                action_dict[actor_id] = 0  # Forward
            else:
                action_dict[actor_id] = [0, 0]  # test values

        start = time.time()
        i = 0
        done = {"__all__": False}
        while not done["__all__"]:
            # while i < 20:  # TEST
            i += 1
            action_dict = env.action_space.sample()
            # print(action_dict)
            obs, reward, done, info = env.step(action_dict)

            # **************action_dict = get_next_actions(info, env.discrete_actions)

            for actor_id in total_reward_dict.keys():
                # print(actor_id,env._prev_measurement[actor_id]["distance_to_goal"])
                total_reward_dict[actor_id] += reward[actor_id]
            # print(":{}\n\t".join(["Step#", "rew", "ep_rew",
            #                       "done{}"]).format(i, reward,
            #                                         total_reward_dict, done))

            # time.sleep(0.1)

        print("episode_", ep, " ends")
        print("{} fps".format(i / (time.time() - start)))

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

    print("time_to_goal", time_to_goal)
    print("time_to_first_collision", time_to_first_collision)
    print("final_reward", final_reward)
    print("collision_times", collision_times)
    print("offlane_time", offlane_times)
    print("collision_to_ego_times", collision_to_ego_times)
    print("collision_to_others", collision_to_others)
    print("collision_to_vehicles", collision_to_vehicles)
    print("lateral_distance_rmse", lateral_distance_rmse)
    import numpy as np

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
    actor_ids = ["ego", "car1", "car2", "car3"]
    for actor_id in actor_ids:
        print(actor_id + ":")
        print("lateral_distance: ", sum(lateral_distance_rmse[actor_id]) / len(lateral_distance_rmse[actor_id]))
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
        collision_times = np.sum(total_collision) if np.sum(total_collision) > 0 else 0.1;
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

    """
        ego
        collision_rate: 0.67, goal_rate: 0.49, avg_ttfc: 99.68, avg_ttg: 189.92, avg_offlane: 0.21, avg_collision_to_ego: 0.0, avg_collision_to_vehicles: 0.75, avg_collision_to_others: 0.0, avg_reward: 10.51999999999999, ego_collision_rate: 0.0, avg_ego_collision_rate: 0.0
        car1
        collision_rate: 0.49, goal_rate: 0.41, avg_ttfc: 146.84, avg_ttg: 187.61, avg_offlane: 0.4, avg_collision_to_ego: 0.28, avg_collision_to_vehicles: 0.15, avg_collision_to_others: 0.1, avg_reward: 56.71707247162206, ego_collision_rate: 0.5283018867924528, avg_ego_collision_rate: 0.5306122448979592
        car2
        collision_rate: 0.59, goal_rate: 0.0, avg_ttfc: 110.9, avg_ttg: 200.0, avg_offlane: 0.13, avg_collision_to_ego: 0.47, avg_collision_to_vehicles: 0.15, avg_collision_to_others: 0.15, avg_reward: -9.275892213115533, ego_collision_rate: 0.6103896103896104, avg_ego_collision_rate: 0.6610169491525424

        ego
        collision_rate: 0.61, goal_rate: 0.53, avg_ttfc: 107.53, avg_ttg: 190.73, avg_offlane: 0.2, avg_collision_to_ego: 0.0, avg_collision_to_vehicles: 0.73, avg_collision_to_others: 0.0, avg_reward: 10.096999999999994, ego_collision_rate: 0.0, avg_ego_collision_rate: 0.0
        car1
        collision_rate: 0.52, goal_rate: 0.43, avg_ttfc: 143.97, avg_ttg: 185.02, avg_offlane: 0.44, avg_collision_to_ego: 0.3, avg_collision_to_vehicles: 0.11, avg_collision_to_others: 0.18, avg_reward: 56.598570287135914, ego_collision_rate: 0.5084745762711864, avg_ego_collision_rate: 0.5345911949685535
        car2
        collision_rate: 0.5, goal_rate: 0.0, avg_ttfc: 123.68, avg_ttg: 200.0, avg_offlane: 0.14, avg_collision_to_ego: 0.43, avg_collision_to_vehicles: 0.11, avg_collision_to_others: 0.13, avg_reward: -9.563344711619365, ego_collision_rate: 0.6417910447761194, avg_ego_collision_rate: 0.71
        """