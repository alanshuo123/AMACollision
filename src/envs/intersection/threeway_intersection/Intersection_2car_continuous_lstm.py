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
                "using_imitation_model":True,
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
                "car1": {
                    "type": "vehicle_2W",
                    "enable_planner": True,
                    "convert_images_to_video": False,
                    "early_terminate_on_collision": False,
                    "reward_function": "advrs",
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
                    "frame_skip": True
                },
                "car2": {
                    "type": "vehicle_4W",
                    "enable_planner": True,
                    "convert_images_to_video": False,
                    "early_terminate_on_collision": False,
                    "reward_function": "advrs",
                    "scenarios": "SSUI3C_TOWN3_CAR2",
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
                "car3": {
                    "type": "vehicle_4W",
                    "enable_planner": True,
                    "convert_images_to_video": False,
                    "early_terminate_on_collision": False,
                    "reward_function": "advrs",
                    "scenarios": "SSUI3C_TOWN3_CAR3",
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
            },
        }
        super(Intersection2carTown03Continuous, self).__init__(self.configs)


if __name__ == "__main__":
    env = Intersection2carTown03Continuous()
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
    for ep in range(2):
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
        print("_lateral_list:" , env._lateral_distance)
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
    print("lateral_distance_rmse",lateral_distance_rmse)
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
    actor_ids = ["ego", "car1", "car2","car3"]
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

    # length = len(prev_result["ego"]["reward"])
    # step = np.arange(1, length + 1)
    #
    # fig, axs = plt.subplots(nrows=5, ncols=1, figsize=(30, 60), dpi=100)
    # colors = ['red', 'green', 'blue']
    # labels = ['ego', 'car2', 'car3']
    #
    # # reward
    # axs[0].plot(step, prev_result["ego"]["reward"], c=colors[0], label=labels[0])
    # # axs[0].scatter(step,prev_result["ego"]["reward"],c=colors[0])
    # # print(prev_result["ego"]["reward"])
    # axs[0].plot(step, prev_result["car2"]["reward"], c=colors[1], label=labels[1])
    # # axs[0].scatter(step,prev_result["car2"]["reward"],c=colors[1])
    #
    # axs[0].plot(step, prev_result["car3"]["reward"], c=colors[2], label=labels[2])
    # # axs[0].scatter(step,prev_result["car3"]["reward"],c=colors[2])
    # axs[0].legend(loc='best')
    # axs[0].set_xlabel('step', fontdict={'size': 16})
    # axs[0].set_ylabel('reward', fontdict={'size': 16})
    #
    # # distance-to-ego
    # axs[1].plot(step, prev_result["ego"]["distance_to_ego"], c=colors[0], label=labels[0])
    # # axs[1].scatter(step,prev_result["ego"]["distance_to_ego"],c=colors[0])
    #
    # axs[1].plot(step, prev_result["car2"]["distance_to_ego"], c=colors[1], label=labels[1])
    # # axs[1].scatter(step,prev_result["car2"]["distance_to_ego"],c=colors[1])
    #
    # axs[1].plot(step, prev_result["car3"]["distance_to_ego"], c=colors[2], label=labels[2])
    # # axs[1].scatter(step,prev_result["car3"]["distance_to_ego"],c=colors[2])
    # axs[1].legend(loc='best')
    # axs[1].set_xlabel('step', fontdict={'size': 16})
    # axs[1].set_ylabel('distance_to_ego', fontdict={'size': 16})
    #
    # # speed
    # axs[2].plot(step, prev_result["ego"]["forward_speed"], c=colors[0], label=labels[0])
    # axs[2].scatter(step, prev_result["ego"]["forward_speed"], c=colors[0], s=5)
    # axs[2].legend(loc='best')
    #
    # axs[3].plot(step, prev_result["car2"]["forward_speed"], c=colors[1], label=labels[1])
    # axs[3].scatter(step, prev_result["car2"]["forward_speed"], c=colors[1], s=5)
    # axs[3].legend(loc='best')
    #
    # axs[4].plot(step, prev_result["car3"]["forward_speed"], c=colors[2], label=labels[2])
    # axs[4].scatter(step, prev_result["car3"]["forward_speed"], c=colors[2], s=5)
    # axs[4].legend(loc='best')
    #
    # axs[2].set_xlabel('step', fontdict={'size': 16})
    # axs[2].set_ylabel('forward_speed', fontdict={'size': 16})
    #
    # # axs[2].set_facecolor("white")
    # fig.autofmt_xdate()
    # plt.show()

    '''
    framskip:true, behavioragent
    time_to_goal defaultdict(<class 'list'>, {'ego': [102, 0, 133, 101, 129, 0, 0, 102, 0, 111, 108, 0, 112, 0, 0, 0, 0, 110, 0, 0, 118, 0, 0, 0, 0, 0, 110, 0, 0, 0, 0, 0, 109, 140, 0, 0, 113, 0, 0, 0, 114, 107, 109, 0, 0, 0, 0, 141, 105, 132, 0, 106, 102, 0, 0, 148, 102, 0, 0, 0, 0, 102, 0, 0, 0, 111, 0, 0, 0, 0, 0, 101, 134, 0, 142, 135, 110, 102, 0, 110, 113, 0, 120, 113, 101, 0, 0, 109, 0, 108, 0, 124, 0, 0, 0, 0, 111, 0, 139, 0], 'car1': [0, 0, 0, 139, 0, 0, 134, 0, 0, 0, 0, 0, 120, 0, 0, 119, 0, 0, 118, 0, 0, 117, 0, 0, 0, 0, 0, 112, 0, 0, 112, 0, 0, 123, 0, 0, 0, 149, 0, 0, 143, 0, 0, 134, 0, 0, 143, 0, 0, 119, 0, 0, 0, 0, 0, 111, 0, 0, 0, 0, 0, 113, 0, 0, 120, 0, 0, 134, 147, 0, 110, 143, 0, 121, 0, 0, 0, 0, 0, 0, 0, 0, 0, 120, 0, 0, 131, 0, 0, 106, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'car2': [100, 0, 61, 93, 135, 0, 0, 0, 0, 112, 0, 131, 126, 73, 0, 0, 0, 117, 0, 0, 116, 0, 0, 66, 0, 0, 0, 0, 0, 73, 0, 0, 120, 97, 0, 0, 115, 0, 0, 0, 101, 148, 115, 0, 0, 0, 76, 79, 129, 68, 0, 116, 0, 0, 0, 103, 0, 0, 0, 0, 148, 103, 0, 79, 0, 121, 0, 115, 0, 67, 0, 132, 0, 0, 0, 0, 99, 97, 0, 114, 118, 0, 112, 0, 101, 0, 0, 103, 0, 101, 0, 65, 0, 92, 0, 0, 105, 0, 92, 0], 'car3': [62, 0, 80, 65, 0, 0, 0, 58, 0, 0, 80, 0, 64, 0, 0, 0, 0, 62, 0, 131, 69, 0, 0, 0, 0, 0, 63, 0, 0, 0, 0, 58, 0, 63, 0, 0, 0, 0, 0, 0, 64, 75, 57, 0, 65, 0, 0, 103, 0, 99, 105, 68, 0, 0, 0, 67, 62, 70, 0, 0, 0, 53, 0, 0, 0, 62, 0, 0, 0, 0, 0, 60, 87, 0, 0, 0, 65, 62, 0, 64, 0, 0, 140, 62, 121, 0, 0, 69, 0, 67, 0, 110, 0, 117, 0, 0, 63, 0, 61, 0]})
    time_to_first_collision defaultdict(<class 'list'>, {'ego': [0, 45, 0, 0, 0, 46, 35, 0, 117, 0, 46, 41, 0, 40, 39, 0, 44, 0, 43, 38, 0, 44, 40, 64, 74, 41, 0, 0, 134, 114, 0, 0, 0, 0, 37, 55, 0, 39, 37, 63, 45, 0, 0, 0, 41, 0, 45, 39, 0, 0, 39, 0, 0, 36, 51, 70, 0, 36, 0, 35, 36, 0, 0, 41, 53, 0, 0, 70, 45, 40, 93, 0, 0, 35, 0, 0, 0, 0, 0, 0, 0, 0, 43, 0, 0, 145, 0, 0, 49, 45, 48, 44, 0, 45, 34, 45, 0, 0, 0, 45], 'car1': [0, 0, 0, 0, 0, 0, 0, 36, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 55, 0, 37, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 39, 37, 0, 0, 34, 0, 0, 0, 34, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'car2': [0, 0, 37, 0, 0, 45, 0, 36, 38, 0, 0, 0, 0, 40, 0, 55, 44, 0, 43, 39, 0, 38, 0, 0, 85, 42, 54, 0, 55, 0, 0, 39, 61, 0, 37, 55, 0, 40, 38, 55, 0, 0, 0, 0, 41, 135, 36, 40, 47, 37, 40, 0, 34, 36, 0, 70, 33, 36, 44, 65, 37, 0, 0, 41, 34, 0, 0, 36, 0, 41, 46, 0, 136, 36, 35, 44, 0, 0, 39, 0, 0, 0, 87, 50, 58, 0, 42, 0, 0, 0, 0, 44, 0, 45, 53, 0, 0, 0, 0, 0], 'car3': [0, 45, 37, 0, 101, 44, 36, 0, 38, 83, 46, 41, 0, 67, 39, 55, 46, 0, 43, 147, 0, 37, 40, 65, 74, 44, 0, 0, 36, 114, 0, 39, 61, 0, 45, 121, 84, 0, 91, 54, 45, 0, 0, 0, 0, 134, 36, 0, 38, 0, 55, 0, 76, 95, 52, 0, 0, 0, 43, 35, 0, 0, 0, 0, 34, 0, 0, 36, 45, 70, 46, 0, 0, 38, 35, 44, 0, 0, 38, 0, 84, 0, 44, 0, 58, 145, 42, 0, 49, 46, 48, 62, 0, 44, 34, 45, 0, 0, 0, 45]})
    final_reward defaultdict(<class 'list'>, {'ego': [14.199999999999967, 4.1999999999999975, 14.199999999999967, 14.199999999999967, 14.199999999999967, 4.199999999999998, 4.199999999999994, 14.199999999999967, 4.1999999999999815, 14.199999999999967, 4.199999999999998, 4.199999999999996, 14.199999999999967, -5.800000000000038, 4.199999999999996, 14.199999999999967, -5.800000000000031, 14.199999999999967, 4.1999999999999975, 4.199999999999995, 14.199999999999967, 4.1999999999999975, 4.199999999999996, 4.199999999999999, -5.800000000000029, -5.800000000000031, 14.199999999999967, 14.199999999999967, 2.4999999999999734, 4.199999999999982, 14.199999999999967, 14.199999999999967, 13.699999999999967, 14.199999999999967, 4.199999999999994, 4.199999999999999, 14.199999999999967, 4.199999999999996, 4.199999999999994, 4.199999999999999, 4.1999999999999975, 14.199999999999967, 14.199999999999967, 14.199999999999967, 4.199999999999996, 14.199999999999967, 4.1999999999999975, 4.199999999999996, 14.199999999999967, 14.199999999999967, 4.199999999999996, 14.199999999999967, 14.199999999999967, 4.199999999999994, 4.199999999999999, 4.199999999999999, 14.199999999999967, 4.199999999999994, 14.199999999999967, 4.199999999999994, 4.199999999999994, 14.199999999999967, 14.199999999999967, 4.199999999999996, 4.199999999999999, 14.199999999999967, 14.199999999999967, 4.199999999999999, 4.1999999999999975, -5.800000000000038, 4.19999999999999, 14.199999999999967, 14.199999999999967, 4.199999999999994, 14.199999999999967, 14.199999999999967, 14.199999999999967, 14.199999999999967, 14.199999999999967, 14.199999999999967, 14.199999999999967, 14.199999999999967, 4.1999999999999975, 14.199999999999967, 14.199999999999967, 4.199999999999969, 14.199999999999967, 14.199999999999967, 3.6999999999999975, 4.1999999999999975, 4.199999999999999, -5.800000000000036, 14.199999999999967, 4.1999999999999975, 4.199999999999993, 4.1999999999999975, 14.199999999999967, 14.199999999999967, 14.199999999999967, 4.1999999999999975], 'car1': [17.158366362647957, 21.841422773036797, 24.015080885532136, 22.60277948427757, -7.309313548274368, 26.07290168678884, 25.51683669218262, -83.57271154396918, 27.218176048060624, 14.915423597967747, 15.729540185137822, 20.315502724608002, 15.088643861183916, 27.480105040791095, 16.012560544174065, 20.79805664482904, 18.515863841704306, 0.8912608601608618, 15.78986574769139, 16.183150609337844, 10.949780926348364, 16.18995930107382, 17.648891772682273, 20.219681559990043, 10.694426475248516, 27.759855868205385, -89.4255038878764, 14.432185592914536, 16.090440466255117, 20.45228611152376, 11.604983354460515, 18.352085078364823, 16.471275632675077, 20.992991935887726, 16.923241117024396, -60.86567700073724, 1.9543034930887773, 30.729057337598398, -58.858222328133145, 27.7628161235819, 32.24836296544172, 15.305555631755553, 24.235405089074334, 32.71271249448451, 17.767387338828097, 19.915024297737126, 36.07091688929967, 17.95978192313715, -8.944785925907594, -3.7376660859022017, 25.734816818896903, 8.740321665929061, -80.5540170412985, -63.68572246595623, 27.443645954849387, 10.942545305619557, 12.20833448322048, 21.32134069328029, -62.635974354494586, 15.435662529862729, 21.231203104108452, 11.988158562479422, 14.630596211778794, 26.495171939419194, 17.731477161339264, 14.039957547278302, -62.44533277114454, 24.888651114769022, 31.037276423329097, 18.987984752312695, 15.616784181605253, 24.728250271922434, 19.604226226745027, 18.64921384962873, -60.0322852510574, 20.50544971181616, 16.900486340834743, 15.212399134767933, 20.84141382284379, 25.611227932925637, 3.421383424976752, 16.189938532433338, 12.070537228252622, 18.284336839013633, 15.712943467647463, 20.34115661396332, 29.40348504041971, 16.25248691211242, -58.98888771793119, 6.921977144645785, 21.962664218869016, 25.893119936117923, -60.028126251722476, 20.77590269265393, 19.662901739419034, -62.70928674102392, 15.75908143111627, -50.20757667936562, 16.697312407239945, -60.396386594759946], 'car2': [-15.509983113487069, -34.95973664413057, -59.493058789326476, -25.81365604411404, 0.91864841842259, -45.22968291592932, -32.21766139478782, -83.53060164217649, 54.62795037692447, -6.664536270663735, 23.71929932154675, 11.550540197243743, 20.02616199589451, 60.243380516834534, -45.57729603312319, -42.84423189117426, 61.90955037837378, -5.911451374297874, -49.58689958682105, 63.79399328996063, -10.031180674216793, -47.14321800497497, -28.177545756075432, -53.750161795637524, 65.2339813754028, 58.97602127725797, -78.5258404971592, -37.24917725612661, -23.58007568767261, -39.70921645269914, -23.126134903397897, -38.303541641727165, -11.246815354055355, -19.435284591060817, 52.22174450330693, 66.2787368942938, -13.153480835801476, 59.629682273762135, 59.99926062997967, 52.06373354791303, 4.958879055502501, 27.957941038732486, 0.7868392482462427, -46.01247142797472, 63.42134383409013, -36.60502532678878, 57.07417911690593, 69.65505035674079, -3.5618774451332547, -50.74312905372642, 52.04718588433344, -18.181236745735077, -80.86658888836004, 51.25837736085933, -37.10201312916178, 78.94807839944573, -10.625120955847553, 58.83552184221933, -41.13103900725555, -45.63122778018782, 111.80846060882963, -13.343358503479646, -43.671462319687215, 70.81847973878163, -47.803027513242675, 5.005587326540892, -45.024939794197294, 81.77473967475439, -35.187662631950595, 49.89286700028793, 51.83262524138551, 16.20732990193005, -11.010139296636536, 50.57887343107215, -51.31317087643736, 2.1683424246039893, -13.844848025441701, -21.830755348998697, -49.06634979076572, 5.3749249076886185, -4.517672276128523, -40.842147286006885, -21.53043652203583, -54.565161724715374, -17.430277739960225, -46.60471216431274, -39.646495471732514, -13.038375163023439, -43.08351708711017, -23.12722169656952, -46.513756758412654, 57.719384741450604, -37.883268391194356, 65.04336970130728, -49.427799070038226, -45.05813786107813, -18.768685522053115, -39.5465449751295, -27.485334731284066, -43.173194838964285], 'car3': [-60.26179603538243, 59.61409785401647, -61.42207997335025, -56.23012769892295, 4.335763800093648, 53.693997065004204, 57.823362560779, -58.65930583006958, -46.23283938664847, -59.823888879193824, 58.88353211240091, 62.733679325086264, -58.13535079179242, 71.41188652521404, 59.09279490831057, -52.826968419937714, 61.11703332524926, -58.89151990821172, 49.27343222535913, 2.8288438629477, -52.698274792900186, 52.893497225533196, 60.925697219337295, 60.539434791520534, 62.31142013859173, 59.16267140075117, -58.91520558690689, -47.722082654029414, -37.366622107291604, 71.25221959849932, -47.751132652761015, -76.33390784465121, -32.41313298695271, -59.28470833346309, -48.69150642946325, -35.01261413407766, -48.93244978807438, -46.93639738439939, -18.58113018763767, -51.65146671465902, 37.96607164555774, -46.539145830387504, -62.66622364483938, -31.694261611282613, -54.300006563743175, -44.4413921307317, -44.255963630977035, -18.72278489109902, -58.581523139256376, -19.67456501167829, -13.93512177617745, -51.585058478705896, -44.88100755610071, -50.26971483135429, 58.26047656330801, -57.57188341887725, -62.97648483094213, -49.98171189571675, -56.37661055272338, 50.680246766210054, -21.793277832158587, -69.5172546741857, -27.76724841737858, -41.300171013706624, 55.5106868333512, -64.35980946144943, -43.84873494459603, -61.40758675411167, 62.041694638206195, 87.07638273561557, -49.62020373711399, -58.9562512536807, -39.53982228684702, -53.233811801752545, -35.589844928523874, -32.59298176817114, -46.13162803933515, -62.28005391683678, -54.54964418179828, -60.65561579270152, -40.64757950142927, -37.31732371116396, 99.13057369705703, -59.22578136708191, -23.614698880878287, 65.50925140137896, 13.054942749624287, -53.15113292939487, 60.0347262893839, 45.307887159604306, 61.2161708474783, 80.36862531017042, -48.531810931741994, -31.72314643685991, 47.13081184242116, 62.32417344628436, -58.284623767331034, -37.52109406738932, -59.7049356861796, 59.79021060186943]})
    collision_times defaultdict(<class 'list'>, {'ego': [0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 2, 1, 0, 2, 0, 1, 1, 0, 1, 1, 1, 2, 2, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 2, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 2, 0, 1, 1, 1, 0, 0, 0, 1], 'car1': [0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 2, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'car2': [0, 0, 1, 0, 0, 1, 0, 1, 2, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 2, 0, 0, 1, 1, 0, 2, 1, 0, 1, 1, 2, 0, 0, 0, 0, 1, 1, 2, 1, 1, 1, 2, 0, 1, 2, 0, 1, 2, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 2, 0, 1, 2, 0, 1, 2, 1, 2, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 2, 1, 0, 0, 0, 0, 0], 'car3': [0, 1, 1, 0, 1, 2, 1, 0, 1, 1, 1, 2, 0, 1, 1, 1, 1, 0, 2, 1, 0, 2, 1, 1, 1, 1, 0, 0, 3, 1, 0, 1, 2, 0, 1, 2, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 3, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 2, 0, 0, 0, 0, 2, 0, 0, 2, 1, 1, 1, 0, 0, 1, 1, 2, 0, 0, 1, 0, 1, 0, 2, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 2, 1, 0, 0, 0, 1]})
    offlane_time defaultdict(<class 'list'>, {'ego': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'car1': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'car2': [1, 0, 12, 7, 2, 0, 0, 0, 0, 9, 1, 1, 1, 5, 0, 0, 0, 6, 0, 0, 6, 0, 0, 17, 0, 0, 0, 0, 0, 15, 0, 0, 4, 11, 0, 0, 5, 0, 0, 0, 8, 2, 7, 0, 0, 0, 13, 16, 0, 11, 0, 8, 0, 0, 0, 9, 4, 0, 0, 0, 2, 11, 0, 9, 0, 7, 0, 5, 0, 12, 0, 4, 3, 0, 0, 5, 4, 12, 0, 2, 7, 0, 5, 0, 0, 0, 0, 5, 0, 7, 1, 18, 0, 8, 0, 0, 5, 0, 9, 0], 'car3': [7, 0, 7, 11, 4, 0, 0, 9, 0, 2, 6, 0, 10, 0, 0, 0, 0, 13, 0, 1, 4, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 10, 0, 6, 0, 4, 3, 0, 4, 0, 13, 10, 15, 0, 7, 0, 0, 10, 2, 2, 0, 9, 3, 0, 0, 8, 6, 7, 0, 0, 0, 14, 0, 0, 0, 11, 0, 2, 0, 0, 0, 7, 10, 0, 0, 4, 11, 10, 0, 9, 1, 0, 2, 2, 4, 0, 3, 7, 0, 6, 0, 3, 0, 3, 0, 0, 10, 0, 11, 0]})
    collision_to_ego_times defaultdict(<class 'list'>, {'ego': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'car1': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'car2': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0], 'car3': [0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1]})
    collision_to_others defaultdict(<class 'list'>, {'ego': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'car1': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'car2': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'car3': [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 2, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]})
    collision_to_vehicles defaultdict(<class 'list'>, {'ego': [0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 2, 1, 0, 2, 0, 1, 1, 0, 1, 1, 1, 2, 2, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 2, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 2, 0, 1, 1, 1, 0, 0, 0, 1], 'car1': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'car2': [0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0], 'car3': [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 2, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]})
    ego
    collision_rate: 0.51, goal_rate: 0.42, avg_ttfc: 124.69, avg_ttg: 164.49, avg_offlane: 0.0, avg_collision_to_ego: 0.0, avg_collision_to_vehicles: 0.57, avg_collision_to_others: 0.0, avg_reward: 8.472999999999978, ego_collision_rate: 0.0, avg_ego_collision_rate: 0.0
    car1
    collision_rate: 0.07, goal_rate: 0.25, avg_ttfc: 188.72, avg_ttg: 181.38, avg_offlane: 0.0, avg_collision_to_ego: 0.0, avg_collision_to_vehicles: 0.08, avg_collision_to_others: 0.03, avg_reward: 6.47067293547683, ego_collision_rate: 0.0, avg_ego_collision_rate: 0.0
    car2
    collision_rate: 0.53, goal_rate: 0.43, avg_ttfc: 119.74, avg_ttg: 158.34, avg_offlane: 3.22, avg_collision_to_ego: 0.27, avg_collision_to_vehicles: 0.35, avg_collision_to_others: 0.04, avg_reward: -4.412075864585689, ego_collision_rate: 0.4090909090909091, avg_ego_collision_rate: 0.4056603773584906
    car3
    collision_rate: 0.61, goal_rate: 0.39, avg_ttfc: 113.14, avg_ttg: 151.53, avg_offlane: 3.3, avg_collision_to_ego: 0.3, avg_collision_to_vehicles: 0.31, avg_collision_to_others: 0.16, avg_reward: -13.627977399329525, ego_collision_rate: 0.38961038961038963, avg_ego_collision_rate: 0.4262295081967213

    '''