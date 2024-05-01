#continuous action, use-lstm, ppo, shared-layer fixed-delta-tick
#!/usr/bin/env python
'''
2 vehicle_4w,
early_terminate_on_collision: false
"send_measurements": True,

'''
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
                "send_measurements": True,
                "enable_planner": True,
                "spectator_loc": [170.5, 60, 30],
                "sync_server": True,
                "fixed_delta_seconds": 0.05,
                "using_imitation_model":False,
                "frame_skip":True
            },
            "actors": {
                "ego": {
                    "type": "vehicle_4W",
                    "enable_planner": True,
                    "convert_images_to_video": False,
                    "early_terminate_on_collision": True,
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
                },
                # "car1": {
                #     "type": "vehicle_2W",
                #     "enable_planner": True,
                #     "convert_images_to_video": False,
                #     "early_terminate_on_collision": True,
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
                # },
                "car2": {
                    "type": "vehicle_4W",
                    "enable_planner": True,
                    "convert_images_to_video": False,
                    "early_terminate_on_collision": True,
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
                },
                "car3": {
                    "type": "vehicle_4W",
                    "enable_planner": True,
                    "convert_images_to_video": False,
                    "early_terminate_on_collision": True,
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

        print("episode_",ep ," ends")
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
            collision_times[actor_id].append(env._collisions[actor_id].collision_vehicles+env._collisions[actor_id].collision_other+env._collisions[actor_id].collision_to_ego)
            collision_to_ego_times[actor_id].append(env._collisions[actor_id].collision_to_ego)
            collision_to_vehicles[actor_id].append(env._collisions[actor_id].collision_vehicles)
            collision_to_others[actor_id].append(env._collisions[actor_id].collision_other)
            offlane_times[actor_id].append(env._lane_invasions[actor_id].offlane)

    print("time_to_goal",time_to_goal)
    print("time_to_first_collision",time_to_first_collision)
    print("final_reward",final_reward)
    print("collision_times",collision_times)
    print("offlane_time",offlane_times)
    print("collision_to_ego_times", collision_to_ego_times)
    print("collision_to_others", collision_to_others)
    print("collision_to_vehicles", collision_to_vehicles)
    # length = len(time_to_goal["ego"])
    #
    # step = np.arange(1, length + 1)
    #
    # fig, axs = plt.subplots(nrows=5, ncols=1, figsize=(30, 60), dpi=100)
    # colors = ['red', 'green', 'blue','yellow']
    # labels = ['ego', 'car1','car2', 'car3']
    #
    # # reward
    # axs[0].plot(step, time_to_goal["ego"], c=colors[0], label=labels[0])
    # # axs[0].scatter(step,time_to_goal["ego"]["reward"],c=colors[0])
    #
    # axs[0].plot(step, time_to_goal["car1"], c=colors[1], label=labels[1])
    # # axs[0].scatter(step,time_to_goal["ego"]["reward"],c=colors[0])
    #
    # axs[0].plot(step, time_to_goal["car2"], c=colors[2], label=labels[2])
    # # axs[0].scatter(step,time_to_goal["ego"]["reward"],c=colors[0])
    #
    # axs[0].plot(step, time_to_goal["car3"], c=colors[3], label=labels[3])
    # # axs[0].scatter(step,time_to_goal["ego"]["reward"],c=colors[0])
    #
    # axs[0].legend(loc='best')
    # axs[0].set_xlabel('episode', fontdict={'size': 16})
    # axs[0].set_ylabel('time_to_goal', fontdict={'size': 16})
    #
    # # reward
    # axs[1].plot(step, time_to_first_collision["ego"], c=colors[0], label=labels[0])
    # # axs[0].scatter(step,time_to_goal["ego"]["reward"],c=colors[0])
    # axs[1].plot(step, time_to_first_collision["car1"], c=colors[1], label=labels[1])
    # # axs[0].scatter(step,time_to_goal["ego"]["reward"],c=colors[0])
    # axs[1].plot(step, time_to_first_collision["car2"], c=colors[2], label=labels[2])
    # # axs[0].scatter(step,time_to_goal["ego"]["reward"],c=colors[0])
    # axs[1].plot(step, time_to_first_collision["car3"], c=colors[3], label=labels[3])
    # # axs[0].scatter(step,time_to_goal["ego"]["reward"],c=colors[0])
    # axs[1].legend(loc='best')
    # axs[1].set_xlabel('episode', fontdict={'size': 16})
    # axs[1].set_ylabel('time_to_first_collision', fontdict={'size': 16})
    #
    #
    #
    # # reward
    # axs[2].plot(step, final_reward["ego"], c=colors[0], label=labels[0])
    # # axs[0].scatter(step,time_to_goal["ego"]["reward"],c=colors[0])
    # axs[2].plot(step, final_reward["car1"], c=colors[1], label=labels[1])
    # # axs[0].scatter(step,time_to_goal["ego"]["reward"],c=colors[0])
    # axs[2].plot(step, final_reward["car2"], c=colors[2], label=labels[2])
    # # axs[0].scatter(step,time_to_goal["ego"]["reward"],c=colors[0])
    # axs[2].plot(step, final_reward["car3"], c=colors[3], label=labels[3])
    # # axs[0].scatter(step,time_to_goal["ego"]["reward"],c=colors[0])
    # axs[2].legend(loc='best')
    # axs[2].set_xlabel('episode', fontdict={'size': 16})
    # axs[2].set_ylabel('total_reward', fontdict={'size': 16})
    #
    # # speed
    # # reward
    # axs[3].plot(step, collision_times["ego"], c=colors[0], label=labels[0])
    # # axs[0].scatter(step,time_to_goal["ego"]["reward"],c=colors[0])
    # axs[3].plot(step, collision_times["car1"], c=colors[1], label=labels[1])
    # # axs[0].scatter(step,time_to_goal["ego"]["reward"],c=colors[0])
    # axs[3].plot(step, collision_times["car2"], c=colors[2], label=labels[2])
    # # axs[0].scatter(step,time_to_goal["ego"]["reward"],c=colors[0])
    # axs[3].plot(step, collision_times["car3"], c=colors[3], label=labels[3])
    # # axs[0].scatter(step,time_to_goal["ego"]["reward"],c=colors[0])
    # axs[3].legend(loc='best')
    # axs[3].set_xlabel('episode', fontdict={'size': 16})
    # axs[3].set_ylabel('collision_times', fontdict={'size': 16})
    #
    #
    # # reward
    # axs[4].plot(step, offlane_times["ego"], c=colors[0], label=labels[0])
    # # axs[0].scatter(step,time_to_goal["ego"]["reward"],c=colors[0])
    # axs[4].plot(step, offlane_times["car1"], c=colors[1], label=labels[1])
    # # axs[0].scatter(step,time_to_goal["ego"]["reward"],c=colors[0])
    # axs[4].plot(step, offlane_times["car2"], c=colors[2], label=labels[2])
    # # axs[0].scatter(step,time_to_goal["ego"]["reward"],c=colors[0])
    # axs[4].plot(step, offlane_times["car3"], c=colors[3], label=labels[3])
    # # axs[0].scatter(step,time_to_goal["ego"]["reward"],c=colors[0])
    # axs[4].legend(loc='best')
    # axs[4].set_xlabel('episode', fontdict={'size': 16})
    # axs[4].set_ylabel('offlane_times', fontdict={'size': 16})
    #
    # # axs[2].set_facecolor("white")
    # fig.autofmt_xdate()
    # plt.show()

