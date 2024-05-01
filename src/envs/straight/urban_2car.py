
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
            "scenarios": "STRAIGHT3C_TOWN04",
            "env": {
                "server_map": "/Game/Carla/Maps/Town04",
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
                "enable_planner": True,
                "spectator_loc": [170.5, 199, 30],
                "sync_server": True,
                "fixed_delta_seconds": 0.05,
                "using_imitation_model":False
            },
            "actors": {
                "ego": {
                    "type": "vehicle_4W",
                    "enable_planner": True,
                    "convert_images_to_video": False,
                    "early_terminate_on_collision": False,
                    "reward_function": "custom",
                    "scenarios": "SSUI3C_TOWN3_CAR1",
                    "manual_control": True,
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
                # "car1": {
                #     "type": "vehicle_4W",
                #     "enable_planner": True,
                #     "convert_images_to_video": False,
                #     "early_terminate_on_collision": True,
                #     "reward_function": "advrs",
                #     "scenarios": "SSUI3C_TOWN3_CAR1",
                #     "manual_control": False,
                #     "auto_control": False,
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
                # "car2": {
                #     "type": "vehicle_4W",
                #     "enable_planner": True,
                #     "convert_images_to_video": False,
                #     "early_terminate_on_collision": True,
                #     "reward_function": "advrs",
                #     "scenarios": "SSUI3C_TOWN3_CAR2",
                #     "manual_control": False,
                #     "auto_control": False,
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
                # "car3": {
                #     "type": "vehicle_4W",
                #     "enable_planner": True,
                #     "convert_images_to_video": False,
                #     "early_terminate_on_collision": True,
                #     "reward_function": "advrs",
                #     "scenarios": "SSUI3C_TOWN3_CAR3",
                #     "manual_control": False,
                #     "auto_control": False,
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
            },
        }
        super(Intersection2carTown03Continuous, self).__init__(self.configs)


if __name__ == "__main__":
    env = Intersection2carTown03Continuous()
    print("observation:",env.observation_space)
    configs = env.configs
    prev_result = defaultdict(lambda: defaultdict(list))
    for ep in range(10):
        obs = env.reset()
        total_reward_dict = {}
        action_dict = {}
        env_config = configs["env"]
        actor_configs = configs["actors"]

        ep_result = defaultdict(lambda: defaultdict(list))
        for actor_id in actor_configs.keys():
            ep_result[actor_id] = defaultdict(list)
            total_reward_dict[actor_id] = 0
            if env._discrete_actions:
                action_dict[actor_id] = 0  # Forward
            else:
                action_dict[actor_id] = [0, 0]  # test values

        start = time.time()
        i = 0
        done = {"__all__": False}
        prev_offlane=0
        prev_offroad=0
        while not done["__all__"]:
            # while i < 20:  # TEST
            i += 1
            action_dict = env.action_space.sample()
            # print(action_dict)
            obs, reward, done, info = env.step(action_dict)
            # print(obs)
            # action_dict = get_next_actions(info, env.discrete_actions)
            # print( "ego_collision:",env._prev_measurement["ego"]["collision_to_ego"] + env._prev_measurement["ego"]["collision_vehicles"])
            # if(env._prev_measurement["ego"]["intersection_offroad"]>prev_offroad or env._prev_measurement["ego"]["intersection_otherlane"]>prev_offlane):
            #     print("ego_offroad:", env._prev_measurement["ego"]["intersection_offroad"])
            #     print("ego_offline:", env._prev_measurement["ego"]["intersection_otherlane"])
            prev_offlane = env._prev_measurement["ego"]["intersection_offroad"]
            prev_offroad = env._prev_measurement["ego"]["intersection_otherlane"]
            # for actor_id in total_reward_dict.keys():
            #     print(actor_id, "-distance-to-ego:", env._prev_measurement[actor_id]["distance_to_goal"], "speed:",
            #           env._prev_measurement[actor_id]["forward_speed"])
            for actor_id in total_reward_dict.keys():
                print(actor_id,env._prev_measurement[actor_id]["distance_to_goal"])
                total_reward_dict[actor_id] += reward[actor_id]
                ep_result[actor_id]["reward"].append(total_reward_dict[actor_id])
                ep_result[actor_id]["distance_to_ego"].append(env._prev_measurement[actor_id]["distance_to_ego"])
                ep_result[actor_id]["collision_to_ego"].append(env._prev_measurement[actor_id]["collision_to_ego"])
                ep_result[actor_id]["collision_vehicles"].append(env._prev_measurement[actor_id]["collision_vehicles"])
                ep_result[actor_id]["collision_pedestrians"].append(
                    env._prev_measurement[actor_id]["collision_pedestrians"])
                ep_result[actor_id]["collision_other"].append(env._prev_measurement[actor_id]["collision_other"])
                ep_result[actor_id]["intersection_offroad"].append(
                    env._prev_measurement[actor_id]["intersection_offroad"])
                ep_result[actor_id]["intersection_otherlane"].append(
                    env._prev_measurement[actor_id]["intersection_otherlane"])
                ep_result[actor_id]["forward_speed"].append(env._prev_measurement[actor_id]["forward_speed"])
            # print("ego velocity:",env._actors["ego"].get_velocity().x,env._actors["ego"].get_velocity().y,env._actors["ego"].get_velocity().z)
            # print("ego location:", env._actors["ego"].get_location().x, env._actors["ego"].get_location().y,
            #       env._actors["ego"].get_location().z)
            print(":{}\n\t".join(["Step#", "rew", "ep_rew",
                                  "done{}"]).format(i, reward,
                                                    total_reward_dict, done))

            # time.sleep(0.1)

        print("episode ends")
        prev_result = ep_result
        print("{} fps".format(i / (time.time() - start)))

    length = len(prev_result["ego"]["reward"])
    step = np.arange(1, length + 1)

    fig, axs = plt.subplots(nrows=5, ncols=1, figsize=(30, 60), dpi=100)
    colors = ['red', 'green', 'blue']
    labels = ['ego', 'car2', 'car3']

    # reward
    axs[0].plot(step, prev_result["ego"]["reward"], c=colors[0], label=labels[0])
    # axs[0].scatter(step,prev_result["ego"]["reward"],c=colors[0])
    # print(prev_result["ego"]["reward"])
    axs[0].plot(step, prev_result["car2"]["reward"], c=colors[1], label=labels[1])
    # axs[0].scatter(step,prev_result["car2"]["reward"],c=colors[1])

    axs[0].plot(step, prev_result["car3"]["reward"], c=colors[2], label=labels[2])
    # axs[0].scatter(step,prev_result["car3"]["reward"],c=colors[2])
    axs[0].legend(loc='best')
    axs[0].set_xlabel('step', fontdict={'size': 16})
    axs[0].set_ylabel('reward', fontdict={'size': 16})

    # distance-to-ego
    axs[1].plot(step, prev_result["ego"]["distance_to_ego"], c=colors[0], label=labels[0])
    # axs[1].scatter(step,prev_result["ego"]["distance_to_ego"],c=colors[0])

    axs[1].plot(step, prev_result["car2"]["distance_to_ego"], c=colors[1], label=labels[1])
    # axs[1].scatter(step,prev_result["car2"]["distance_to_ego"],c=colors[1])

    axs[1].plot(step, prev_result["car3"]["distance_to_ego"], c=colors[2], label=labels[2])
    # axs[1].scatter(step,prev_result["car3"]["distance_to_ego"],c=colors[2])
    axs[1].legend(loc='best')
    axs[1].set_xlabel('step', fontdict={'size': 16})
    axs[1].set_ylabel('distance_to_ego', fontdict={'size': 16})

    # speed
    axs[2].plot(step, prev_result["ego"]["forward_speed"], c=colors[0], label=labels[0])
    axs[2].scatter(step, prev_result["ego"]["forward_speed"], c=colors[0], s=5)
    axs[2].legend(loc='best')

    axs[3].plot(step, prev_result["car2"]["forward_speed"], c=colors[1], label=labels[1])
    axs[3].scatter(step, prev_result["car2"]["forward_speed"], c=colors[1], s=5)
    axs[3].legend(loc='best')

    axs[4].plot(step, prev_result["car3"]["forward_speed"], c=colors[2], label=labels[2])
    axs[4].scatter(step, prev_result["car3"]["forward_speed"], c=colors[2], s=5)
    axs[4].legend(loc='best')

    axs[2].set_xlabel('step', fontdict={'size': 16})
    axs[2].set_ylabel('forward_speed', fontdict={'size': 16})

    # axs[2].set_facecolor("white")
    fig.autofmt_xdate()
    plt.show()