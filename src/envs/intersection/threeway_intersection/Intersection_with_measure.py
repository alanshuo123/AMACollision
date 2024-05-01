#!/usr/bin/env python
'''
2 vehicle_4w, 1 pedestrain, 1 motocycle
early_terminate_on_collision: false
"send_measurements": False,
'''
import time
import numpy as np
from macad_gym.carla.multi_env import MultiCarlaEnv

'''
Enum('CameraType', ['rgb',
                'depth_raw',
                'depth',
                'semseg_raw',
                'semseg'])
preprocess_image(image, config) set actor's x_res,y_res
'''
class Intersection2carTown03M(MultiCarlaEnv):
    """A 4-way signalized intersection Multi-Agent Carla-Gym environment"""
    def __init__(self):
        self.configs = {
            "scenarios": "SSUI3C_TOWN3",
            "env": {
                "server_map": "/Game/Carla/Maps/Town03",
                "render": False,
                "render_x_res": 800,
                "render_y_res": 600,
                "x_res": 84,
                "y_res": 84,
                "framestack": 1,
                "discrete_actions": True,
                "squash_action_logits": False,
                "verbose": False,
                "use_depth_camera": False,
                "use_semantic_segmentation_camera": True,
                "send_measurements": True,
                "enable_planner": True,
                "spectator_loc": [170.5, 80, 15],
                "sync_server": True,
                "fixed_delta_seconds": 0.05,
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
                    "auto_control": False,
                    "camera_type": "semseg",
                    "collision_sensor": "on",
                    "lane_sensor": "on",
                    "log_images": False,
                    "log_measurements": False,
                    "render": False,
                    "x_res": 84,
                    "y_res": 84,
                    "use_depth_camera": False,
                    "send_measurements": True,
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
                    "camera_type": "semseg",
                    "collision_sensor": "on",
                    "lane_sensor": "on",
                    "log_images": False,
                    "log_measurements": False,
                    "render": False,
                    "x_res": 84,
                    "y_res": 84,
                    "use_depth_camera": False,
                    "send_measurements": True,
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
                    "camera_type": "semseg",
                    "collision_sensor": "on",
                    "lane_sensor": "on",
                    "log_images": False,
                    "log_measurements": False,
                    "render": False,
                    "x_res": 84,
                    "y_res": 84,
                    "use_depth_camera": False,
                    "send_measurements": True,
                },
            },
        }
        super(Intersection2carTown03M, self).__init__(self.configs)


if __name__ == "__main__":
    env = Intersection2carTown03M()
    configs = env.configs

    for ep in range(2):
        obs = env.reset()

        total_reward_dict = {}
        action_dict = {}

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

        print(np.shape(obs["ego"]))
        print(obs["ego"])
        while not done["__all__"]:
            # while i < 20:  # TEST
            i += 1
            obs, reward, done, info = env.step(action_dict)
            # action_dict = get_next_actions(info, env.discrete_actions)
            print(np.array(obs))
            for actor_id in total_reward_dict.keys():
                total_reward_dict[actor_id] += reward[actor_id]
            # print("ego velocity:",env._actors["ego"].get_velocity().x,env._actors["ego"].get_velocity().y,env._actors["ego"].get_velocity().z)
            # print("ego location:", env._actors["ego"].get_location().x, env._actors["ego"].get_location().y,
            #       env._actors["ego"].get_location().z)
            # print(":{}\n\t".join(["Step#", "rew", "ep_rew",
            #                       "done{}"]).format(i, reward,
            #                                         total_reward_dict, done))

            time.sleep(0.1)

        print("{} fps".format(i / (time.time() - start)))
