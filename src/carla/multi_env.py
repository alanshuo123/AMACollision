"""
multi_env.py: Multi-agent gym environment interface for CARLA
support manually config action_space & observation_space
__author__: @TianShuo
"""
from collections import defaultdict
import argparse
import atexit
import shutil
from datetime import datetime
import logging
import json
import os
import random
import signal
import subprocess
import sys
import time
import traceback
import socket
import math

import cv2
import tensorflow as tf
import numpy as np  # linalg.norm is used
import GPUtil
from gym.spaces import Box, Discrete, Tuple, Dict
import pygame
import carla

from macad_gym.core.controllers.traffic import apply_traffic
from macad_gym.multi_actor_env import MultiActorEnv
from macad_gym import LOG_DIR
from macad_gym.core.sensors.utils import preprocess_image
from macad_gym.core.maps.nodeid_coord_map import MAP_TO_COORDS_MAPPING
from macad_gym.agents.imitationmodel.imitation_learning import ImitationLearning
# from macad_gym.core.sensors.utils import get_transform_from_nearest_way_point
from macad_gym.carla.reward import Reward
from macad_gym.core.sensors.hud import HUD
from macad_gym.viz.render import Render
from macad_gym.carla.scenarios import Scenarios
from macad_gym.carla.Utils import calculate_forward_velocity
# The following imports require carla to be imported already.
from macad_gym.core.sensors.camera_manager import CameraManager, CAMERA_TYPES
from macad_gym.core.sensors.derived_sensors import LaneInvasionSensor
from macad_gym.core.sensors.derived_sensors import CollisionSensor
from macad_gym.core.controllers.keyboard_control import KeyboardControl
from macad_gym.carla.PythonAPI.agents.navigation.global_route_planner_dao import (  # noqa: E501
    GlobalRoutePlannerDAO,
)
from macad_gym.carla.PythonAPI.agents.navigation.behavior_agent import BehaviorAgent
from macad_gym.carla.Utils import calculate_distance
from macad_gym.carla.PythonAPI.agents.tools.misc import draw_waypoints,calculate_lateral_distance
# The following imports depend on these paths being in sys path
# TODO: Fix this. This probably won't work after packaging/distribution
sys.path.append("src/macad_gym/carla/PythonAPI")
from macad_gym.core.maps.nav_utils import PathTracker
from macad_gym.carla.PythonAPI.agents.navigation.global_route_planner_revised import (
    GlobalRoutePlannerv1,
)
from macad_gym.carla.PythonAPI.agents.navigation.local_planner import (
    RoadOption,
)

logger = logging.getLogger(__name__)
# Set this where you want to save image outputs (or empty string to disable)
CARLA_OUT_PATH = os.environ.get("CARLA_OUT", os.path.expanduser("~/carla_out"))
if CARLA_OUT_PATH and not os.path.exists(CARLA_OUT_PATH):
    os.makedirs(CARLA_OUT_PATH)

# Set this to the path of your Carla binary
SERVER_BINARY = os.environ.get(
    "CARLA_SERVER", os.path.expanduser("/media/simplexity/d7591000-e74c-4eb0-a7f9-3b3780ea1e09/CARLA_0.9.13/CarlaUE4.sh")
)

# Check if is using on Windows
IS_WINDOWS_PLATFORM = "win" in sys.platform

assert os.path.exists(SERVER_BINARY), (
    "Make sure CARLA_SERVER environment"
    " variable is set & is pointing to the"
    " CARLA server startup script (Carla"
    "UE4.sh). Refer to the README file/docs."
)

# Carla planner commands
COMMANDS_ENUM = {
    0.0: "REACH_GOAL",
    5.0: "GO_STRAIGHT",
    4.0: "TURN_RIGHT",
    3.0: "TURN_LEFT",
    2.0: "LANE_FOLLOW",
}

# Mapping from string repr to one-hot encoding index to feed to the model
COMMAND_ORDINAL = {
    "REACH_GOAL": 0,
    "GO_STRAIGHT": 1,
    "TURN_RIGHT": 2,
    "TURN_LEFT": 3,
    "LANE_FOLLOW": 4,
}

ROAD_OPTION_TO_COMMANDS_MAPPING = {
    RoadOption.VOID: "REACH_GOAL",
    RoadOption.STRAIGHT: "GO_STRAIGHT",
    RoadOption.RIGHT: "TURN_RIGHT",
    RoadOption.LEFT: "TURN_LEFT",
    RoadOption.LANEFOLLOW: "LANE_FOLLOW",
}

# Threshold to determine that the goal has been reached based on distance
DISTANCE_TO_GOAL_THRESHOLD = 0.5

# Threshold to determine that the goal has been reached based on orientation
ORIENTATION_TO_GOAL_THRESHOLD = math.pi / 4.0

# Number of retries if the server doesn't respond
RETRIES_ON_ERROR = 2

# Dummy Z coordinate to use when we only care about (x, y)
GROUND_Z = 22

# DISCRETE_ACTIONS = {
#     # middle forward
#     0: [0.5, 0.0],
#     # turn left
#     1: [0.4, -0.5],
#     # turn right
#     2: [0.4, 0.5],
#     # forward
#     3: [0.75, 0.0],
#     # slow forward
#     4: [0.25, 0.0],
#     # forward left
#     5: [0.45, -0.2],
#     # forward right
#     6: [0.45, 0.2],
#     # sharp left
#     7: [0.3, -0.75],
#     # sharp right
#     8: [0.3, 0.75],
# }
DISCRETE_ACTIONS = {
    # middle forward
    0: [0.7, 0.0],
    # turn left
    1: [0.6, -0.5],
    # turn right
    2: [0.6, 0.5],
    # forward
    3: [1, 0.0],
    # slow forward
    4: [0.45, 0.0],
    # forward left
    5: [0.65, -0.2],
    # forward right
    6: [0.65, 0.2],
    # sharp left
    7: [0.5, -0.75],
    # sharp right
    8: [0.5, 0.75],
}
WEATHERS = {
    0: carla.WeatherParameters.ClearNoon,
    1: carla.WeatherParameters.CloudyNoon,
    2: carla.WeatherParameters.WetNoon,
    3: carla.WeatherParameters.WetCloudyNoon,
    4: carla.WeatherParameters.MidRainyNoon,
    5: carla.WeatherParameters.HardRainNoon,
    6: carla.WeatherParameters.SoftRainNoon,
    7: carla.WeatherParameters.ClearSunset,
    8: carla.WeatherParameters.CloudySunset,
    9: carla.WeatherParameters.WetSunset,
    10: carla.WeatherParameters.WetCloudySunset,
    11: carla.WeatherParameters.MidRainSunset,
    12: carla.WeatherParameters.HardRainSunset,
    13: carla.WeatherParameters.SoftRainSunset,
}

live_carla_processes = set()


def cleanup():
    print("Killing live carla processes", live_carla_processes)
    for pgid in live_carla_processes:
        if IS_WINDOWS_PLATFORM:
            # for Windows
            subprocess.call(["taskkill", "/F", "/T", "/PID", str(pgid)])
        else:
            # for Linux
            os.killpg(pgid, signal.SIGKILL)

    live_carla_processes.clear()


def termination_cleanup(*_):
    cleanup()
    sys.exit(0)


signal.signal(signal.SIGTERM, termination_cleanup)
signal.signal(signal.SIGINT, termination_cleanup)
atexit.register(cleanup)

MultiAgentEnvBases = [MultiActorEnv]
try:
    from ray.rllib.env import MultiAgentEnv

    MultiAgentEnvBases.append(MultiAgentEnv)
except ImportError:
    logger.warning("\n Disabling RLlib support.", exc_info=True)


class MultiCarlaEnv(*MultiAgentEnvBases):
    def __init__(self, configs=None):
        """MACAD-Gym environment implementation.

        Provides a generic MACAD-Gym environment implementation that can be
        customized further to create new or variations of existing
        multi-agent learning environments. The environment settings, scenarios
        and the actors in the environment can all be configured using
        the `configs` dict.

        Args:
            configs (dict): Configuration for environment specified under the
                `env` key and configurations for each actor specified as dict
                under `actor`.
                Example:
                    # >>> configs = {"env":{
                    "server_map":"/Game/Carla/Maps/Town05",
                    "discrete_actions":True,...},
                    "actor":{
                    "actor_id1":{"enable_planner":True,...},
                    "actor_id2":{"enable_planner":False,...}
                    }}
        """
        if configs is None:
            configs = DEFAULT_MULTIENV_CONFIG
        self._env_config = configs["env"]
        if self._env_config.get("using_imitation_model", False):
            config = tf.ConfigProto()
            config.gpu_options.per_process_gpu_memory_fraction = 0.2
            self._tf_session = tf.Session(config=config)
            self._image_agent = ImitationLearning(self._tf_session)
            start_time = datetime.now()
            self._image_agent.load_model()
            end_time = datetime.now()
            print("time:", (end_time - start_time).total_seconds())
            print("load successfully")
            data = array = np.random.random((88, 200, 3)) * 255
            start_time = datetime.now()
            feature_vec = self._image_agent.compute_feature(data)
            feature_vec = np.array(feature_vec)
            end_time = datetime.now()
            print("time:", (end_time-start_time).total_seconds())
            print("feature:", feature_vec.shape)


        # Functionalities classes
        self._reward_policy = Reward()
        configs["scenarios"] = Scenarios.resolve_scenarios_parameter(
            configs["scenarios"]
        )

        self._scenario_config = configs["scenarios"]
        self._actor_configs = configs["actors"]

        # At most one actor can be manual controlled
        manual_control_count = 0
        for _, actor_config in self._actor_configs.items():
            if actor_config["manual_control"]:
                if "vehicle" not in actor_config["type"]:
                    raise ValueError("Only vehicles can be manual controlled.")

                manual_control_count += 1

        assert manual_control_count <= 1, (
            "At most one actor can be manually controlled. "
            f"Found {manual_control_count} actors with manual_control=True"
        )

        # Camera position is problematic for certain vehicles and even in
        # autopilot they are prone to error
        self.exclude_hard_vehicles = True
        # list of str: Supported values for `type` filed in `actor_configs`
        # for actors than can be actively controlled
        self._supported_active_actor_types = [
            "vehicle_4W",
            "vehicle_2W",
            "pedestrian",
            "traffic_light",
        ]
        # list of str: Supported values for `type` field in `actor_configs`
        # for actors that are passive. Example: A camera mounted on a pole
        self._supported_passive_actor_types = ["camera"]

        # Set attributes as in gym's specs
        self.reward_range = (-float("inf"), float("inf"))
        self.metadata = {"render.modes": "human"}

        # Belongs to env_config.
        self._server_map = self._env_config["server_map"]
        self._map = self._server_map.split("/")[-1]
        self._render = self._env_config["render"]
        self._framestack = self._env_config["framestack"]
        self._discrete_actions = self._env_config["discrete_actions"]
        self._squash_action_logits = self._env_config["squash_action_logits"]
        self._verbose = self._env_config["verbose"]
        self._render_x_res = self._env_config["render_x_res"]
        self._render_y_res = self._env_config["render_y_res"]
        self._x_res = self._env_config["x_res"]
        self._y_res = self._env_config["y_res"]
        self._use_depth_camera = self._env_config["use_depth_camera"]
        # self._use_semantic_segmentation_camera = self._env_config["use_semantic_segmentation_camera"]
        self._sync_server = self._env_config["sync_server"]
        self._fixed_delta_seconds = self._env_config["fixed_delta_seconds"]
        self._episode = 0
        # Initialize to be compatible with cam_manager to set HUD.
        pygame.font.init()  # for HUD
        self._hud = HUD(self._render_x_res, self._render_y_res)

        # For manual_control
        self._control_clock = None
        self._manual_controller = None
        self._manual_control_camera_manager = None

        # Render related
        Render.resize_screen(self._render_x_res, self._render_y_res)

        self._camera_poses, window_dim = Render.get_surface_poses(
            [self._x_res, self._y_res], self._actor_configs
        )

        if manual_control_count == 0:
            Render.resize_screen(window_dim[0], window_dim[1])
        else:
            self._manual_control_render_pose = (0, window_dim[1])
            Render.resize_screen(
                max(self._render_x_res, window_dim[0]),
                self._render_y_res + window_dim[1],
            )

        # Actions space
        if self._discrete_actions:
            self.action_space = Dict(
                {
                    actor_id: Discrete(len(DISCRETE_ACTIONS))
                    for actor_id in self._actor_configs.keys()
                }
            )
        else:
            self.action_space = Dict(
                {
                    actor_id: Box(-1.0, 1.0, shape=(2,))
                    for actor_id in self._actor_configs.keys()
                }
            )

        # Output space of images after preprocessing
        if self._use_depth_camera:
            self._image_space = Box(
                0.0, 255.0, shape=(self._y_res, self._x_res, 1 * self._framestack)
            )
        else:
            self._image_space = Box(
                0.0, 1.0, shape=(self._y_res, self._x_res, 3 * self._framestack)
            )
        self._feature_space = Box(0.0,1.0,shape=(512,))
        # Observation space in output
        if self._env_config["send_measurements"]:
            if self._env_config["using_imitation_model"]:
                self.observation_space = Dict(
                    {
                        actor_id: Tuple(
                            [
                                self._feature_space,  # feature
                                # Discrete(len(COMMANDS_ENUM)),  # next_command
                                Box(
                                    -20.0, 20.0, shape=(11,)
                                ),  # directionx/y, rel_speed_x/y, rel_loc_x/y, next-command-onehot encode(0~4)
                            ]
                        )
                        for actor_id in self._actor_configs.keys()
                    }
                )
            else:
                self.observation_space = Dict(
                    {
                        actor_id: Tuple(
                            [
                                self._image_space,  # image
                                # Discrete(len(COMMANDS_ENUM)),  # next_command
                                Box(
                                    -20.0, 20.0, shape=(11,)
                                ),  # directionx/y, rel_speed_x/y, rel_loc_x/y, next-command-onehot encode(0~4)
                            ]
                        )
                        for actor_id in self._actor_configs.keys()
                    }
                )
        else:
            self.observation_space = Dict(
                {actor_id: self._image_space for actor_id in self._actor_configs.keys()}
            )

        # Set appropriate node-id to coordinate mappings for Town01 or Town02.
        self.pos_coor_map = MAP_TO_COORDS_MAPPING[self._map]

        self._spec = lambda: None
        self._spec.id = "Carla-v0"
        self._server_port = None
        self._server_process = None
        self._client = None
        self._num_steps = {}
        self._total_reward = {}
        self._target_waypoint = {}
        self._prev_measurement = {}
        self._prev_image = {}
        self._episode_id_dict = {}
        self._measurements_file_dict = {}
        self._weather = None
        self._start_pos = {}  # Start pose for each actor
        self._end_pos = {}  # End pose for each actor
        self._start_coord = {}
        self._end_coord = {}
        self._last_obs = None
        self._image = None
        self._surface = None
        self._video = False
        self._obs_dict = {}
        self._done_dict = {}
        self._previous_actions = {}
        self._previous_rewards = {}
        self._last_reward = {}
        self._npc_vehicles = []  # List of NPC vehicles
        self._npc_pedestrians = []  # List of NPC pedestrians
        self._agents = {}  # Dictionary of macad_agents with agent_id as key
        self._actors = {}  # Dictionary of actors with actor_id as key
        self._actor_id_in_carla = {} # Dictionary of actor_id, with actor_id_in_carla as key, for a given actor_id_in_carla, we can get the actor_id 
        self._cameras = {}  # Dictionary of sensors with actor_id as key
        self._path_trackers = {}  # Dictionary of sensors with actor_id as key
        self._collisions = {}  # Dictionary of sensors with actor_id as key
        self._lane_invasions = {}  # Dictionary of sensors with actor_id as key
        self._scenario_map = {}  # Dictionary with current scenario map config
        self._behavior_agents = {}
        self._time_to_firstcollision={}
        self._time_to_goal={}
        self._episode_reward=defaultdict(list)

    @staticmethod
    def _get_tcp_port(port=0):
        """
        Get a free tcp port number
        :param port: (default 0) port number. When `0` it will be assigned a free port dynamically
        :return: a port number requested if free otherwise an unhandled exception would be thrown
        """
        s = socket.socket()
        s.bind(("", port))
        server_port = s.getsockname()[1]
        s.close()
        return server_port

    def _init_server(self):
        """Initialize carla server and client

        Returns:
            N/A
        """
        print("Initializing new Carla server...")
        # Create a new server process and start the client.
        # First find a port that is free and then use it in order to avoid
        # crashes due to:"...bind:Address already in use"
        self._server_port = self._get_tcp_port()

        multigpu_success = False
        gpus = GPUtil.getGPUs()
        log_file = os.path.join(LOG_DIR, "server_" + str(self._server_port) + ".log")
        logger.info(
            f"1. Port: {self._server_port}\n"
            f"2. Map: {self._server_map}\n"
            f"3. Binary: {SERVER_BINARY}"
        )
        if multigpu_success is False:
            try:
                print("Using single gpu to initialize carla server")

                self._server_process = subprocess.Popen(
                    [
                        SERVER_BINARY,
                        "-windowed ",
                        "-ResX=",
                        str(self._env_config["render_x_res"]),
                        "-ResY=",
                        str(self._env_config["render_y_res"]),
                        # "-prefernvidia "
                        # "-quality-level=Low",
                        # "-RenderOffScreen",
                        "-benchmark -fps=10",
                        "-carla-server",
                        "-carla-rpc-port={}".format(self._server_port),
                        "-carla-streaming-port=0",
                    ],
                    # for Linux
                    preexec_fn=None if IS_WINDOWS_PLATFORM else os.setsid,
                    # for Windows (not necessary)
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
                    if IS_WINDOWS_PLATFORM
                    else 0,
                    stdout=open(log_file, "w"),
                )
                print("Running simulation in single-GPU mode")
            except Exception as e:
                logger.debug(e)
                print("FATAL ERROR while launching server:", sys.exc_info()[0])

        if IS_WINDOWS_PLATFORM:
            live_carla_processes.add(self._server_process.pid)
        else:
            live_carla_processes.add(os.getpgid(self._server_process.pid))

        # Start client
        self._client = None
        while self._client is None:
            try:
                print("connecting to carla ... server port:",self._server_port)
                self._client = carla.Client("localhost", self._server_port)
                # The socket establishment could takes some time
                time.sleep(2)
                self._client.set_timeout(20.0)
                print(
                    "Client successfully connected to server, Carla-Server version: ",
                    self._client.get_server_version(),
                )
            except RuntimeError as re:
                if "timeout" not in str(re) and "time-out" not in str(re):
                    print("Could not connect to Carla server because:", re)
                self._client = None

        self._client.set_timeout(60.0)
        # load map using client api since 0.9.6+
        self._client.load_world(self._server_map)
        self.world = self._client.get_world()
        world_settings = self.world.get_settings()
        world_settings.synchronous_mode = self._sync_server
        if self._sync_server:
            # Synchronous mode
            # try:
            # Available with CARLA version>=0.9.6
            # Set fixed_delta_seconds to have reliable physics between sim steps
            world_settings.fixed_delta_seconds = self._fixed_delta_seconds
        self.world.apply_settings(world_settings)
        # Set up traffic manager
        self._traffic_manager_port = self._get_tcp_port()
        self._traffic_manager = self._client.get_trafficmanager(self._traffic_manager_port)
        self._traffic_manager.set_global_distance_to_leading_vehicle(2.5)
        self._traffic_manager.set_respawn_dormant_vehicles(True)
        self._traffic_manager.set_synchronous_mode(self._sync_server)
        # Set the spectator/server view if rendering is enabled
        if self._render and self._env_config.get("spectator_loc"):
            spectator = self.world.get_spectator()
            spectator_loc = carla.Location(*self._env_config["spectator_loc"])
            d = 6.4
            angle = 160  # degrees
            a = math.radians(angle)
            location = (
                carla.Location(d * math.cos(a), d * math.sin(a), 2.0) + spectator_loc
            )
            spectator.set_transform(
                # carla.Transform(location, carla.Rotation(yaw=180 + angle, pitch=-15))
                carla.Transform(location, carla.Rotation(pitch=-90))
            )

        if self._env_config.get("enable_planner"):
            planner_dao = GlobalRoutePlannerDAO(self.world.get_map())
            self.planner = GlobalRoutePlannerv1(planner_dao)
            self.planner.setup()

    def _clean_world(self):
        """Destroy all actors cleanly before exiting

        Returns:
            N/A
        """
        for colli in self._collisions.values():
            if colli.sensor.is_alive:
                colli.sensor.destroy()
        for lane in self._lane_invasions.values():
            if lane.sensor.is_alive:
                lane.sensor.destroy()
        for actor in self._actors.values():
            if actor.is_alive:
                actor.destroy()
        for npc in self._npc_vehicles:
            npc.destroy()
        for npc in zip(*self._npc_pedestrians):
            npc[1].stop()  # stop controller
            npc[0].destroy()  # kill entity
        # Note: the destroy process for cameras is handled in camera_manager.py

        self._cameras = {}
        self._actors = {}
        self._actor_id_in_carla = {}
        self._npc_vehicles = []
        self._npc_pedestrians = []
        self._path_trackers = {}
        self._collisions = {}
        self._lane_invasions = {}
        self._behavior_agents = {}
        self._time_to_firstcollision={}
        self._time_to_goal={}
        print("Cleaned-up the world...")

    def _clear_server_state(self):
        """Clear server process"""
        print("Clearing Carla server state")
        try:
            if self._client:
                self._client = None
        except Exception as e:
            print("Error disconnecting client: {}".format(e))
        if self._server_process:
            if IS_WINDOWS_PLATFORM:
                subprocess.call(
                    ["taskkill", "/F", "/T", "/PID", str(self._server_process.pid)]
                )
                live_carla_processes.remove(self._server_process.pid)
            else:
                pgid = os.getpgid(self._server_process.pid)
                os.killpg(pgid, signal.SIGKILL)
                live_carla_processes.remove(pgid)

            self._server_port = None
            self._server_process = None

    def reset(self):
        """Reset the carla world, call _init_server()

        Returns:
            N/A
        """
        # World reset and new scenario selection if multiple are available
        self._episode += 1
        self._load_scenario(self._scenario_config)

        for retry in range(RETRIES_ON_ERROR):
            try:
                if not self._server_process:
                    self._init_server()
                    self._reset(clean_world=False)
                else:
                    self._reset()
                break
            except Exception as e:
                print("Error during reset: {}".format(traceback.format_exc()))
                print("reset(): Retry #: {}/{}".format(retry + 1, RETRIES_ON_ERROR))
                self._clear_server_state()
                raise e

        # Set appropriate initial values for all actors
        for actor_id, actor_config in self._actor_configs.items():
            if self._done_dict.get(actor_id, True):
                self._target_waypoint[actor_id] = None
                self._last_reward[actor_id] = None
                self._total_reward[actor_id] = None
                self._num_steps[actor_id] = 0
                # self._behavior_agents[actor_id]=None
                self._time_to_firstcollision[actor_id]=None
                self._time_to_goal[actor_id]=None
                py_measurement = self._read_observation(actor_id)
                self._prev_measurement[actor_id] = py_measurement
                cam = self._cameras[actor_id]
                # Wait for the sensor (camera) actor to start streaming
                # Shouldn't take too long
                while cam.callback_count == 0:
                    if self._sync_server:
                        self.world.tick()
                        # `wait_for_tick` is no longer needed, see https://github.com/carla-simulator/carla/pull/1803
                        # self.world.wait_for_tick()
                if cam.image is None:
                    print("callback_count:", actor_id, ":", cam.callback_count)
                obs = self._encode_obs(actor_id, cam.image, py_measurement)
                self._obs_dict[actor_id] = obs
                # Actor correctly reset
                self._done_dict[actor_id] = False

        return self._obs_dict

    # ! Deprecated method
    def _on_render(self):
        """Render the pygame window.

        Args:

        Returns:
            N/A
        """
        pass

    def _spawn_new_actor(self, actor_id):
        """Spawn an agent as per the blueprint at the given pose

        Args:
            blueprint: Blueprint of the actor. Can be a Vehicle or Pedestrian
            pose: carla.Transform object with location and rotation

        Returns:
            An instance of a subclass of carla.Actor. carla.Vehicle in the case
            of a Vehicle agent.

        """
        actor_type = self._actor_configs[actor_id].get("type", "vehicle_4W")
        if actor_type not in self._supported_active_actor_types:
            print("Unsupported actor type:{}. Using vehicle_4W as the type")
            actor_type = "vehicle_4W"

        if actor_type == "traffic_light":
            # Traffic lights already exist in the world & can't be spawned.
            # Find closest traffic light actor in world.actor_list and return
            from macad_gym.core.controllers import traffic_lights

            loc = carla.Location(
                self._start_pos[actor_id][0],
                self._start_pos[actor_id][1],
                self._start_pos[actor_id][2],
            )
            rot = (
                self.world.get_map()
                .get_waypoint(loc, project_to_road=True)
                .transform.rotation
            )
            #: If yaw is provided in addition to (X, Y, Z), set yaw
            if len(self._start_pos[actor_id]) > 3:
                rot.yaw = self._start_pos[actor_id][3]
            transform = carla.Transform(loc, rot)
            self._actor_configs[actor_id]["start_transform"] = transform
            tls = traffic_lights.get_tls(self.world, transform, sort=True)
            return tls[0][0]  #: Return the key (carla.TrafficLight object) of
            #: closest match

        if actor_type == "pedestrian":
            blueprints = self.world.get_blueprint_library().filter(
                "walker.pedestrian.*"
            )

        elif actor_type == "vehicle_4W":
            blueprints = self.world.get_blueprint_library().filter("vehicle")
            # Further filter down to 4-wheeled vehicles
            blueprints = [
                b for b in blueprints if int(b.get_attribute("number_of_wheels")) == 4
            ]
            if self.exclude_hard_vehicles:
                blueprints = list(
                    filter(
                        lambda x: not (
                            x.id.endswith("microlino")
                            or x.id.endswith("carlacola")
                            or x.id.endswith("cybertruck")
                            or x.id.endswith("t2")
                            or x.id.endswith("sprinter")
                            or x.id.endswith("firetruck")
                            or x.id.endswith("ambulance")
                            or x.id.endswith("patrol_2021")
                            or x.id.endswith("patrol")
                            or x.id.endswith("t2")
                        ),
                        blueprints,
                    )
                )
        elif actor_type == "vehicle_2W":
            # blueprints = self.world.get_blueprint_library().filter("vehicle")
            # # Further filter down to 2-wheeled vehicles
            # blueprints = [
            #     b for b in blueprints if int(b.get_attribute("number_of_wheels")) == 2
            # ]
            blueprints = [self.world.get_blueprint_library().find("vehicle.kawasaki.ninja"),
                          self.world.get_blueprint_library().find("vehicle.harley-davidson.low_rider"),
                          self.world.get_blueprint_library().find("vehicle.yamaha.yzf")]

        blueprint = random.choice(blueprints)
        loc = carla.Location(
            x=self._start_pos[actor_id][0],
            y=self._start_pos[actor_id][1],
            z=self._start_pos[actor_id][2],
        )
        rot = (
            self.world.get_map()
            .get_waypoint(loc, project_to_road=True)
            .transform.rotation
        )
        #: If yaw is provided in addition to (X, Y, Z), set yaw
        if len(self._start_pos[actor_id]) > 3:
            rot.yaw = self._start_pos[actor_id][3]
        transform = carla.Transform(loc, rot)
        self._actor_configs[actor_id]["start_transform"] = transform
        vehicle = None
        if actor_id == "ego":
            blueprint = self.world.get_blueprint_library().find("vehicle.lincoln.mkz_2017")
            blueprint.set_attribute('role_name', 'ego_vehicle')
        print(actor_id," blueprint:",blueprint)
        for retry in range(RETRIES_ON_ERROR):
            vehicle = self.world.try_spawn_actor(blueprint, transform)
            if self._sync_server:
                self.world.tick()
            if vehicle is not None and vehicle.get_location().z > 0.0:
                # Register it under traffic manager
                # Walker vehicle type does not have autopilot. Use walker controller ai
                if actor_type == "pedestrian":
                    # vehicle.set_simulate_physics(False)
                    pass
                else:
                    vehicle.set_autopilot(False, self._traffic_manager.get_port())
                break
            # Wait to see if spawn area gets cleared before retrying
            # time.sleep(0.5)
            # self._clean_world()
            print("spawn_actor: Retry#:{}/{}".format(retry + 1, RETRIES_ON_ERROR))
        if vehicle is None:
            # Request a spawn one last time possibly raising the error
            vehicle = self.world.spawn_actor(blueprint, transform)
            
        return vehicle

    def _reset(self, clean_world=True):
        """Reset the state of the actors.
        A "soft" reset is performed in which the existing actors are destroyed
        and the necessary actors are spawned into the environment without
        affecting other aspects of the environment.
        If the "soft" reset fails, a "hard" reset is performed in which
        the environment's entire state is destroyed and a fresh instance of
        the server is created from scratch. Note that the "hard" reset is
        expected to take more time. In both of the reset modes ,the state/
        pose and configuration of all actors (including the sensor actors) are
        (re)initialized as per the actor configuration.

        Returns:
            dict: Dictionaries of observations for actors.

        Raises:
            RuntimeError: If spawning an actor at its initial state as per its'
            configuration fails (eg.: Due to collision with an existing object
            on that spot). This Error will be handled by the caller
            `self.reset()` which will perform a "hard" reset by creating
            a new server instance
        """

        self._done_dict["__all__"] = False
        self._lateral_distance = defaultdict(list)
        if clean_world:
            self._clean_world()

        weather_num = 0
        if "weather_distribution" in self._scenario_map:
            weather_num = random.choice(self._scenario_map["weather_distribution"])
            if weather_num not in WEATHERS:
                weather_num = 0

        self.world.set_weather(WEATHERS[weather_num])

        self._weather = [
            self.world.get_weather().cloudiness,
            self.world.get_weather().precipitation,
            self.world.get_weather().precipitation_deposits,
            self.world.get_weather().wind_intensity,
        ]

        for actor_id, actor_config in self._actor_configs.items():
            # if self._done_dict.get(actor_id, True):
            if True:
                self._measurements_file_dict[actor_id] = None
                self._episode_id_dict[actor_id] = datetime.today().strftime(
                    "%Y-%m-%d_%H-%M-%S_%f"
                )
                actor_config = self._actor_configs[actor_id]

                # Try to spawn actor (soft reset) or fail and reinitialize the server before get back here
                try:
                    self._actors[actor_id] = self._spawn_new_actor(actor_id)
                    actor_id_in_carla = self._actors[actor_id].id 
                    self._actor_id_in_carla[actor_id_in_carla] = actor_id
                except RuntimeError as spawn_err:
                    del self._done_dict[actor_id]
                    # Chain the exception & re-raise to be handled by the caller `self.reset()`
                    raise spawn_err from RuntimeError(
                        "Unable to spawn actor:{}".format(actor_id)
                    )

                if self._env_config["enable_planner"]:
                    self._path_trackers[actor_id] = PathTracker(
                        self.world,
                        self.planner,
                        (
                            self._start_pos[actor_id][0],
                            self._start_pos[actor_id][1],
                            self._start_pos[actor_id][2],
                        ),
                        (
                            self._end_pos[actor_id][0],
                            self._end_pos[actor_id][1],
                            self._end_pos[actor_id][2],
                        ),
                        self._actors[actor_id],
                    )

                    # print("pathtracker ",actor_id)
                # if actor_config["auto_control"]:
                self._behavior_agents[actor_id] = BehaviorAgent(self._actors[actor_id], behavior="cautious")
                endloc = carla.Location(self._end_pos[actor_id][0], self._end_pos[actor_id][1], self._end_pos[actor_id][2])
                self._behavior_agents[actor_id].set_destination(endloc)

                # Spawn collision and lane sensors if necessary
                if actor_config["collision_sensor"] == "on":
                    collision_sensor = CollisionSensor(self._actors[actor_id], hud=0, ego=self._actors["ego"])
                    self._collisions.update({actor_id: collision_sensor})

                if actor_config["lane_sensor"] == "on":
                    lane_sensor = LaneInvasionSensor(self._actors[actor_id], 0)
                    self._lane_invasions.update({actor_id: lane_sensor})

                # Spawn cameras
                pygame.font.init()  # for HUD
                hud = HUD(self._env_config["x_res"], self._env_config["y_res"])
                camera_manager = CameraManager(self._actors[actor_id], hud)
                if actor_config["log_images"]:
                    # TODO: The recording option should be part of config
                    # 1: Save to disk during runtime
                    # 2: save to memory first, dump to disk on exit
                    camera_manager.set_recording_option(1)

                # in CameraManger's._sensors
                camera_type = self._actor_configs[actor_id]["camera_type"]
                camera_pos = getattr(
                    self._actor_configs[actor_id], "camera_position", 0
                )
                camera_types = [ct.name for ct in CAMERA_TYPES]
                assert (
                    camera_type in camera_types
                ), "Camera type `{}` not available. Choose in {}.".format(
                    camera_type, camera_types
                )
                camera_manager.set_sensor(
                    CAMERA_TYPES[camera_type].value - 1, int(camera_pos), notify=False
                )
                assert camera_manager.sensor.is_listening
                self._cameras.update({actor_id: camera_manager})

                # Manual Control
                if actor_config["manual_control"]:
                    self._control_clock = pygame.time.Clock()

                    self._manual_controller = KeyboardControl(
                        self, actor_config["auto_control"]
                    )
                    self._manual_controller.actor_id = actor_id

                    self.world.on_tick(self._hud.on_world_tick)
                    self._manual_control_camera_manager = CameraManager(
                        self._actors[actor_id], self._hud
                    )
                    self._manual_control_camera_manager.set_sensor(
                        CAMERA_TYPES["rgb"].value - 1, pos=2, notify=False
                    )

                self._start_coord.update(
                    {
                        actor_id: [
                            self._start_pos[actor_id][0] // 100,
                            self._start_pos[actor_id][1] // 100,
                        ]
                    }
                )
                self._end_coord.update(
                    {
                        actor_id: [
                            self._end_pos[actor_id][0] // 100,
                            self._end_pos[actor_id][1] // 100,
                        ]
                    }
                )

                print(
                    "Actor: {} start_pos_xyz(coordID): {} ({}), "
                    "end_pos_xyz(coordID) {} ({})".format(
                        actor_id,
                        self._start_pos[actor_id],
                        self._start_coord[actor_id],
                        self._end_pos[actor_id],
                        self._end_coord[actor_id],
                    )
                )
        time.sleep(2)

        print("New episode initialized with actors:{}".format(self._actors.keys()))

        #set red trafficlight when egovehicle is spawned
        if self._actors["ego"].is_at_traffic_light():
            traffic_light = self._actors["ego"].get_traffic_light()
            # if traffic_light.get_state() == carla.TrafficLightState.Green:
            #     traffic_light.set_state(carla.TrafficLightState.Red)
            # traffic_light.set_red_time(1)
            if traffic_light.get_state() == carla.TrafficLightState.Red:
                traffic_light.set_state(carla.TrafficLightState.Green)


        self._npc_vehicles, self._npc_pedestrians = apply_traffic(
            self.world,
            self._traffic_manager,
            self._scenario_map.get("num_vehicles", 0),
            self._scenario_map.get("num_pedestrians", 0),
        )


    def _load_scenario(self, scenario_parameter):
        self._scenario_map = {}
        # If config contains a single scenario, then use it,
        # if it's an array of scenarios,randomly choose one and init
        if isinstance(scenario_parameter, dict):
            scenario = scenario_parameter
        else:  # instance array of dict
            scenario = random.choice(scenario_parameter)

        # if map_name not in (town, "OpenDriveMap"):  TODO
        #     print("The CARLA server uses the wrong map: {}".format(map_name))
        #     print("This scenario requires to use map: {}".format(town))
        #     return False

        self._scenario_map = scenario
        for actor_id, actor in scenario["actors"].items():
            if isinstance(actor["start"], int):
                self._start_pos[actor_id] = self.pos_coor_map[str(actor["start"])]
            else:
                self._start_pos[actor_id] = actor["start"]

            if isinstance(actor["end"], int):
                self._end_pos[actor_id] = self.pos_coor_map[str(actor["end"])]
            else:
                self._end_pos[actor_id] = actor["end"]

    def _decode_obs(self, actor_id, obs):
        """Decode actor observation into original image reversing the pre_process() operation.
        Args:
            actor_id (str): Actor identifier
            obs (dict): Properly encoded observation data of an actor

        Returns:
            image (array): Original actor camera view
        """
        if self._env_config["using_imitation_model"]:
            # return self._prev_image[actor_id]
            obs = self._prev_image[actor_id]
            obs = np.multiply(obs.astype(np.float32), 255.0)
            # print(obs.shape)
            return obs.swapaxes(0, 1)
        if self._actor_configs[actor_id]["send_measurements"]:
            obs = obs[0]
        # Reverse the processing operation
        if self._actor_configs[actor_id]["use_depth_camera"]:
            img = np.tile(obs.swapaxes(0, 1), 3)
        else:
            # print("obs shape",obs.shape)
            img = obs.swapaxes(0, 1)
            # print("img render shape",img.shape)
        img = np.multiply(img.astype(np.float32), 255.0)
        return img

    def _encode_obs(self, actor_id, image, py_measurements):
        """Encode sensor and measurements into obs based on state-space config.

        Args:
            actor_id (str): Actor identifier
            image (array): original unprocessed image
            py_measurements (dict): measurement file

        Returns:
            obs (dict): properly encoded observation data for each actor
        """
        assert self._framestack in [1, 2]
        # Apply preprocessing
        config = self._actor_configs[actor_id]
        image = preprocess_image(image, config)
        # Stack frames
        prev_image = self._prev_image.get(actor_id,None)
        self._prev_image[actor_id] = image
        if prev_image is None:
            prev_image = image
        if self._framestack == 2:
            # image = np.concatenate([prev_image, image], axis=2)
            image = np.concatenate([prev_image, image],axis=2)
        # Structure the observation
        if not self._actor_configs[actor_id]["send_measurements"]:
            return image
        ego_location = self._actors["ego"].get_location()
        location = self._actors[actor_id].get_location()
        goal = self._end_pos[actor_id]
        ego_speed = self._actors["ego"].get_velocity()
        speed = self._actors[actor_id].get_velocity()

        rel_loc = np.array([ego_location.x-location.x, ego_location.y-location.y])
        rel_speed = np.array([ego_speed.x-speed.x, ego_speed.y-speed.y])
        rel_loc = np.clip(rel_loc/5, -20, 20)
        rel_speed = np.clip(rel_speed,-20, 20)

        speed = np.array([speed.x, speed.y])
        norm = np.linalg.norm(speed)
        if (norm != 0):
            speed = speed / norm
        # command_index = COMMAND_ORDINAL[py_measurements["next_command"]]
        # command_onehot = np.zeros(5)
        # command_onehot[command_index]=1
        rel_to_goal = np.array([goal[0]-location.x, goal[1]-location.y])
        norm = np.linalg.norm(rel_to_goal)
        if(norm!=0):
            rel_to_goal /= norm
        command_onehot = np.zeros(5)
        command_onehot[0]=rel_to_goal[0]
        command_onehot[1]=rel_to_goal[1]
        if py_measurements.get("autocontrol",None) is not None:
            control = py_measurements["autocontrol"]
            command_onehot[2] = control["steer"]
            command_onehot[3] = control["throttle"]
            command_onehot[4] = control["brake"]
        # norm = np.linalg.norm(rel_speed)
        #
        # if (norm != 0):
        #     rel_speed = rel_speed / norm
        #speed = math.sqrt(pow(self._actors[actor_id].get_velocity().x,2) + pow(self._actors[actor_id].get_velocity().y,2) + pow(self._actors[actor_id].get_velocity().z,2))
        # direction_to_ego = np.array([ego_location.x-location.x, ego_location.y-location.y])
        # norm = np.linalg.norm(direction_to_ego)
        # if(norm != 0):
        #     direction_to_ego = direction_to_ego/norm
        # distance_to_ego = calculate_distance(location.x,location.y,location.z, ego_location.x,ego_location.y,ego_location.z)
        if self._env_config["using_imitation_model"]:
            #image shape (88, 200, 3)
            image = cv2.resize(image, (88, 200), interpolation=cv2.INTER_AREA)
            feature = self._image_agent.compute_feature(image)
            feature = np.clip(feature,0,1)
            obs = (
                feature,
                # direction, relative speed and relative position, and command
                [speed[0], speed[1], rel_speed[0], rel_speed[1], rel_loc[0], rel_loc[1], command_onehot[0],
                 command_onehot[1], command_onehot[2], command_onehot[3], command_onehot[4]],
            )
        else:
            obs = (
                image,
                #direction, relative speed and relative position, and command
                [speed[0],speed[1],rel_speed[0],rel_speed[1],rel_loc[0],rel_loc[1],command_onehot[0],command_onehot[1],command_onehot[2],command_onehot[3],command_onehot[4]],
            )

        self._last_obs = obs
        return obs

    def step(self, action_dict):
        """Execute one environment step for the specified actors.

        Executes the provided action for the corresponding actors in the
        environment and returns the resulting environment observation, reward,
        done and info (measurements) for each of the actors. The step is
        performed asynchronously i.e. only for the specified actors and not
        necessarily for all actors in the environment.

        Args:
            action_dict (dict): Actions to be executed for each actor. Keys are
                agent_id strings, values are corresponding actions.

        Returns
            obs (dict): Observations for each actor.
            rewards (dict): Reward values for each actor. None for first step
            dones (dict): Done values for each actor. Special key "__all__" is
            set when all actors are done and the env terminates
            info (dict): Info for each actor.

        Raises
            RuntimeError: If `step(...)` is called before calling `reset()`
            ValueError: If `action_dict` is not a dictionary of actions
            ValueError: If `action_dict` contains actions for nonexistent actor
        """
        # print("step")
        if (not self._server_process) or (not self._client):
            raise RuntimeError("Cannot call step(...) before calling reset()")

        assert len(self._actors), (
            "No actors exist in the environment. Either"
            " the environment was not properly "
            "initialized using`reset()` or all the "
            "actors have exited. Cannot execute `step()`"
        )

        if not isinstance(action_dict, dict):
            raise ValueError(
                "`step(action_dict)` expected dict of actions. "
                "Got {}".format(type(action_dict))
            )
        # Make sure the action_dict contains actions only for actors that
        # exist in the environment
        if not set(action_dict).issubset(set(self._actors)):
            raise ValueError(
                "Cannot execute actions for non-existent actors."
                " Received unexpected actor ids:{}".format(
                    set(action_dict).difference(set(self._actors))
                )
            )

        try:
            obs_dict = {}
            reward_dict = {}
            info_dict = {}

            for actor_id, action in action_dict.items():
                obs, reward, done, info = self._step(actor_id, action)
                # time.sleep(1)
                obs_dict[actor_id] = obs
                reward_dict[actor_id] = reward
                if not self._done_dict.get(actor_id, False):
                    self._done_dict[actor_id] = done
                info_dict[actor_id] = info
            self._done_dict["__all__"] = sum(self._done_dict.values()) >= len(
                self._actors
            )
            # if(self._done_dict["__all__"]):
                # print(self._total_reward)
                # for actor_id, total_reward in self._total_reward.items():
                #     self._episode_reward[actor_id].append(total_reward)
                # print("episode_reward:",self._episode_reward)
            # Find if any actor's config has render=True & render only for
            # that actor. NOTE: with async server stepping, enabling rendering
            # affects the step time & therefore MAX_STEPS needs adjustments
            render_required = [
                k for k, v in self._actor_configs.items() if v.get("render", False)
            ]
            if render_required:
                images = {
                    k: self._decode_obs(k, v)
                    for k, v in obs_dict.items()
                    if self._actor_configs[k]["render"]
                }

                Render.multi_view_render(images, self._camera_poses)
                if self._manual_controller is None:
                    Render.dummy_event_handler()

            return obs_dict, reward_dict, self._done_dict, info_dict
        except Exception:
            print(
                "Error during step, terminating episode early.", traceback.format_exc()
            )
            self._clear_server_state()

    def _step(self, actor_id, action):
        """Perform the actual step in the CARLA environment

        Applies control to `actor_id` based on `action`, process measurements,
        compute the rewards and terminal state info (dones).

        Args:
            actor_id(str): Actor identifier
            action: Actions to be executed for the actor.

        Returns
            obs (obs_space): Observation for the actor whose id is actor_id.
            reward (float): Reward for actor. None for first step
            done (bool): Done value for actor.
            info (dict): Info for actor.
        """

        if self._discrete_actions:
            action = DISCRETE_ACTIONS[int(action)]
        assert len(action) == 2, "Invalid action {}".format(action)
        if self._squash_action_logits:
            forward = 2 * float(sigmoid(action[0]) - 0.5)
            throttle = float(np.clip(forward, 0, 1))
            brake = float(np.abs(np.clip(forward, -1, 0)))
            steer = 2 * float(sigmoid(action[1]) - 0.5)
        else:
            throttle = float(np.clip(action[0], 0, 1))
            brake = float(np.abs(np.clip(action[0], -1, 0)))
            steer = float(np.clip(action[1], -1, 1))
        reverse = False
        hand_brake = False
        if self._verbose:
            print(
                "steer", steer, "throttle", throttle, "brake", brake, "reverse", reverse
            )

        config = self._actor_configs[actor_id]

        #behavioragent
        if self._behavior_agents[actor_id] is None:
            self._behavior_agents[actor_id] = BehaviorAgent(self._actors[actor_id], behavior="cautious")
            endloc = carla.Location(self._end_pos[actor_id][0],
                                    self._end_pos[actor_id][1],
                                    self._end_pos[actor_id][2])
            self._behavior_agents[actor_id].set_destination(endloc)
        if self._behavior_agents[actor_id].done():
            if self._time_to_goal[actor_id] is None:
                self._time_to_goal[actor_id] = self._num_steps[actor_id]
            print("ego target reached, searching another endpoint")
            self._behavior_agents[actor_id].set_destination(
                random.choice(self.world.get_map().get_spawn_points()).location)
        control = self._behavior_agents[actor_id].run_step()
        control.manual_gear_shift = False

        #reality metric: calcute lateral distance to planned waypoint
        self._target_waypoint[actor_id] = self._behavior_agents[actor_id]._local_planner.target_waypoint
        draw_waypoints(self.world, [self._target_waypoint[actor_id]], 1.0);
        self._lateral_distance[actor_id].append(calculate_lateral_distance(self._target_waypoint[actor_id], self._actors[actor_id].get_location()))

        if config["manual_control"]:
            self._control_clock.tick(60)
            self._manual_control_camera_manager._hud.tick(
                self.world,
                self._actors[actor_id],
                self._collisions[actor_id],
                self._control_clock,
            )
            self._manual_controller.parse_events(self, self._control_clock)

            # TODO: consider move this to Render as well
            self._manual_control_camera_manager.render(
                Render.get_screen(), self._manual_control_render_pose
            )
            self._manual_control_camera_manager._hud.render(
                Render.get_screen(), self._manual_control_render_pose
            )
            pygame.display.flip()
        elif config["auto_control"]:
            # if getattr(self._actors[actor_id], "set_autopilot", 0):
            #     self._actors[actor_id].set_autopilot(
            #         True, self._traffic_manager.get_port()
            #     )

            self._actors[actor_id].apply_control(control)
        else:
            if config.get("frame_skip", False) == False:
                agent_type = config.get("type", "vehicle")
                if agent_type == "pedestrian":
                    rotation = self._actors[actor_id].get_transform().rotation
                    rotation.yaw += steer * 180.0
                    # rotation.yaw = 180
                    x_dir = math.cos(math.radians(rotation.yaw))
                    y_dir = math.sin(math.radians(rotation.yaw))
                    self._actors[actor_id].apply_control(
                        carla.WalkerControl(
                            speed=3.0 * throttle,
                            direction=carla.Vector3D(x_dir, y_dir, 0.0),
                        )
                    )
                elif "vehicle" in agent_type:
                    self._actors[actor_id].apply_control(
                        carla.VehicleControl(
                            throttle=throttle,
                            steer=steer,
                            brake=brake,
                            hand_brake=hand_brake,
                            reverse=reverse,
                        )
                    )
            else: #frame_skip mode, every 5 step intervals apply vehicle_control by trained_agent, otherwise use behavioragent control
                if self._num_steps[actor_id] % 10 < 5 :
                    #control command by trained policy
                    agent_type = config.get("type", "vehicle")
                    if agent_type == "pedestrian":
                        rotation = self._actors[actor_id].get_transform().rotation
                        rotation.yaw += steer * 180.0
                        # rotation.yaw = 180
                        x_dir = math.cos(math.radians(rotation.yaw))
                        y_dir = math.sin(math.radians(rotation.yaw))

                        self._actors[actor_id].apply_control(
                            carla.WalkerControl(
                                speed=3.0 * throttle,
                                direction=carla.Vector3D(x_dir, y_dir, 0.0),
                            )
                        )
                    elif "vehicle" in agent_type:
                        self._actors[actor_id].apply_control(
                            carla.VehicleControl(
                                throttle=throttle,
                                steer=steer,
                                brake=brake,
                                hand_brake=hand_brake,
                                reverse=reverse,
                            )
                        )
                #set autocontrol by behavioragent
                else:
                    self._actors[actor_id].apply_control(control)


        # Asynchronosly (one actor at a time; not all at once in a sync) apply
        # actor actions & perform a server tick after each actor's apply_action
        # if running with sync_server steps
        if self._sync_server:
            self.world.tick()

        # Process observations
        py_measurements = self._read_observation(actor_id)
        if self._verbose:
            print("Next command", py_measurements["next_command"])
        # Store previous action
        self._previous_actions[actor_id] = action
        if type(action) is np.ndarray:
            py_measurements["action"] = [float(a) for a in action]
        else:
            py_measurements["action"] = action

        py_measurements["autocontrol"] = {
            "steer": control.steer,
            "throttle": control.throttle,
            "brake": control.brake,
        }
        # Compute done
        done = (
            self._num_steps[actor_id] > self._scenario_map["max_steps"]
            or py_measurements["next_command"] == "REACH_GOAL"
            or (
                config["early_terminate_on_collision"]
                and collided_done(py_measurements)
            )
            or(
                self._env_config.get("terminate_on_collision_to_ego", False)
                and py_measurements["collision_to_ego"]>0
            )
        )
        py_measurements["done"] = done

        # Compute reward
        config = self._actor_configs[actor_id]
        flag = config["reward_function"]
        ego_curr_measurement = self._read_observation("ego")
        reward = self._reward_policy.compute_reward(
            self._prev_measurement[actor_id], 
            py_measurements,
            ego_prev_measurement=self._prev_measurement["ego"], 
            ego_curr_measurement=ego_curr_measurement, 
            flag=flag
        )

        self._previous_rewards[actor_id] = reward
        if self._total_reward[actor_id] is None:
            self._total_reward[actor_id] = reward
        else:
            self._total_reward[actor_id] += reward

        py_measurements["reward"] = reward
        py_measurements["total_reward"] = self._total_reward[actor_id]

        # End iteration updating parameters and logging
        self._prev_measurement[actor_id] = py_measurements
        self._num_steps[actor_id] += 1

        if config["log_measurements"] and CARLA_OUT_PATH:
            # Write out measurements to file
            if not self._measurements_file_dict[actor_id]:
                self._measurements_file_dict[actor_id] = open(
                    os.path.join(
                        CARLA_OUT_PATH,
                        "measurements_{}.json".format(self._episode_id_dict[actor_id]),
                    ),
                    "w",
                )
            self._measurements_file_dict[actor_id].write(json.dumps(py_measurements))
            self._measurements_file_dict[actor_id].write("\n")
            if done:
                self._measurements_file_dict[actor_id].close()
                self._measurements_file_dict[actor_id] = None
                # if self.config["convert_images_to_video"] and\
                #  (not self.video):
                #    self.images_to_video()
                #    self.video = Trueseg_city_space
        original_image = self._cameras[actor_id].image

        return (
            self._encode_obs(actor_id, original_image, py_measurements),
            reward,
            done,
            py_measurements,
        )

    def _read_observation(self, actor_id):
        """Read observation and return measurement.

        Args:
            actor_id (str): Actor identifier

        Returns:
            dict: measurement data.

        """
        cur = self._actors[actor_id]
        cur_config = self._actor_configs[actor_id]
        planner_enabled = cur_config["enable_planner"]
        planner_enabled = False
        next_command = "LANE_FOLLOW"
        if planner_enabled:
            dist = self._path_trackers[actor_id].get_distance_to_end()
            orientation_diff = self._path_trackers[
                actor_id
            ].get_orientation_difference_to_end_in_radians()
            commands = self.planner.plan_route(
                (cur.get_location().x, cur.get_location().y),
                (self._end_pos[actor_id][0],self._end_pos[actor_id][1]),
            )
            if len(commands) > 0:
                next_command = ROAD_OPTION_TO_COMMANDS_MAPPING.get(
                    commands[0], "LANE_FOLLOW"
                )
            elif (
                dist <= DISTANCE_TO_GOAL_THRESHOLD
                and orientation_diff <= ORIENTATION_TO_GOAL_THRESHOLD
            ):
                next_command = "REACH_GOAL"
            else:
                next_command = "LANE_FOLLOW"

            # DEBUG
            # self.path_trackers[actor_id].draw()
        else:
            next_command = "LANE_FOLLOW"

        collision_vehicles = self._collisions[actor_id].collision_vehicles
        collision_pedestrians = self._collisions[actor_id].collision_pedestrians
        collision_other = self._collisions[actor_id].collision_other
        intersection_otherlane = self._lane_invasions[actor_id].offlane
        intersection_offroad = self._lane_invasions[actor_id].offroad
        collision_to_ego = self._collisions[actor_id].collision_to_ego

        distance_to_goal = 0.0
        if planner_enabled:
            distance_to_goal = self._path_trackers[actor_id].get_distance_to_end()
        else:
            distance_to_goal = float(
                np.linalg.norm(
                    [
                        self._actors[actor_id].get_location().x
                        - self._end_pos[actor_id][0],
                        self._actors[actor_id].get_location().y
                        - self._end_pos[actor_id][1],
                    ]
                )
            )
        ego_location = self._actors["ego"].get_location()
        location = self._actors[actor_id].get_location()
        distance_to_ego = calculate_distance(location.x,location.y,location.z, ego_location.x,ego_location.y,ego_location.z)

        if (collision_vehicles+collision_other+collision_pedestrians+collision_to_ego) > 0 and (self._time_to_firstcollision[actor_id] is None):
            self._time_to_firstcollision[actor_id] = self._num_steps[actor_id]



        py_measurements = {
            "actor_id": actor_id,
            # "episode_id": self._episode_id_dict[actor_id],
            "episode": self._episode,
            "step": self._num_steps[actor_id],
            # "location": self._actors[actor_id].get_location(),
            # "rotation": self._actors[actor_id].get_transform().rotation,
            # "velocity": self._actors[actor_id].get_velocity(),
            "x": self._actors[actor_id].get_location().x,
            "y": self._actors[actor_id].get_location().y,
            "z": self._actors[actor_id].get_location().z,
            "pitch": self._actors[actor_id].get_transform().rotation.pitch,
            "yaw": self._actors[actor_id].get_transform().rotation.yaw,
            "roll": self._actors[actor_id].get_transform().rotation.roll,
            "forward_speed": math.sqrt(pow(self._actors[actor_id].get_velocity().x,2) + pow(self._actors[actor_id].get_velocity().y,2) + pow(self._actors[actor_id].get_velocity().z,2)),
            "distance_to_ego": distance_to_ego,
            "distance_to_goal": distance_to_goal,
            # "distance_to_goal_euclidean": distance_to_goal_euclidean,
            "collision_to_ego": collision_to_ego,
            "time_to_firstcollision": self._time_to_firstcollision,
            "time_to_goal":self._time_to_goal,
            "collision_vehicles": collision_vehicles,
            "collision_pedestrians": collision_pedestrians,
            "collision_other": collision_other,
            "intersection_offroad": intersection_offroad,
            "intersection_otherlane": intersection_otherlane,
            # "weather": self._weather,
            # "map": self._server_map,
            # "start_coord": self._start_coord[actor_id],
            # "end_coord": self._end_coord[actor_id],
            # "current_scenario": self._scenario_map,
            # "x_res": self._x_res,
            # "y_res": self._y_res,
            # "max_steps": self._scenario_map["max_steps"],
            "next_command": next_command,
            "previous_action": self._previous_actions.get(actor_id, None),
            "previous_reward": self._previous_rewards.get(actor_id, None),
        }

        return py_measurements

    def close(self):
        """Clean-up the world, clear server state & close the Env"""
        self._clean_world()
        self._clear_server_state()


def print_measurements(measurements):
    number_of_agents = len(measurements.non_player_agents)
    player_measurements = measurements.player_measurements
    message = "Vehicle at ({pos_x:.1f}, {pos_y:.1f}), "
    message += "{speed:.2f} km/h, "
    message += "Collision: {{vehicles={col_cars:.0f}, "
    message += "pedestrians={col_ped:.0f}, other={col_other:.0f}}}, "
    message += "{other_lane:.0f}% other lane, {offroad:.0f}% off-road, "
    message += "({agents_num:d} non-player macad_agents in the scene)"
    message = message.format(
        pos_x=player_measurements.transform.location.x,
        pos_y=player_measurements.transform.location.y,
        speed=player_measurements.forward_speed,
        col_cars=player_measurements.collision_vehicles,
        col_ped=player_measurements.collision_pedestrians,
        col_other=player_measurements.collision_other,
        other_lane=100 * player_measurements.intersection_otherlane,
        offroad=100 * player_measurements.intersection_offroad,
        agents_num=number_of_agents,
    )
    print(message)


def sigmoid(x):
    x = float(x)
    return np.exp(x) / (1 + np.exp(x))


def collided_done(py_measurements):
    """Define the main episode termination criteria"""
    m = py_measurements
    collided = (
        m["collision_vehicles"] > 0
        or m["collision_pedestrians"] > 0
        or m["collision_other"] > 0
        or m["collision_to_ego"] > 0
    )
    return bool(collided)  # or m["total_reward"] < -100)


def get_next_actions(measurements, is_discrete_actions):
    """Get/Update next action, work with way_point based planner.

    Args:
        measurements (dict): measurement data.
        is_discrete_actions (bool): whether use discrete actions

    Returns:
        dict: action_dict, dict of len-two integer lists.
    """
    # action_dict = {}
    # for actor_id, meas in measurements.items():
    #
    # return action_dict


if __name__ == "__main__":
    env = MultiCarlaEnv()
    # get_next_actions
    print(env.observation_space)
    print(env.observation_space.spaces['vehicle1'])
