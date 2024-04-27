import os
import sys
import logging

from gym.envs.registration import register

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "yes please"
LOG_DIR = os.path.join(os.getcwd(), "logs")
if not os.path.isdir(LOG_DIR):
    os.mkdir(LOG_DIR)

# Init and setup the root logger
logging.basicConfig(filename=LOG_DIR + '/macad-gym.log', level=logging.DEBUG)

# Fix path issues with included CARLA API
sys.path.append(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "carla/PythonAPI"))

# Declare available environments with a brief description
_AVAILABLE_ENVS = {
    'Town03I4C2B1P4-v0':{
        "entry_point":
        "macad_gym.envs:Town03I4C2B1P4",
        "description":
        "Observable 4 way Intersection Multi-Agent scenario with "
        "2 Cars 4 ped 1 bike in Town3, version 0"
    },
    'Town03I3C2-v0':{
        "entry_point":
        "macad_gym.envs:Town03I3C2",
        "description":
        "Observable 3 way Intersection Multi-Agent scenario with "
        "2 Cars  in Town3, version 0"
    },
    'Town03I3C2_measure-v0':{
        "entry_point":
        "macad_gym.envs:Town03I3C2_measure",
        "description":
        "Observable 3 way Intersection Multi-Agent scenario with "
        "2 Cars  in Town3, version 0"
    },
    'Town03I3C2_measure_continuous-v0':{
        "entry_point":
        "macad_gym.envs:Town03I3C2_measure_continuous",
        "description":
        "Observable 3 way Intersection Multi-Agent scenario with "
        "2 Cars  in Town3, continuous action, version 0"
    },
    'Town03_roundabout-v0':{
        "entry_point":
        "macad_gym.envs:Town03_roundabout",
        "description":
        "Observable roundabout Multi-Agent scenario with "
        "2 Cars  in Town3, discrete action, version 0"
    }
}

for env_id, val in _AVAILABLE_ENVS.items():
    register(id=env_id, entry_point=val.get("entry_point"))


def list_available_envs():
    print("Environment-ID: Short-description")
    import pprint
    available_envs = {}
    for env_id, val in _AVAILABLE_ENVS.items():
        available_envs[env_id] = val.get("description")
    pprint.pprint(available_envs)
