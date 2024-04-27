import numpy as np
from pprint import pprint
from macad_gym.carla.Utils import calculate_distance


class Reward(object):
    def __init__(self):
        self.reward = 0.0
        self.prev = None
        self.curr = None

    def compute_reward(self, prev_measurement, curr_measurement, ego_prev_measurement, ego_curr_measurement, flag):
        self.prev = prev_measurement
        self.curr = curr_measurement
        self.ego_prev = ego_prev_measurement
        self.ego_curr = ego_curr_measurement
        # pprint (curr_measurement["next_command"])
        if flag == "custom":
            return self.egoReward()
        elif flag == "advrs":
            return self.advCollisionRewardv2()
        elif flag == "advrs1":
            return self.advCollisionRewardv1()
        elif flag == "none":
            return self._None_()
        # elif flag == "corl2017":
        #     return self.compute_reward_corl2017()
        # elif flag == "lane_keep":
        #     return self.compute_reward_lane_keep()

    def _None_(self):
        return 0.0

    '''
    advCollisionReward() version1

    only consider the situation that NPCs collide with ego vehicle,then give NPC 
    
    if ego collides to npc_vehicle_i, then npc_vehicle_i got a collisionReward
    elif ego collides to other_actors, all npc_vehicle got a global_reward
    elif ego offline or offroad, all npc_vehicle got a global_reward
    '''

    def advCollisionRewardv1(self):
        self.reward = 0.0
        distance = self.prev["distance_to_ego"] - self.curr["distance_to_ego"]
        self.reward += np.clip(distance * 0.5, -5, 5)
        speed_change = self.curr["forward_speed"] - self.prev["forward_speed"]
        if self.curr["forward_speed"] < 2:
            self.reward -=0.5
        elif self.curr["forward_speed"] < 10:
            self.reward += np.clip(speed_change, -1, 1)
        # self.reward += np.clip(self.curr["forward_speed"], 0.0, 30.0) * 0.01

        new_collision_to_ego = (
                self.curr["collision_to_ego"] - self.prev["collision_to_ego"]
        )
        new_collision_to_others = (
                self.curr["intersection_offroad"] + self.curr["collision_vehicles"] + self.curr["collision_other"] +
                self.curr["collision_pedestrians"]
                - self.prev["intersection_offroad"] - self.prev["collision_vehicles"] - self.prev["collision_other"] -
                self.prev["collision_pedestrians"]
        )
        new_own_offlane = (  # offline和offroad区分开
                self.curr["intersection_otherlane"]
                - self.prev["intersection_otherlane"]
        )
        if new_collision_to_ego:
            self.reward += new_collision_to_ego * 100
        # if new_ego_offlane + new_ego_collision_to_others:
        #     self.reward += new_ego_offlane * 0.5 + new_ego_collision_to_others
        if new_collision_to_others:
            self.reward -= new_collision_to_others*5
        if new_own_offlane:
            self.reward += new_own_offlane * 0.5

        return self.reward

    '''
    advCollisionReward() version2 

    add distance_reward, consider the distance to ego_vehicle 

    only consider the situation that NPCs collide with ego vehicle,then give NPC 
    
    if ego collides to npc_vehicle_i, then npc_vehicle_i got a collisionReward
    elif ego collides to other_actors, all npc_vehicle got a global_reward
    elif ego offline or offroad, all npc_vehicle got a global_reward
    '''

    def advCollisionRewardv2(self):
        self.reward = 0.0
        cur_dist = self.curr["distance_to_goal"]
        prev_dist = self.prev["distance_to_goal"]
        self.reward += np.clip(prev_dist - cur_dist, -1.0, 1.0)
        if(self.curr["forward_speed"] < 0.5):
            self.reward -= (0.5-self.curr["forward_speed"])
        # elif self.curr["forward_speed"] < 5:
        #     self.reward += np.clip(self.curr["forward_speed"], 0.0, 10.0) * 0.15
        # else:
        #     self.reward -= 0.1
        if(self.prev["distance_to_ego"]-self.curr["distance_to_ego"]) > 0 :
            if self.curr["distance_to_ego"] < 2 and self.curr["forward_speed"] > 0:
                self.reward += (2-self.curr["distance_to_ego"])
            elif self.curr["distance_to_ego"] < 5 and self.curr["forward_speed"] > 0.5:
                self.reward += 0.2
            elif self.curr["distance_to_ego"] < 8 and self.curr["forward_speed"] > 1:
                self.reward += 0.1
        elif (prev_dist-cur_dist) < 0:
            self.reward += np.clip(self.prev["distance_to_ego"]-self.curr["distance_to_ego"], -0.5,0)

        new_collision_to_ego = (
                self.curr["collision_to_ego"] - self.prev["collision_to_ego"]
        )
        new_collision_to_others  = (
            self.curr["collision_vehicles"]+self.curr["collision_other"]+self.curr["collision_pedestrians"]
            -self.prev["collision_vehicles"]-self.prev["collision_other"]-self.prev["collision_pedestrians"]
        )
        new_own_offlane = ( # offline和offroad区分开
            self.curr["intersection_otherlane"]
            -self.prev["intersection_otherlane"]
        )
        new_ego_offlane = (
            self.ego_curr["intersection_otherlane"]
            - self.ego_prev["intersection_otherlane"]
        )
        new_ego_collision_to_others = (
            self.ego_curr["collision_vehicles"] + self.ego_curr["collision_other"] + self.ego_curr["collision_pedestrians"]
            - self.ego_prev["collision_vehicles"] - self.ego_prev["collision_other"] - self.ego_prev["collision_pedestrians"]
        )
        if new_collision_to_ego:
            self.reward += new_collision_to_ego * 100
        if new_ego_offlane + new_ego_collision_to_others:
            self.reward += new_ego_offlane * 0.1 + new_ego_collision_to_others*1
        if new_collision_to_others:
            self.reward -= new_collision_to_others*10
        if new_own_offlane:
            self.reward -= new_own_offlane*0.2

        return self.reward


    def egoReward(self):
        self.reward = 0.0
        # self.reward += np.clip(self.curr["forward_speed"], 0.0, 30.0) * 0.01
        new_ego_offlane = (
                self.ego_curr["intersection_offroad"] + self.ego_curr["intersection_otherlane"]
                -self.ego_prev["intersection_offroad"]-self.ego_prev["intersection_otherlane"]
        )
        new_ego_collision_to_others = (
                self.ego_curr["collision_other"] + self.ego_curr["collision_vehicles"] + self.ego_curr[
            "collision_pedestrians"]
                - self.ego_prev["collision_other"] - self.ego_prev["collision_vehicles"] - self.ego_prev[
                    "collision_pedestrians"]
        )
        if (new_ego_collision_to_others):
            self.reward -= new_ego_collision_to_others*10

        self.reward -= new_ego_offlane*0.5

        self.reward += 0.1

        #add a neg-rwd to get the ego faster to collision, but only suitable for centralized training
        #self.reward -= 0.01

        return self.reward

    def compute_reward_corl2017(self):
        self.reward = 0.0
        cur_dist = self.curr["distance_to_goal"]
        prev_dist = self.prev["distance_to_goal"]
        # Distance travelled toward the goal in m
        self.reward += np.clip(prev_dist - cur_dist, -10.0, 10.0)
        # Change in speed (km/h)
        self.reward += 0.05 * (
                self.curr["forward_speed"] - self.prev["forward_speed"])
        # New collision damage
        self.reward -= .00002 * (
                self.curr["collision_vehicles"] +
                self.curr["collision_pedestrians"] + self.curr["collision_other"] -
                self.prev["collision_vehicles"] -
                self.prev["collision_pedestrians"] - self.prev["collision_other"])

        # New sidewalk intersection
        self.reward -= 2 * (self.curr["intersection_offroad"] -
                            self.prev["intersection_offroad"])

        # New opposite lane intersection
        self.reward -= 2 * (self.curr["intersection_otherlane"] -
                            self.prev["intersection_otherlane"])

        return self.reward


    def compute_reward_lane_keep(self):
        self.reward = 0.0
        # Speed reward, up 30.0 (km/h)
        self.reward += np.clip(self.curr["forward_speed"], 0.0, 30.0) / 10
        # New collision damage
        new_damage = (
            self.curr["collision_vehicles"] +
            self.curr["collision_pedestrians"] + self.curr["collision_other"] -
            self.prev["collision_vehicles"] -
            self.prev["collision_pedestrians"] - self.prev["collision_other"])
        if new_damage:
            self.reward -= 100.0
        # Sidewalk intersection
        self.reward -= self.curr["intersection_offroad"]
        # Opposite lane intersection
        self.reward -= self.curr["intersection_otherlane"]

        return self.reward
    '''
    advCollisionProbability()
    * distance_to_ego ---> probability of collision to ego(PoCTE),  as reward of NPC
    '''

    def advCollisionProbability(self):
        self.reward = 0.0

        return self.reward

    def advrs(self):
        self.reward = 0.0
        # distance and speed
        x = self.curr["x"]
        y = self.curr["y"]
        z = self.curr["z"]
        ego_x = self.ego_curr["x"]
        ego_y = self.ego_curr["y"]
        ego_z = self.ego_curr["z"]
        disance = calculate_distance(x, y, z, ego_x, ego_y, ego_z)
        if disance < 10:
            self.reward += np.clip(1 / disance, 0, 1)
        else:
            self.reward -= np.clip(disance / 20, 0, 1)*0.5 
        if self.curr["forward_speed"] < 2:
            self.reward -= self.curr["forward_speed"]*0.1
        else:
            self.reward += np.clip(self.curr["forward_speed"], 0.0, 30.0) * 0.01

        new_collision_to_ego = (
                self.curr["collision_to_ego"] - self.prev["collision_to_ego"]
        )
        new_ego_offlane = (
                self.ego_curr["intersection_offroad"] + self.ego_curr["intersection_otherlane"]
        )
        new_ego_collision_to_others = (
                self.ego_curr["collision_other"] + self.ego_curr["collision_vehicles"] + self.ego_curr[
            "collision_pedestrians"]
                - self.ego_prev["collision_other"] - self.ego_prev["collision_vehicles"] - self.ego_prev[
                    "collision_pedestrians"]
        )

        if new_collision_to_ego:
            self.reward += new_collision_to_ego * 10
        if new_ego_offlane + new_ego_collision_to_others:
            self.reward += new_ego_offlane * 0.5 + new_ego_collision_to_others

        self.reward += self.curr["intersection_offroad"] * 0.5
        self.reward += self.curr["intersection_otherlane"] * 0.5

        return self.reward

    def compute_reward_custom(self):
        self.reward = 0.0
        cur_dist = self.curr["distance_to_goal"]
        prev_dist = self.prev["distance_to_goal"]
        self.reward += np.clip(prev_dist - cur_dist, -10.0, 10.0)
        self.reward += np.clip(self.curr["forward_speed"], 0.0, 30.0) / 10
        new_damage = (
                self.curr["collision_vehicles"]  # + self.curr["collision_pedestrians"]
                + self.curr["collision_other"] -
                self.prev["collision_vehicles"] -  # - self.prev["collision_pedestrians"]
                self.prev["collision_other"])
        if new_damage:
            self.reward -= 100.0

        self.reward -= self.curr["intersection_offroad"] * 0.5
        self.reward -= self.curr["intersection_otherlane"] * 0.5

        if self.curr["next_command"] == "REACH_GOAL":
            self.reward += 100
        if self.curr["next_command"] == "LANE_FOLLOW":
            self.reward += 0.5
        return self.reward

    # def compute_reward_corl2017(self):
    #     self.reward = 0.0
    #     cur_dist = self.curr["distance_to_goal"]
    #     prev_dist = self.prev["distance_to_goal"]
    #     # Distance travelled toward the goal in m
    #     self.reward += np.clip(prev_dist - cur_dist, -10.0, 10.0)
    #     # Change in speed (km/h)
    #     self.reward += 0.05 * (
    #         self.curr["forward_speed"] - self.prev["forward_speed"])
    #     # New collision damage
    #     self.reward -= .00002 * (
    #         self.curr["collision_vehicles"] +
    #         self.curr["collision_pedestrians"] + self.curr["collision_other"] -
    #         self.prev["collision_vehicles"] -
    #         self.prev["collision_pedestrians"] - self.prev["collision_other"])

    #     # New sidewalk intersection
    #     self.reward -= 2 * (self.curr["intersection_offroad"] -
    #                         self.prev["intersection_offroad"])

    #     # New opposite lane intersection
    #     self.reward -= 2 * (self.curr["intersection_otherlane"] -
    #                         self.prev["intersection_otherlane"])

    #     return self.reward

    # def compute_reward_lane_keep(self):
    #     self.reward = 0.0
    #     # Speed reward, up 30.0 (km/h)
    #     self.reward += np.clip(self.curr["forward_speed"], 0.0, 30.0) / 10
    #     # New collision damage
    #     new_damage = (
    #         self.curr["collision_vehicles"] +
    #         self.curr["collision_pedestrians"] + self.curr["collision_other"] -
    #         self.prev["collision_vehicles"] -
    #         self.prev["collision_pedestrians"] - self.prev["collision_other"])
    #     if new_damage:
    #         self.reward -= 100.0
    #     # Sidewalk intersection
    #     self.reward -= self.curr["intersection_offroad"]
    #     # Opposite lane intersection
    #     self.reward -= self.curr["intersection_otherlane"]

    #     return self.reward

    # def destory(self):
    #     pass
