import math
import numpy as np
import carla

def calculate_forward_velocity(vehicle):
    
    vehicle_location = vehicle.get_location()
    vehicle_rotation = vehicle.get_rotation()
    vehicle_velocity = vehicle.get_velocity()

    
    pitch = math.radians(vehicle_rotation.pitch)
    yaw = math.radians(vehicle_rotation.yaw)
    roll = math.radians(vehicle_rotation.roll)

    forward_vector = carla.Vector3D()
    forward_vector.x = math.cos(yaw) * math.cos(pitch)
    forward_vector.y = math.sin(yaw) * math.cos(pitch)
    forward_vector.z = math.sin(pitch)

    forward_velocity = vehicle_velocity.x * forward_vector.x + vehicle_velocity.y * forward_vector.y + vehicle_velocity.z * forward_vector.z

    return forward_velocity

def calculate_angle(vector1, vector2):
    cos_theta = (vector1.x * vector2.x + vector1.y * vector2.y + vector1.z * vector2.z) / \
                ((math.sqrt(vector1.x * vector1.x + vector1.y * vector1.y + vector1.z * vector1.z) *
                  math.sqrt(vector2.x * vector2.x + vector2.y * vector2.y + vector2.z * vector2.z)))
    theta = np.arccos(cos_theta)
    # print(cos_theta)
    return theta


# def calculate_distance(vector1, vector2):
#     distance = math.sqrt(pow(vector1.x - vector2.x, 2) + pow(vector1.y - vector2.y, 2) + pow(vector1.z - vector2.z, 2))
#     return distance
def calculate_distance(x,y,z, ego_x,ego_y,ego_z):
    distance = math.sqrt(pow(x - ego_x, 2) + pow(y - ego_y, 2) + pow(z - ego_z, 2))
    return distance


def calculate_safe_distance(velocity):
    '''
        Calculate safe distance
        param: velocity (x,y,z)
        return: safe_distance
    '''
    #braking deceleration = 0.6*9.8 m/s^2
    speed = math.sqrt(pow(velocity.x,2) + pow(velocity.y,2) + pow(velocity.z,2))
    u = 0.6
    safe_distance = (velocity_x * velocity_x) / (2 * 9.8 * u)
    return safe_distance


# Calculate collision probability
def calculate_collision_probability(safe_distance, current_distance):
    collision_probability = None
    if current_distance >= safe_distance:
        collision_probability = 0
    elif current_distance < safe_distance:
        collision_probability = (safe_distance - current_distance) / safe_distance
    return collision_probability


'''
x->forward, y->right, z->height

vehicle.get_location().x/y/z
get_acceleration.x/y/z
rotation = vehicle.get_transform().rotation.roll/pitch/yaw
(pitch, yaw, roll) -->方向向量(x,y,z)
x = math.cos(pitch) * math.cos(yaw)
y = math.cos(pitch) * math.sin(yaw)
z = math.sin(pitch)
velocity = vehicle.get_velocity().x,y,z
'''

def get_collision_probability(actor1, actor2):

    return 0

def get_collision_probabilityv0(agents, ego, agents_len,  z_axis):
    ego_speed = ego.get_velocity()
    ego_transform = ego.get_transform()
    probability = probability1 = probability2 = probability3 = 0
    break_distance = calculate_safe_distance(ego_speed)

    for i in range(1, agents_len):
        transform = agents[i].state.transform
        current_distance = calculate_distance(transform.position, ego_transform.position)
        # print('current distance: ', current_distance)
        if current_distance > 40:
            continue
        if ego_transform.rotation.y - 10 < transform.rotation.y < ego_transform.rotation.y + 10:
            # In this case, we can believe the ego vehicle and obstacle are on the same direction.
            vector = transform.position - ego_transform.position
            if ego_transform.rotation.y - 10 < calculate_angle(vector, z_axis) < ego_transform.rotation.y + 10:
                # In this case, we can believe the ego vehicle and obstacle are driving on the same lane.
                safe_distance = break_distance
                probability1 = calculate_collision_probability(safe_distance, current_distance)
            else:
                # In this case, the ego vehicle and obstacle are not on the same lane. They are on two parallel lanes.
                safe_distance = 1
                probability2 = calculate_collision_probability(safe_distance, current_distance)
        else:
            # In this case, the ego vehicle and obstacle are either on the same direction or the same lane.
            safe_distance = 5
            probability3 = calculate_collision_probability(safe_distance, current_distance)
        new_probability = probability1 + (1 - probability1) * 0.2 * probability2 + \
                          (1 - probability1) * 0.8 * probability3
        if new_probability > probability:
            probability = new_probability
    # print(probability)
    return str(probability)