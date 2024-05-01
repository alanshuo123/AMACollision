# advCollisionReward():
if ego collides to npc_vehicle_i, then npc_vehicle_i got a collisionReward
elif ego collides to other_actors, all npc_vehicle got a global_reward
elif ego offline or offroad, all npc_vehicle got a global_reward


#observation_space:
add distance_to_ego, rotation_to_ego