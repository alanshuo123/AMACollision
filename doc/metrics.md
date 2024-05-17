# Evaluation Metrics

To evaluate the performance of $AMACollision$, we employ following metrics.  We compute the average value of  $TTFC$, $OL$, $TTD$ during all episodes in the experiment, to assess the efficiency of AMACollision in generating failure scenarios. 

$R_{collision}$ and $R_{destination}$ represent the rates of collision occurrences and successful arrival at the destination, respectively.  Considering that AVUT can recover from low-intensity collisions and continue towards its destination, the sum of the $R_{collision}$ and $R_{destination}$ does not necessarily equal. 

$C_{EGO}$ assesses the frequency of meaningful interactions between the NPC and the ego vehicle, reflecting the aggressiveness and relevance of the adversarial strategies.  

$LDE$ evaluates the realism of generated trajectories by computing the lateral deviation error between  NPC’s trajectory and the optimal trajectory planned by localplanner in Carla.

The calculation rules of these metrics are given here:

## **$R_{collision}$**

The collision rate, defined as the frequency of collisions between the *AVUT* and NPCs, computed across all episodes. Higher values are preferable as they indicate more collisions involving the *AVUT*.

$$
\text{$R_{collision}$} = \frac {\textit{Num}(\text{episodes with collisions involving AVUT})} {\textit{Num}(\text{total episodes})}
$$

where Num(⋅) means the number.  

## **$R_{destination}$**

The success rate of the *AVUT* reaching its destination, computed across all episodes. Lower values indicate $AMACollision$'s greater effectiveness in preventing the *AVUT* from completing its mission.



$$
\text{$R_{destination}$} = \frac {\textit{Num}(\text{episode in which AVUT reaches destination})} {\textit{Num}(\text{total episodes})}
$$



## **$OL$**

 The number of instances where the *AVUT* deviates from its lane per episode, indicating steering errors. Higher values suggest more effective adversarial interventions by $AMACollision$.



$$
\text{$OL$} = \frac {\textit{Num}(\text{offlane times})} {\textit{Num}(\text{total episodes})}
$$



## **$TTFC$**

The timesteps taken to encounter the first collision in an episode. Lower values are desirable as they indicate faster generation of collision scenarios.



$$
\text{$TTFC$} = \frac {\textit{Sum}(\text{timesteps it takes to first collision per episode})} {\textit{Num}(\text{total episodes})}
$$



## **$TTD$**

The number of timesteps taken for the *AVUT* to reach its destination in an episode. Higher values are preferable in contrast to $R_{destination}$.



$$
\text{$TTD$} = \frac {\textit{Sum}(\text{timesteps it takes for AVUT to reach destination per episode})} {\textit{Num}(\text{total episodes})}
$$



## **$C_{EGO}$**

The proportion of a NPC’s collisions involving the *AVUT* out of its total collisions across all episodes. $C_{EGO}$ reflects the aggressiveness of adversarial NPCs. Higher values are preferable, as they indicate more meaningful interactions involving the *AVUT*.



$$
\text{$C_{EGO}$} = \frac {\textit{Num}(\text{collision times involving AVUT})} {\textit{Num}(\text{total collision times})}
$$



## **$LDE$**

The lateral deviation error, measured as the NPC’s average deviation from the optimal trajectory per episode. Lower values indicate greater realism of generated scenarios.

The realism of driving trajectories is commonly assessed using distance-based evaluation metrics, which compare the generated trajectories to a real trajector. Acoordingly, we propose $LDE$, a distance-based scenario metric to evaluate the realism of generated trajectories, which measures the lateral deviation of the NPC's driving trajectory from the optimal trajectory planned by the $local planner$ in Carla. Lower $LDE$ values suggest a higher degree of realism in the NPCs’ movement patterns. The Root Mean Square Error (RMSE) of the LDE is then computed as an indicator of the trajectory's authenticity:

$$
LDE_{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} \left( lateral\_error(Loc_i, Waypoint_i) \right)^2}
$$

Where $Loc_i$ is the vehicle's position at time step $i$, $Waypoint_i$ is the optimal path point at time step $i$, and $lateral\_error$ is the lateral deviation between $Loc_i$ and $Waypoint_i$.
