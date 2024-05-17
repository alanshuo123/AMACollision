To validate the performance of AMACollision under diverse driving scenarios, we adopt three various road structures provided by the Town03 map of Carla, as detailed below:

1. **T-Junction:** A three-way intersection which requires precise maneuvering of the *AVUT* when merging or yielding, e.g., right-of-way conflicts.

2. **Intersection:** A classic four-way crossroads without traffic signals, including multiple conflict points which tests the *AVUT*'s ability to coordinate with other agents.

3. **Roundabout:** This road introduces a circular segment, significantly increasing the complexity due to the need for continuous lane-following while managing entry, circulate and exit from the roundabout unpredictably.

For all three road structures, we deploy three distinct NPC configurations to enhance the complexity and variability of testing conditions. The first configuration comprises two four-wheeled vehicles. The second configuration adds one extra motorcycle to the first configuration. The third configuration is expanded to include two four-wheeled vehicles, one motorcycle, and one pedestrian. Each NPC is assigned a unique starting position and destination. 

For **RQ1**, we select the second NPC configuration and conduct experiments to compare the effectiveness of AMACollision with two baselines across all road structures. 

To address **RQ2**, we conduct comparative experiments using all three NPC configurations across each road structure. 

For **RQ3**, we employ ten distinct speed limits for *AVUT*, ranging from 1 m/s to 10 m/s, in 1 m/s increments. The experiments for RQ3 are conducted on each speed limit, utilizing the second NPC configuration across all road structures. The *AVUT* is subjected to 500 testing episodes for each experiment.
