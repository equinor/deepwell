# DeepWell Environments
Brief explination of the different OpenAI Gym environments in the repo

## DeepWellEnv
First functional 2D environment using cartesian coordinates and one target and the closest hazard in the observation space.

**Coordinates:**&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Cartesian\
**Observation space:**&nbsp;&nbsp;&nbsp;&nbsp;[target_xdist, target_ydist, xd, yd, hazard_xdist, hazard_ydist]\
**Action space:**&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;MultiDiscrete([3]*2)\
**Plotting:**&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Local plotting supported through render function inside environment\
**Preferred learn. alg:**&nbsp;&nbsp;&nbsp;TRPO

## DeepWellEnv_v2
Extension of DeepWellEnv including the second next target in the observation space.

**Coordinates:**&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Cartesian\
**Observation space:**&nbsp;&nbsp;&nbsp;&nbsp;[target_xdist1, target_ydist1, target_xdist2, target_ydist2, xd, yd, hazard_xdist, hazard_ydist]\
**Action space:**&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;MultiDiscrete([3]*2)\
**Plotting:**&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Local plotting supported through render function inside environment\
**Preferred learn. alg:**&nbsp;&nbsp;&nbsp;TRPO

## DeepWellEnvHER
Alternative 2D Goalenvironment inheriting from gym.GoalEnv with a single discrete action. Created for testing the Hindsight Experience Replay method wrapper. 

**Coordinates:**&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Cartesian\
**Observation space:**&nbsp;&nbsp;&nbsp;&nbsp;[target_xdist1, target_ydist1, xd, yd, hazard_xdist, hazard_ydist]\
**Action space:**&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Discrete(9)\
**Plotting:**&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Local plotting supported through render function inside environment\
**Preferred learn. alg:**&nbsp;&nbsp;&nbsp;HER (model_class = DQN)

## DeepWellEnv3d
Extension of DeepWellEnv_v2 to work in 3D.

**Coordinates:**&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Cartesian\
**Observation space:**&nbsp;&nbsp;&nbsp;&nbsp;[target_xdist1, target_ydist1, target_zdist1, target_xdist2, target_ydist2, target_zdist2,\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;xd, yd, zd, hazard_xdist, hazard_ydist, hazard_zdist]\
**Action space:**&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;MultiDiscrete([3]*3)\
**Plotting:**&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Local plotting supported through render function inside environment\
**Preferred learn. alg:**&nbsp;&nbsp;&nbsp;PPO2
## DeepWellEnvSpher
Modified version of DeepWellEnv3d to use spherical coordinates.

**Coordinates:**&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Spherical\
**Observation space:**&nbsp;&nbsp;&nbsp;&nbsp;[vert_angAcc, hor_angAcc, vert_angVel, hor_angVel, rel_vert_targetAng1, rel_hor_targetAng1,\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;rel_vert_targetAng2, rel_hor_targetAng2, rel_vert_hazardAng, rel_hor_hazardAng, target_dist1, target_dist2, hazard_dist2]\
**Action space:**&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Discrete(9)\
**Plotting:**&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Plotting supported through the PlotServer in main.py\
**Preferred learn. alg:**&nbsp;&nbsp;&nbsp;DQN

## DeepWellEnvSpherLevels
Inheriting from DeepWellEnvSpher to create different instances (levels) of the spherical environment with increasing difficulty. 

**Coordinates:**&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Spherical\
**Observation space:**&nbsp;&nbsp;&nbsp;&nbsp;[vert_angAcc, hor_angAcc, vert_angVel, hor_angVel, rel_vert_targetAng1, rel_hor_targetAng1,\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;rel_vert_targetAng2, rel_hor_targetAng2, rel_vert_hazardAng, rel_hor_hazardAng, target_dist1, target_dist2, hazard_dist2]\
**Action space:**&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Discrete(9)\
**Plotting:**&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Plotting supported through the PlotServer in main.py\
**Preferred learn. alg:**&nbsp;&nbsp;&nbsp;DQN

## DeepWellEnvSpherSmallObs
Modified version of DeepWellEnvSpher using a smaller observation space. Stable and easy-to-learn environment, but agents won't be able to steer away from hazards.

**Coordinates:**&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Spherical\
**Observation space:**&nbsp;&nbsp;&nbsp;&nbsp;[vert_angAcc, hor_angAcc, vert_angVel, hor_angVel, rel_vert_targetAng1, rel_hor_targetAng1]\
**Action space:**&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Discrete(9)\
**Plotting:**&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Plotting supported through the PlotServer in main.py\
**Preferred learn. alg:**&nbsp;&nbsp;&nbsp;PP02
