import gym
from gym import error, spaces, utils
import numpy as np
import random


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


class DeepWellEnv(gym.GoalEnv):

    def __init__(self):
        super().__init__()

        self.action_space = spaces.Discrete(9)
        self.actions_dict = {   
            0:[-0.01, -0.01],
            1:[-0.01, 0],
            2:[-0.01, 0.01],
            3:[0, -0.01],
            4:[0, 0],
            5:[0, 0.01],
            6:[0.01, -0.01],
            7:[0.01, 0],
            8:[0.01, 0.01],
            }
            # x = action[0], z = action[1]. 3 options = [-0.01, 0, 0.01], number of unique actions: 3*3=9
            

        #Drilling state bounds:
        self.xmin = 0
        self.xmax = 3000
        self.zmin = 0
        self.zmax = 3000
        self.max_dist = 3*np.sqrt(self.xmax**2 + self.zmax**2)
        goal_low = np.array([0., 0.])
        goal_high = np.array([self.xmax, self.zmax])

        #self.state_low = np.array([self.xmin, self.zmin, -self.xmax, -self.zmax, -1, -1, goal_low[0], goal_low[1]])
        #self.state_high = np.array([self.xmax, self.zmax, self.xmax, self.zmax, 1, 1, goal_high[0], goal_high[1]])
        self.state_low = np.array([-self.xmax, -self.zmax, -1, -1])#, goal_low[0], goal_low[1]])
        self.state_high = np.array([self.xmax, self.zmax, 1, 1])#, goal_high[0], goal_high[1]])
        

        self.stepsize = 10
        
        self.state = self.init_state()
        obs = self.get_obs()

        self.observation_space = spaces.Dict({
            'achieved_goal':spaces.Box(low=goal_low, high=goal_high, dtype='float64'),
            'desired_goal':spaces.Box(low=goal_low, high=goal_high, dtype='float64'),
            'observation':spaces.Box(low=self.state_low, high=self.state_high, dtype='float64')
        })


    def init_state(self):
        self.x, self.z = random.randint(0,1000), 0.
        self.dx, self.dz = 0, 1
        self.targetball = {'center':np.array([random.randint(int(self.x + 100), 2500), random.randint(1500, 2500)]), 'R':100}
        self.goal = self.targetball['center']
        self.xdist1 = self.goal[0] - self.x 
        self.zdist1 = self.goal[1] - self.z 
        #self.dist_diff = goal_distance(self.goal, np.array([self.x, self.z]))
        self.dist_traveled = 0
        #self.state = np.array([self.x, self.z, self.xdist1, self.zdist1, self.dx, self.dz, self.goal[0], self.goal[1]])#, self.dist_diff])
        self.state = np.array([self.xdist1, self.zdist1, self.dx, self.dz])#, self.goal[0], self.goal[1]])#, self.dist_diff])
        return self.state

    
    def get_obs(self):
        pos = np.array([self.x, self.z])
        return {
            'observation': self.state.copy(),
            'achieved_goal': pos.copy(),
            'desired_goal': self.goal.copy(),
        }

    def reset(self):
        self.state = self.init_state()
        obs = self.get_obs()
        return obs


    def step(self, action):
        """Steps through one action decision and iterates substeps # times with diff eq """

        self.dist = np.linalg.norm([self.xdist1, self.zdist1]) #Distance to next target
        for _ in range(self.stepsize):   
            dx = self.actions_dict[action][0] + self.dx #update dx
            dz = self.actions_dict[action][1] + self.dz #update dy
            velocity = np.linalg.norm([dx, dz])
            if velocity == 0:
                velocity = 1
            normal_velocity = np.array([dx, dz])/velocity  # Normalised velocity
            self.dx = normal_velocity[0]
            self.dz = normal_velocity[1]
            #self.state[0] += self.state[2] #update pos with updated and normalized vel. vector
            #self.state[1] += self.state[3] 
            self.x += self.dx
            self.z += self.dz

        self.xdist1 = self.goal[0] - self.x  #x-axis distance to next target
        self.zdist1 = self.goal[1] - self.z  #z-axis distance to next target
        #self.state = np.array([self.x, self.z, self.xdist1, self.zdist1, self.dx, self.dz, self.goal[0], self.goal[1]])
        self.state = np.array([self.xdist1, self.zdist1, self.dx, self.dz])#
        obs = self.get_obs()
        info = None
        reward = self.compute_reward(obs['achieved_goal'], self.goal, info)

        done = False
        info = {'is_success':False}

        if self.dist_new < self.targetball['R']:
            print("Target hit!#############################################################################")
            done = True
            info['is_success'] = True

        if not self.valid_state():
            done = True

        self.dist_traveled += self.stepsize
        if self.dist_traveled > self.max_dist:
            done = True

        return obs, reward, done, info


    def compute_reward(self, achieved_goal, desired_goal, info):
        # Reward if moved closer to target and penalise if moved further away
        
        self.dist_new = np.linalg.norm([self.xdist1, self.zdist1])  
        dist_diff = self.dist_new - self.dist
        reward = -dist_diff
        
        if not self.valid_state():
            reward -= 3000

        #Check if inside target radius (reward)
        if self.dist_new < self.targetball['R']:
            reward += 3000

        #Check if maximum travel range has been reached
        if self.dist_traveled > self.max_dist:
            reward -= 3000

        if self.targethit(np.array([self.x, self.z]), self.goal): # < self.targetball['R']:
            reward += 3000

        return reward


    def targethit(self, achieved_goal, desired_goal):
        """Checks if position of drillbit is inside target ball. """
        if np.linalg.norm(achieved_goal - desired_goal) < self.targetball['R']:
            return True
        else:
            return False


    def valid_state(self):
        """Checks if state is within bounds """
        x = (self.xmin <= self.x) and (self.x <= self.xmax)
        z = (self.zmin <= self.z) and (self.z <= self.zmax)
        return x and z

