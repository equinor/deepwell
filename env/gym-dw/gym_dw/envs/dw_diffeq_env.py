import gym
from gym import error, spaces, utils
import numpy as np
import random


class DwDiffeqEnv(gym.Env):
    metadata = {'render.modes': ['console']}

    def __init__(self):
        super().__init__()

        self.actions_dict = {0:-0.01, 1:0, 2:0.01}        #Actions: 0:Decreaase, 1:Maintain, 2:Increase
        self.action_space = spaces.MultiDiscrete([3]*2)

        #Drilling state bounds:
        self.stateLow = np.array([[0., 0.], [-1, -1], [0., 0.]])
        self.stateHigh = np.array([[3000., 3000.], [1, 1], [3000., 3000.]])
        self.substeps = 1
        self.state = self.init_state()
        self.observation_space = spaces.Box(low=self.stateLow, high=self.stateHigh, dtype=np.float64)

    def init_state(self):
        self.x, self.z = random.randint(0,1000), 0.
        self.dx, self.dz = 0, 1
        self.targetball = {'center':np.array([random.randint(int(self.x*1.1 + 100), 2500), random.randint(1500, 2500)]), 'R':150}
        self.state = np.array([[self.x, self.z], [self.dx, self.dz], self.targetball['center']])
        return self.state


    def step(self, action):
        """Steps through one action decision and iterates substeps # times with diff eq """

        old_dist = np.linalg.norm(self.state[0] - self.targetball['center'])
        
        for _ in range(self.substeps):    
            dx = self.actions_dict[action[0]] + self.state[1,0] #update dx
            dz = self.actions_dict[action[1]] + self.state[1,1] #update dy
            velocity = np.linalg.norm([dx,dz])
            if velocity == 0:
                velocity = 1
            self.state[1] = np.array([dx, dz])/velocity  # Normalised velocity
            self.state[0] += self.state[1] #update pos with updated and normalized vel. vector
        
        new_dist = np.linalg.norm(self.state[0] - self.targetball['center'])
        dist_diff = new_dist - old_dist

        reward, done = self.get_reward(dist_diff)

        return self.state, reward, done, {}


    def get_reward(self, dist_diff):
        # Reward if moved closer to target and penalise if moved further away
        done = False
        reward = - dist_diff
        
        valid = self.valid_state()
        # Done if it moves outside bounds
        if not valid:
            done = True
            reward -= 10

        # Reward if target is reached
        if self.targethit():
            reward += 100
            done = True

        return reward, done


    def targethit(self):
        """Checks if position of drillbit is inside target ball. """
        relpos = self.state[0] - self.targetball['center']
        if np.linalg.norm(relpos) < self.targetball['R']:
            print("Target hit!#############################################################################")
            return True
        else:
            return False


    def valid_state(self):
        """Checks if state is within bounds """
        x = (self.stateLow[0,0] <= self.state[0,0]) and (self.state[0,0] <= self.stateHigh[0,0])
        z = (self.stateLow[0,1] <= self.state[0,1]) and (self.state[0,1] <= self.stateHigh[0,1])
        return x and z


    def reset(self):
        self.init_state()
        return self.state


