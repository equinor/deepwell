import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt


class DeepWellEnv(gym.Env):
    
    def __init__(self):
        super().__init__()
        
        self.action_space = spaces.MultiDiscrete([3]*2)
        #Drilling state bounds:
        self.stateLow = np.array([0, 0, -10, -10, 0, 0])
        self.stateHigh = np.array([3000, 3000, 10, 10, 3000,2000])
        
        self.observation_space = spaces.Box(low=self.stateLow, high=self.stateHigh, dtype=np.float64)

        self.x0 = 300
        self.y0 = 0
        self.x0d = 0
        self.y0d = 3
        self.xtarget = 2500
        self.ytarget = 1500
        self.radius_target = 20
        self.stepsize = 1#Number of timesteps between each decision
        self.dirspeedlimit = 10
        self.state = self.init_states() #[x, y, xd, yd, xtarget, ytarget]
        
    
    def diff_eq(self, dd, d, d0):

        d = d + dd
        d0 = d0 + d
        return dd, d, d0

    def step(self, action):
        acc = (action -1) #Make acceleration input lay in range [-0.1 -> 0.1]
        done = False
        dist = np.linalg.norm([self.state[4]-self.state[0],self.state[5]-self.state[1]]) #Distance to target
        for _ in range(self.stepsize): #Calculate next states
            _,self.state[2],self.state[0] = self.diff_eq(acc[0],self.state[2],self.state[0])
            _,self.state[3],self.state[1] = self.diff_eq(acc[1],self.state[3],self.state[1])
        dist_new = np.linalg.norm([self.state[4]-self.state[0],self.state[5]-self.state[1]]) #New distance to target
        dist_diff = dist_new - dist
        velocity = np.linalg.norm([self.state[2],self.state[3]])
        
        
        reward = -dist_diff
        #Check if breaking the speedlimit
        if self.state[2] > self.dirspeedlimit or self.state[3] > self.dirspeedlimit:
            done = True   
            reward -=10   

        #Check if outside grid
        if (self.state[0]<self.stateLow[0]) or (self.state[1]<self.stateLow[1]) or (self.state[0]>self.stateHigh[0]) or (self.state[1]>self.stateHigh[1]):
            reward -=10
            done = True
        #Check if in radius of target
        if dist_new < self.radius_target:
            reward += 100
            done = True
        
        return self.state, reward, done, {}



    def init_states(self):
        self.state = np.array([self.x0,
            self.y0,
            self.x0d,
            self.y0d,
            self.xtarget,
            self.ytarget])
        return self.state
        
    def reset(self):
        self.init_states()
        return self.state
