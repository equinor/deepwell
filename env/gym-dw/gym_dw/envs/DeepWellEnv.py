import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
import random


class DeepWellEnv(gym.Env):
    
    def __init__(self):
        super().__init__()
        
        self.action_space = spaces.MultiDiscrete([3]*2)
        #Drilling state bounds:
        self.stateLow = np.array([0, 0, -1., -1., 0, 1000])
        self.stateHigh = np.array([3000, 3000, 1., 1., 3000,3000])
        
        self.observation_space = spaces.Box(low=self.stateLow, high=self.stateHigh, dtype=np.float64)
        self.x0 = 0.0 #decided in init_states()
        self.y0 = 0.0 #decided in init_states()
        self.x0d = 0.0
        self.y0d = 1.0
        self.xtarget = 0 #decided in init_states()
        self.ytarget = 0 #decided in init_states()
        self.radius_target = 100
        self.stepsize = 1#Number of timesteps between each decision
        self.state = self.init_states() #[x, y, xd, yd, xtarget, ytarget]
        
          
    def step(self, action):
        acc = (action - 1)/10 #Make acceleration input lay in range [-0.1 -> 0.1]
        done = False
        dist = np.linalg.norm([self.state[4]-self.state[0],self.state[5]-self.state[1]]) #Distance to target
        for _ in range(self.stepsize): #Calculate next states
            xd = acc[0] + self.state[2] #update xd
            yd = acc[1] + self.state[3] #update yd
            velocity = np.linalg.norm([xd,yd])
            if velocity == 0:
                velocity = 1
            normal_vel = np.array([xd, yd])/velocity
            self.state[2] = normal_vel[0]
            self.state[3] = normal_vel[1]
            self.state[0] = self.state[0] + normal_vel[0] #update x with updated and normalized vel. vector 
            self.state[1] = self.state[1] + normal_vel[1] #update y with updated and normalized vel. vector 
        dist_new = np.linalg.norm([self.state[4]-self.state[0],self.state[5]-self.state[1]]) #New distance to target       
        dist_diff = dist_new - dist
        reward = -dist_diff

        #Check if outside grid
        if (self.state[0]<self.stateLow[0]) or (self.state[1]<self.stateLow[1]) or (self.state[0]>self.stateHigh[0]) or (self.state[1]>self.stateHigh[1]):
            reward -=100
            done = True
        #Check if in radius of target
        if dist_new < self.radius_target:
            reward += 100
            done = True
        
        return self.state, reward, done, {}



    def init_states(self):
        self.xtarget = random.randint(0,3000)
        self.ytarget = random.randint(1000,3000)
        self.x0 = random.randint(0,3000)
        self.y0 = random.randint(0,1000)
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
