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
        self.stateLow = np.array([-3000, -3000, -1., -1.])
        self.stateHigh = np.array([3000, 3000, 1., 1.])
        
        self.observation_space = spaces.Box(low=self.stateLow, high=self.stateHigh, dtype=np.float64)
        self.xmin = 0
        self.xmax = 3000
        self.ymin = 0
        self.ymax = 3000
        self.x = 0.0            #decided in init_states()
        self.y = 0.0            #decided in init_states()
        self.xd0 = 0.0
        self.yd0 = 1.0
        self.xd = 0             #decided in init_states()
        self.yd = 0             #decided in init_states()
        self.xtarget = 0        #decided in init_states()
        self.ytarget = 0        #decided in init_states()
        self.xdist = 0          #decided in init_states()
        self.ydist = 0          #decided in init_states()
        self.radius_target = 100
        self.stepsize = 1       #Number of timesteps between each decision
        self.state = self.init_states() #[xdist, ydist, xd, yd]
        
    #Create figure to send to server
    def render(self, xcoord, ycoord):
        fig = plt.figure()
        subplot = fig.add_subplot(111)
        subplot.plot(xcoord,ycoord)
        plt.gca().invert_yaxis()
        subplot.scatter(self.xtarget,self.ytarget,s=150)
        plt.xlabel("Horizontal")
        plt.ylabel("Depth")
        return fig

    def step(self, action):
        acc = (action - 1)/100 #Make acceleration input lay in range [-0.1 -> 0.1]
        done = False
        dist = np.linalg.norm([self.xdist,self.ydist]) #Distance to target
        for _ in range(self.stepsize): #Calculate next states
            xd = acc[0] + self.xd           #update xd (unnormalized)
            yd = acc[1] + self.yd           #update yd (unnormalized)
            velocity = np.linalg.norm([xd,yd])
            if velocity == 0:
                velocity = 1
            normal_vel = np.array([xd, yd])/velocity
            self.xd = normal_vel[0]         #update normalized vel. vector 
            self.yd = normal_vel[1]         #update normalized vel. vector 
            self.x = self.x + self.xd       #update x 
            self.y = self.y + self.yd       #update y
        self.xdist = self.xtarget-self.x
        self.ydist = self.ytarget-self.y
        #Update state vector
        self.state[0] = self.xdist
        self.state[1] = self.ydist
        self.state[2] = self.xd
        self.state[3] = self.yd

        #Check new distance (reward)
        dist_new = np.linalg.norm([self.xdist,self.ydist]) #New distance to target       
        dist_diff = dist_new - dist
        reward = -dist_diff

        #Check if outside grid (reward)
        if (self.x<self.xmin) or (self.y<self.ymin) or (self.x>self.xmax) or (self.y>self.ymax):
            reward -=100
            done = True
        #Check if in radius of target (reward)
        if dist_new < self.radius_target:
            reward += 100
            done = True
        
        info = {'x':self.x, 'y':self.y, 'xt':self.xtarget, 'yt':self.ytarget}
        return self.state, reward, done, info



    def init_states(self):
        self.xtarget = random.randint(0,3000)
        self.ytarget = random.randint(1000,3000)
        self.x = random.randint(0,3000)
        self.y = random.randint(0,1000)
        self.xdist = self.x-self.xtarget
        self.ydist = self.y-self.ytarget
        self.xd = self.xd0
        self.yd = self.yd0
        self.state = np.array([self.xdist,
            self.ydist,
            self.xd,
            self.yd])
        return self.state
        
    def reset(self):
        self.init_states()
        return self.state
