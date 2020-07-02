import gym
from gym import error, spaces, utils
#from gym.utils import seeding

import numpy as np
import matplotlib.pyplot as plt
import random

#gym.logger.set_level(40)



class WellPlot3Env(gym.Env):
    metadata = {'render.modes': ['console']}

    def __init__(self):
        super(WellPlot3Env, self).__init__()

        self.fig, self.ax = plt.subplots()

        self.reward_dict = {}
        self.actions_dict = {0:-0.01, 1:0, 2:0.01}        #Actions: 0:Decreaase, 1:Maintain, 2:Increase
        
        self.action_space = spaces.Tuple((spaces.Discrete(3), spaces.Discrete(3)))

        #Drilling state bounds:
        self.stateLow = np.array([[0., 0.,], [-10., -25.], [-10., -25.]])
        self.stateHigh = np.array([[1000, 2800,], [10., 25.], [1, 1]])
        
        self.observation_space = spaces.Box(low=self.stateLow, high=self.stateHigh)

        self.X = 1000 #Size in feet of position grid x perimeter
        self.Z = 2800 #Size in feet of position grid y perimeter

        self.default_reward = 0                      #This is the default reward for an action that has not been assigned another by the agent yet
        self.substeps = 10
        self.curve = []

        # Initialise starting point
        self.x, self.z = 0, 0
        self.dx, self.dz = 0, 0
        self.d2x, self.d2z = 0, 0
        
        self.init_state = np.array([[self.x, self.z], [self.dx, self.dz], [self.d2x, self.d2z]])
        self.state = self.init_state
        self.targetball = {'center':np.array([600, 1000]), 'R':50}  
        # Extend this to list of several targetballs and randomize variables later

        self.update_plot()

    ###################### SETUP HELP METHODS BELOW #######################

    #Method for updating plot in case user edits default values (ex: size,init/end state)
    def update_plot(self):
        self.ax.cla()
        self.ax.plot(*zip(*self.curve), 'o-')
        self.ax.plot(self.init_state[0,0], self.init_state[0,1], "or", label="start", markersize=10)
        circle = plt.Circle(self.targetball['center'], self.targetball['R'], color='g', label='target')
        self.ax.add_artist(circle)

        self.ax.set_xlim(-100, self.X)
        self.ax.set_ylim(-100, self.Z)
        self.ax.invert_yaxis()
        self.ax.legend()


    ##################### OPENAI GYM ENV METHODS BELOW #######################

    def diff_eq(self, d2, d1, d0):
        d1 += d2
        d0 += d1
        return d2, d1, d0


    def targethit(self):
        relpos = self.state[0] - self.targetball['center']
        if np.linalg.norm(relpos) < self.targetball['R']:
            return True
        else:
            return False


    def step(self, action):
        actionx = self.actions_dict[action[0]]
        actionz = self.actions_dict[action[1]]
        for i in range(self.substeps):
            self.d2x, self.dx, self.x = self.diff_eq(actionx, self.dx, self.x)
            self.d2z, self.dz, self.z = self.diff_eq(actionz, self.dz, self.z)
            self.curve.append([self.x, self.z])

        self.state = np.array([[self.x, self.z], [self.dx, self.dz], [self.d2x, self.d2z]])
        done = self.targethit()
        #reward = self.get_reward(self.state)
        #return self.state, reward, done, {}


    def render(self, path_x,path_y):

        plt.xlim([0,(self.grid_width-1)*self.distance_points])
        plt.ylim([(self.grid_height-1)*self.distance_points,0])

        plt.xlabel('Depth') 
        plt.ylabel('Horizontal  ')
        self.subplot.plot(path_x, path_y)
        self.subplot.grid()
        self.subplot.set_axisbelow(True)
        
        return self.fig

    def reset(self):
        self.state = self.init_state
        return self.state

    def close(self):
        return self.fig

    
    ########################### BONUS METHODS BELOW ###########################

    #Returns the next step without changing state
    def check_step(self,action):
        action = self.actions_dict[action]
        return np.array([self.state[0]+action[0], self.state[1]+action[1]])
    

    def valid_state(self,state):
        return (0 <= state[0] <= (self.grid_width-1)*self.distance_points) and (0 <= state[1] <= (self.grid_height-1)*self.distance_points)
    

    #Allows you to get reward for specified state
    def get_reward(self, state):
        state = tuple(state)
        if self.valid_state(state)==0:
            return -1
        if state in self.reward_dict:
            return self.reward_dict[state]
        else:
            self.reward_dict[state] = self.default_reward
            return self.default_reward


    #Allows you to set reward for specified state
    def set_reward(self, state, reward):
        state = tuple(state)
        self.reward_dict[state] = reward


test = WellPlot3Env()
for i in range(100):
    test.step([random.randint(0, 2), random.randint(0, 2)])
test.update_plot()
plt.show()