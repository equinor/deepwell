import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
import random



class DeepWellEnv(gym.Env):
    
    def __init__(self):
        super().__init__()
               
        self.stepsize = 10       #Number of timesteps between each decision

        self.xmin = 0
        self.xmax = 3000         #(>1000)
        self.ymin = 0
        self.ymax = 3000         #(>1000)
        self.x = 0.0             #decided in init_states()
        self.y = 0.0             #decided in init_states()
        self.xd0 = 0.0
        self.yd0 = 1.0
        self.xd = 0              #decided in init_states()
        self.yd = 0              #decided in init_states()
        self.xdist1 = 0          #decided in init_states()
        self.ydist1 = 0          #decided in init_states()
        self.min_tot_dist = 0
        self.dist_traveled = 0
        self.rel_max_dist = 3    #decides when to exit episode (When dist_traveled > rel_max_dist*distance traveled)
        self.max_tot_dist = 0    #decided in init_states()
        
        self.numtargets = 10      #==SET NUMBER OF TARGETS==#
        self.radius_target = 50
        self.target_hits = 0        
        self.targets = []        #decided in init_states()
        
        self.state = self.init_states() #[xdist1, ydist1, xd, yd]
        #Set action and observation space
        self.action_space = spaces.MultiDiscrete([3]*2)
        self.stateLow = np.array([ -self.xmax, -self.ymax,  -1., -1.])
        self.stateHigh = np.array([ self.xmax, self.ymax, 1., 1.])
        self.observation_space = spaces.Box(low=self.stateLow, high=self.stateHigh, dtype=np.float64)

    #Create figure to send to server
    def render(self, xcoord, ycoord, xt, yt):
        fig = plt.figure()
        subplot = fig.add_subplot(111)
        subplot.plot(xcoord,ycoord)
        plt.gca().invert_yaxis()
        
        for i in range(len(xt)):
            x = xt[i]
            y = yt[i]    
            plt.scatter(x,y,s=150)
            plt.annotate(i+1, (x,y))

        plt.xlim([self.xmin,self.xmax])
        plt.ylim([self.ymax,self.ymin])
        plt.xlabel("Horizontal")
        plt.ylabel("Depth")
        return fig
          
    def step(self, action):
        acc = (action - 1)/100 #Make acceleration input lay in range [-0.01, -0.01] -> [0.01, 0.01]
        done = False
        dist = np.linalg.norm([self.xdist1,self.ydist1]) #Distance to next target
        #Iterate (stepsize) steps with selected acceleration
        for _ in range(self.stepsize):
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
        
        #Calculate and update distance to target(s)
        self.xdist1 = self.targets[self.target_hits][0]-self.x  #x-axis distance to next target
        self.ydist1 = self.targets[self.target_hits][1]-self.y  #y-axis distance to next target

        #Update state vector
        self.state[0] = self.xdist1
        self.state[1] = self.ydist1
        self.state[2] = self.xd
        self.state[3] = self.yd
        
        #Check new target distance (reward)
        dist_new = np.linalg.norm([self.xdist1,self.ydist1])  
        dist_diff = dist_new - dist
        reward = -dist_diff

        #Check if outside grid (reward)
        if (self.x<self.xmin) or (self.y<self.ymin) or (self.x>self.xmax) or (self.y>self.ymax):
            reward -= 3000
            done = True

        #Check if inside target radius (reward)
        if dist_new < self.radius_target:
            reward += 3000
            self.target_hits += 1
            if self.target_hits == self.numtargets:
                done = True
        #Check if maximum travel range has been reached
        self.dist_traveled += self.stepsize
        if self.dist_traveled > self.max_dist:
            reward -= 3000
            done = True

        #Info for plotting and printing in run-file
        info = {'x':self.x, 'y':self.y, 'xtargets': [element[0] for element in self.targets],
                'ytargets': [element[1] for element in self.targets], 'hits': self.target_hits, 'tot_dist':self.dist_traveled, 'min_dist':self.min_tot_dist}

        return self.state, reward, done, info



    def init_states(self):
        #Set starting drill position and velocity
        self.dist_traveled = 0
        self.target_hits = 0
        self.x = random.randint(0,self.xmax)
        self.y = random.randint(0,500)
        self.xd = self.xd0
        self.yd = self.yd0
        #Initialize target(s)
        self.targets = []
        for _ in range(self.numtargets):
            self.targets.append((random.randint(200,self.xmax-200),random.randint(1000,self.ymax-200)))
        #Set distances to target
        self.xdist1 = self.x-self.targets[0][0]
        self.ydist1 = self.y-self.targets[0][1]

        #Calculate minimum and maximum total distance
        self.min_tot_dist = 0
        prev_p = np.array([self.x,self.y])
        for i in range(self.numtargets):
            self.min_tot_dist += np.linalg.norm([self.targets[i][0]-prev_p[0],self.targets[i][1]-prev_p[1]])
            prev_p = np.array(self.targets[i])
        

        self.max_dist = self.rel_max_dist*self.min_tot_dist

        self.state = np.array([
            self.xdist1,
            self.ydist1,

            self.xd,
            self.yd])
        return self.state
        
    def reset(self):
        self.init_states()
        return self.state


