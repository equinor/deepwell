import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import random


class DeepWellEnv(gym.Env):
    
    def __init__(self):
        super().__init__()
               
        self.stepsize = 10       #Number of timesteps between each decision
        self.xmin = 0
        self.xmax = 3000         #(>1000)
        self.ymin = 0
        self.ymax = 3000         #(>1000)
        self.zmin = 0
        self.zmax = 3000         #(>1000)

        self.xd0 = 0.0
        self.yd0 = 1.0
        self.zd0 = 0.0

        self.rel_max_dist = 3    #Set when to exit episode (dist_traveled > rel_max_dist*min_tot_dist = max_tot_dist)
        self.numtargets = 5     #==SET NUMBER OF TARGETS==#
        self.min_radius = 50
        self.max_radius = 50
       
        self.numhazards = 8     #==SET NUMBER OF HAZARDS==# 
        self.min_radius_hazard = 100
        self.max_radius_hazard = 100    

        self.state = self.init_states() #[xdist1, ydist1, zdist1, xdist2, ydist2, zdist2 xd, yd, zd, x_hz_dist, y_hz_dist, z_hz_dist]
        
        #Set action and observation space
        self.action_space = spaces.MultiDiscrete([3]*3)
        self.stateLow = np.array([ -self.xmax, -self.ymax, -self.zmax, -self.xmax, -self.ymax, -self.zmax -1., -1., -1., -self.xmax, -self.ymax])
        self.stateHigh = np.array([ self.xmax, self.ymax, self.zmax, self.xmax, self.ymax, self.zmax, 1., 1., 1., self.xmax, self.ymax, self.zmax])
        self.observation_space = spaces.Box(low=self.stateLow, high=self.stateHigh, dtype=np.float64)

        #Create figure to send to server
    def render(self, xcoord, ycoord, xt, yt, rt, xhz, yhz, rhz):
        fig, ax = plt.subplots()
        ax.plot(xcoord, ycoord)
        fig.gca().invert_yaxis()
        
        for i in range(len(xt)): 
            target = plt.Circle((xt[i], yt[i]), rt[i], color='g')
            ax.add_artist(target)
            ax.annotate(i+1, (xt[i], yt[i]))
        for i in range(len(xhz)):
            hazard = plt.Circle((xhz[i], yhz[i]), rhz[i], color='r')
            ax.add_artist(hazard)

        green_circle = Line2D([0], [0], marker='o', color='w', label='Target',
                        markerfacecolor='g', markersize=15)
        red_circle = Line2D([0], [0], marker='o', color='w', label='Hazard',
                        markerfacecolor='r', markersize=15)
        ax.legend(handles=[green_circle, red_circle])

        ax.set_xlim([self.xmin, self.xmax])
        ax.set_ylim([self.ymax, self.ymin])
        ax.set_xlabel("Horizontal")
        ax.set_ylabel("Depth")
        return fig
               
    def step(self, action):
        acc = (action - 1)/100 #Make acceleration input lay in range [-0.01, -0.01] -> [0.01, 0.01]
        done = False
        dist = np.linalg.norm([self.xdist1,self.ydist1,self.zdist1]) #Distance to next target
        #Iterate (stepsize) steps with selected acceleration
        for _ in range(self.stepsize):
            xd = acc[0] + self.xd           #update xd (unnormalized)
            yd = acc[1] + self.yd           #update yd (unnormalized)
            zd = acc[2] + self.zd           #update zd (unnormalized)
            velocity = np.linalg.norm([xd,yd,zd])
            if velocity == 0:
                velocity = 1
            normal_vel = np.array([xd, yd, zd])/velocity
            self.xd = normal_vel[0]         #update normalized vel. vector 
            self.yd = normal_vel[1]         #update normalized vel. vector
            self.zd = normal_vel[2]         #update normalized vel. vector 
            self.x = self.x + self.xd       #update x 
            self.y = self.y + self.yd       #update y
            self.z = self.z + self.zd       #update z
        
        #Calculate and update distance to target(s)
        self.xdist1 = self.targets[self.target_hits]['pos'][0]-self.x  #x-axis distance to next target
        self.ydist1 = self.targets[self.target_hits]['pos'][1]-self.y  #y-axis distance to next target
        self.zdist1 = self.targets[self.target_hits]['pos'][2]-self.z  #z-axis distance to next target
        if self.target_hits == self.numtargets - 1: #Only one target left
            self.xdist2 = self.xdist1
            self.ydist2 = self.ydist1
            self.zdist2 = self.zdist1
        else:
            self.xdist2 = self.targets[self.target_hits+1]['pos'][0]-self.x  #x-axis distance to second next target
            self.ydist2 = self.targets[self.target_hits+1]['pos'][1]-self.y  #y-axis distance to second next target
            self.zdist2 = self.targets[self.target_hits+1]['pos'][2]-self.z  #z-axis distance to second next target

        self.state[0] = self.xdist1
        self.state[1] = self.ydist1
        self.state[2] = self.zdist1
        self.state[3] = self.xdist2
        self.state[4] = self.ydist2
        self.state[5] = self.zdist2
        self.state[6] = self.xd
        self.state[7] = self.yd
        self.state[8] = self.zd
        
        #Check new target distance (reward)
        dist_new = np.linalg.norm([self.xdist1,self.ydist1, self.zdist1])  
        dist_diff = dist_new - dist
        reward = -dist_diff

        #Check new hazard distance (reward)
        if self.numhazards > 0:
            diff = [(np.array(hazard['pos'])-[self.x,self.y,self.z]) for hazard in self.hazards]
            diffnorms = [np.linalg.norm([element[0], element[1], element[2]]) for element in diff]
            closest_hz = np.argmin(diffnorms)
            dist_hazard = diffnorms[closest_hz]
            
            if dist_hazard < self.hazards[closest_hz]['radius']:
                reward -= 2000
                done = True
            
            if dist_hazard < self.hazards[closest_hz]['radius']*2:
                rel_safe_dist = (self.hazards[closest_hz]['radius']*2 - dist_hazard)/(self.hazards[closest_hz]['radius']) # 0 if dist_hazard = 2*radius_hazard, 1 if dist_hazard = radius_hazard
                reward -= 50*rel_safe_dist**2
            self.xdist_hazard = diff[closest_hz][0]
            self.ydist_hazard = diff[closest_hz][1]
            self.zdist_hazard = diff[closest_hz][2]
            self.state[9] = self.xdist_hazard
            self.state[10] = self.ydist_hazard
            self.state[11] = self.zdist_hazard
        #Check if outside grid (reward)
        if (self.x<self.xmin) or (self.y<self.ymin) or (self.z<self.zmin)or (self.x>self.xmax) or (self.y>self.ymax) or (self.z>self.zmax):
            reward -= 3000
            done = True

        #Check if maximum travel range has been reached
        self.dist_traveled += self.stepsize
        if self.dist_traveled > self.max_dist[self.target_hits]:
            reward -= 3000
            done = True

        #Check if inside target radius (reward)
        if dist_new < self.targets[self.target_hits]['radius']: #self.radius_target:
            reward += 3000
            self.target_hits += 1
           
            if self.target_hits == self.numtargets:
                done = True
            else:
                self.xdist1 = self.targets[self.target_hits]['pos'][0]-self.x  #x-axis distance to next target
                self.ydist1 = self.targets[self.target_hits]['pos'][1]-self.y  #y-axis distance to next target
                self.zdist1 = self.targets[self.target_hits]['pos'][2]-self.z  #z-axis distance to next target
                if self.target_hits != self.numtargets - 1:
                    self.xdist2 = self.targets[self.target_hits+1]['pos'][0]-self.x  #x-axis distance to next target
                    self.ydist2 = self.targets[self.target_hits+1]['pos'][1]-self.y  #y-axis distance to next target
                    self.zdist2 = self.targets[self.target_hits+1]['pos'][2]-self.z  #z-axis distance to next target
                else: 
                    self.xdist2 = self.xdist1
                    self.ydist2 = self.ydist1
                    self.zdist2 = self.zdist1
                

        #Info for plotting and printing in run-file
        info = {'x':self.x, 'y':self.y,
                'xtargets': [target['pos'][0] for target in self.targets],
                'ytargets': [target['pos'][1] for target in self.targets],
                'ztargets': [target['pos'][2] for target in self.targets]
                't_radius': [target['radius'] for target in self.targets],
                'hits': self.target_hits, 'tot_dist':self.dist_traveled, 
                'min_dist':self.min_tot_dist,
                'xhazards': [hazard['pos'][0] for hazard in self.hazards],
                'yhazards': [hazard['pos'][1] for hazard in self.hazards],
                'zhazards': [hazard['pos'][2] for hazard in self.hazards],
                'h_radius': [hazard['radius'] for hazard in self.hazards]}

        return self.state, reward, done, info


    def init_states(self):
        #Set starting drill position and velocity
        self.dist_traveled = 0
        self.target_hits = 0
        self.x = random.randint(0, 600)
        self.y = 0
        self.z = self.zmax/2 
        self.xd = self.xd0
        self.yd = self.yd0
        self.zd = self.zd0
        #Initialize target(s)
        self.targets = self.init_targets()
        self.hazards = self.init_hazards()
        self.xdist1 = self.x-self.targets[0]['pos'][0]
        self.ydist1 = self.y-self.targets[0]['pos'][1]
        self.zdist1 = self.z-self.targets[0]['pos'][2]

        if self.numtargets > 1:
            self.xdist2 = self.x-self.targets[1]['pos'][0]
            self.ydist2 = self.y-self.targets[1]['pos'][1]
            self.zdist2 = self.y-self.targets[1]['pos'][2]


        #Set distances to closest hazard
        if self.numhazards > 0:
            diff = [(np.array(hazard['pos'])-[self.x,self.y,self.z]) for hazard in self.hazards]
            diffnorms = [np.linalg.norm([element[0], element[1], element[2]]) for element in diff]
            closest_hz = np.argmin(diffnorms)
            self.xdist_hazard = diff[closest_hz][0]
            self.ydist_hazard = diff[closest_hz][1]
            self.zdist_hazard = diff[closest_hz][2]
        else:
            self.xdist_hazard = -self.xmax + 2*random.randint(0,1)*self.xmax
            self.ydist_hazard = -self.ymax + 2*random.randint(0,1)*self.ymax
            self.zdist_hazard = -self.zmax + 2*random.randint(0,1)*self.zmax

        #Calculate minimum and maximum total distance
        self.max_dist = []
        self.min_tot_dist = 0
        prev_p = np.array([self.x,self.y, self.z])
        for i in range(self.numtargets):
            self.min_tot_dist += np.linalg.norm([self.targets[i]['pos'][0]-prev_p[0],self.targets[i]['pos'][1]-prev_p[1], self.targets[i]['pos'][2]-prev_p[2]])
            prev_p = np.array(self.targets[i]['pos'])
            self.max_dist.append(self.rel_max_dist*self.min_tot_dist)
        
        self.max_tot_dist = self.rel_max_dist*self.min_tot_dist
        self.state = np.array([
            self.xdist1,
            self.ydist1,
            self.zdist1,
            self.xdist2,
            self.ydist2,
            self.zdist2
            self.xd,
            self.yd,
            self.zd,
            self.xdist_hazard,
            self.ydist_hazard,
            self.zdist_hazard])
        return self.state


    def init_targets(self):
        """
        Initiates targets that are drawn randomly from equally spaced bins in
        x-direction. Constraint applied to max change in y-direction. Radius
        randomly drawn between self.min_radius and self.max_radius.
        """
        # Separate targets in to equally spaced bins to avoid overlap
        xsep = (self.xmax - self.xmin - 2*200)/self.numtargets
        maxy_change = (self.ymax - 200 - 1000)/2

        targets = [None]*(self.numtargets)
        for i in range(self.numtargets):
            radius = random.randint(self.min_radius, self.max_radius)
            # x drawn randomnly within bin edges minus the radius on each side
            x = random.randint(200 + i*xsep + radius, 200 + (i+1)*xsep - radius)
            if i == 0:
                y = random.randint(1000, self.ymax - 200)
            else: 
                # y drawn randomly within its allowed values, with limit to ychange from previous target
                y = random.randint(np.clip(y-maxy_change, 1000, self.ymax-200),
                                     np.clip(y+maxy_change, 1000, self.ymax-200))
            
            targets[i] = ({'pos': np.array([x ,y]), 
                                 'radius': radius,
                                 'order':i})
        return targets


    def init_hazards(self):
        """
        Initiates hazards with random position and radii. Ensures that no hazards
        overlaps any targets. 
        """

        hazards = [None]*(self.numhazards)
        for i in range(self.numhazards):
            radius = random.randint(self.min_radius_hazard, self.max_radius_hazard)
            valid = False
            while valid == False:
                x = random.randint(0, self.xmax)
                y = random.randint(500, self.ymax)
                pos = np.array([x, y])

                # Check if hazard is overlapping with targets (*10% of sum of radii)
                for x in range(self.numtargets):
                    relpos = np.linalg.norm(pos - self.targets[x]['pos'])
                    if relpos > (self.targets[x]['radius'] + radius)*1.1:
                        valid = True
                    else: 
                        valid = False
                        break
            hazards[i] = ({'pos': pos, 'radius': radius})
        return hazards

        
    def reset(self):
        self.init_states()
        return self.state

