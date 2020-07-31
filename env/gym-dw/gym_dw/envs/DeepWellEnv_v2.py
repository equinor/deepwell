import gym
from gym import spaces
import numpy as np
import random

class DeepWellEnvV2(gym.Env):
    
    def __init__(self):
        super().__init__()
               
        self.stepsize = 10       #Number of timesteps between each decision
        self.xmin = 0
        self.xmax = 3000         #(>1000)
        self.ymin = 0
        self.ymax = 3000         #(>1000)
        
        self.rel_max_dist = 3    #Set when to exit episode (dist_traveled > rel_max_dist*min_tot_dist = max_tot_dist)
        self.numtargets = 5     #==SET NUMBER OF TARGETS==#
        self.min_radius = 50
        self.max_radius = 50
        self.target_hits = 0        
        
        self.numhazards = 8     #==SET NUMBER OF HAZARDS==# 
        self.min_radius_hazard = 100
        self.max_radius_hazard = 100    

        self.state = self.init_states() #[xdist1, ydist1, xdist2, ydist2, xd, yd, x_hz_dist, y_hz_dist]
        
        #Set action and observation space
        self.action_space = spaces.MultiDiscrete([3]*2)
        state_high = np.array([self.xmax, self.ymax, self.xmax, self.ymax,
                               1., 1.,self.xmax, self.ymax])
        state_low = - state_high.copy()
        self.observation_space = spaces.Box(low=state_low, high=state_low, dtype=np.float64)

    def init_states(self):
        #Set starting drill position and velocity
        self.dist_traveled = 0
        self.target_hits = 0
        self.x = random.randint(0, 600)
        self.y = 0
        self.xd = 0
        self.yd = 1
        #Initialize target(s)
        self.targets = self.init_targets()
        self.hazards = self.init_hazards()
        self.xdist1 = self.x-self.targets[0]['pos'][0]
        self.ydist1 = self.y-self.targets[0]['pos'][1]

        if self.numtargets > 1:
            self.xdist2 = self.x-self.targets[1]['pos'][0]
            self.ydist2 = self.y-self.targets[1]['pos'][1]

        #Set distances to closest hazard
        if self.numhazards > 0:
            diff = [(np.array(hazard['pos'])-[self.x,self.y]) for hazard in self.hazards]
            diffnorms = [np.linalg.norm([element[0], element[1]]) for element in diff]
            closest_hz = np.argmin(diffnorms)
            self.xdist_hazard = diff[closest_hz][0]
            self.ydist_hazard = diff[closest_hz][1]
        else:
            self.xdist_hazard = -self.xmax + 2*random.randint(0,1)*self.xmax
            self.ydist_hazard = -self.ymax + 2*random.randint(0,1)*self.ymax

        #Calculate minimum and maximum total distance
        self.max_dist, self.min_tot_dist = self.calc_max_tot_dist()
        self.max_tot_dist = self.rel_max_dist*self.min_tot_dist

        self.state = np.array([
            self.xdist1,
            self.ydist1,
            self.xdist2,
            self.ydist2,
            self.xd,
            self.yd,
            self.xdist_hazard,
            self.ydist_hazard])
        return self.state
               
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
        self.xdist1 = self.targets[self.target_hits]['pos'][0] - self.x  #x-axis distance to next target
        self.ydist1 = self.targets[self.target_hits]['pos'][1] - self.y  #y-axis distance to next target
        if self.target_hits == self.numtargets - 1: #Only one target left
            self.xdist2 = self.xdist1
            self.ydist2 = self.ydist1
        else:
            self.xdist2 = self.targets[self.target_hits+1]['pos'][0] - self.x  #x-axis distance to second next target
            self.ydist2 = self.targets[self.target_hits+1]['pos'][1] - self.y  #y-axis distance to second next target

        self.state[0] = self.xdist1
        self.state[1] = self.ydist1
        self.state[2] = self.xdist2
        self.state[3] = self.ydist2
        self.state[4] = self.xd
        self.state[5] = self.yd
        
        #Check new target distance (reward)
        dist_new = np.linalg.norm([self.xdist1, self.ydist1])  
        dist_diff = dist_new - dist
        reward = -dist_diff

        #Check new hazard distance (reward)
        if self.numhazards > 0:
            diff = [(np.array(hazard['pos'])-[self.x, self.y]) for hazard in self.hazards]
            diffnorms = [np.linalg.norm([element[0], element[1]]) for element in diff]
            closest_hz = np.argmin(diffnorms)
            dist_hazard = diffnorms[closest_hz]
            haz_rad = self.hazards[closest_hz]['radius']
            
            if dist_hazard < haz_rad:
                reward -= 2000
                done = True
            
            if dist_hazard < 2*haz_rad:
                rel_safe_dist = (2*haz_rad - dist_hazard)/(haz_rad) # 0 if dist_hazard = 2*radius_hazard, 1 if dist_hazard = radius_hazard
                reward -= 50*rel_safe_dist**2
            self.xdist_hazard = diff[closest_hz][0]
            self.ydist_hazard = diff[closest_hz][1]
            self.state[6] = self.xdist_hazard
            self.state[7] = self.ydist_hazard

        #Check if outside grid (reward)
        if self.outside_bounds():
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
                self.xdist1 = self.targets[self.target_hits]['pos'][0] - self.x  #x-axis distance to next target
                self.ydist1 = self.targets[self.target_hits]['pos'][1] - self.y  #y-axis distance to next target
                if self.target_hits != self.numtargets - 1:
                    self.xdist2 = self.targets[self.target_hits+1]['pos'][0] - self.x  #x-axis distance to next target
                    self.ydist2 = self.targets[self.target_hits+1]['pos'][1] - self.y  #y-axis distance to next target
                else: 
                    self.xdist2 = self.xdist1
                    self.ydist2 = self.ydist1

        #Info for plotting and printing in run-file
        info = self.get_info(done)

        return self.state, reward, done, info

    def get_info(self, done):
        #Info for plotting and printing in run-file
        if done == True:
            info = {
                'pos': np.array([self.x, self.y]),
                'targets': self.targets,
                'hazards': self.hazards,
                'hits': self.target_hits, 
                'min_dist': self.min_tot_dist,
                'tot_dist':self.dist_traveled
                }
        else:
            info = {'pos': np.array([self.x, self.y])}
        return info

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
            
            targets[i] = ({'pos': np.array([x ,y]), 'radius': radius})
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

    def outside_bounds(self):
        x = (self.x < self.xmin) or (self.x > self.xmax)
        y = (self.y < self.ymin) or (self.y > self.ymax)
        return x or y

    def calc_max_tot_dist(self):
        max_dist = np.zeros(self.numtargets)
        prev_p = np.array([self.x,self.y])
        min_tot_dist = 0
        for i in range(self.numtargets):
            min_tot_dist += np.linalg.norm(self.targets[i]['pos'] - prev_p)
            max_dist[i] = self.rel_max_dist*min_tot_dist 
            prev_p = self.targets[i]['pos']
        return max_dist, min_tot_dist

if __name__ == '__main__' :
    env = DeepWellEnvV2()
    env.reset()
    for _ in range(10):
        action = env.action_space.sample()
        print("Step: ", _ , " this is what the current state is:")
        print(env.step(action))
