import gym
from gym import spaces
import numpy as np
import random

MAX_ANGVEL = 0.05
MAX_ANGACC = 0.01

# The allowed increment. We either add or remove this value to the angular acceleration
ANGACC_INCREMENT = 0.005
STEP_LENGTH = 10.0


class DeepWellEnvSpher(gym.Env):
    
    def __init__(self):
        super().__init__()
               
        self.stepsize = 10       #Number of timesteps between each decision
        self.xmin = 0
        self.xmax = 3000         #(>1000)
        self.ymin = 0
        self.ymax = 3000         #(>1000)
        self.zmin = 0
        self.zmax = 3000         #(>1000)

        self.rel_max_dist = 3    #Set when to exit episode (dist_traveled > rel_max_dist*min_tot_dist = max_tot_dist)
        self.numtargets = 5     #==SET NUMBER OF TARGETS==#
        self.min_radius = 100
        self.max_radius = 100
       
        self.numhazards = 2     #==SET NUMBER OF HAZARDS==# 
        self.min_radius_hazard = 100
        self.max_radius_hazard = 100

        self.deltaY_target = 250
        self.maxdeltaZ_target = 500

        self.state = self.init_states()
        
        #Set action and observation space
        incr = ANGACC_INCREMENT
        self.actions_dict = {
            0:[-incr, -incr],
            1:[-incr, 0],
            2:[-incr, incr],       # Actions: [0] = vertical acceleration,
            3:[0, -incr],          # [1] = horizontal acceleration 
            4:[0, 0],              # -1:Decreaase, 0:Maintain, 1:Increase
            5:[0, incr],
            6:[incr, -incr],
            7:[incr, 0],
            8:[incr, incr],}        
        self.action_space = spaces.Discrete(9)


        state_high = np.array([MAX_ANGACC, MAX_ANGACC,        # vertical_ang_acc, horizontal_ang_acc,
                               MAX_ANGVEL, MAX_ANGVEL,            # vertical_ang_vel, horizontal_ang_vel,
                               np.pi, np.pi, np.sqrt(self.xmax**2 + self.ymax**2 + self.zmax**2),       #rel_vertical_target_ang1, rel_horizontal_target_ang1, target_dist1
                               np.pi, np.pi, np.sqrt(self.xmax**2 + self.ymax**2 + self.zmax**2),       #rel_vertical_target_ang2, rel_horizontal_target_ang2, target_dist2
                               np.pi, np.pi, np.sqrt(self.xmax**2 + self.ymax**2 + self.zmax**2)])       #rel_vertical_hazard_ang, rel_horizontal_hazard_ang, hazard_dist
        state_low = -state_high.copy()
        state_low[6], state_low[9], state_low[12] = 0, 0, 0
        self.observation_space = spaces.Box(low=state_low, high=state_high, dtype=np.float64)


    def init_states(self):
        #Set starting drill position
        self.dist_traveled = 0
        self.target_hits = 0
        self.x = random.randint(0, 600)
        self.y = self.ymax/2
        self.z = 0 

        # Spherical coordinates
        self.horizontal_ang = random.uniform(0, 2*np.pi)  # random.uniform(np.pi/10, np.pi/2-np.pi/10)  # change later?
        self.vertical_ang = random.uniform(0, 5*np.pi/180)

        self.horizontal_angVel = 0
        self.vertical_angVel = 0

        self.horizontal_angAcc = 0
        self.vertical_angAcc = 0

        #Initialize targets and hazards
        self.targets = self.init_targets()
        self.hazards = self.init_hazards()
        self.dist1, self.dist2 = self.calc_dist_to_target()
        self.calc_target_ang()
        self.calc_rel_target_ang()
        
        #Set distances to closest hazard
        if self.numhazards > 0:
            _ = self.calc_hazard_distances()
            self.hazard_dist = np.linalg.norm([self.xdist_hazard, self.ydist_hazard, self.zdist_hazard])
        else:
            self.xdist_hazard = -self.xmax + 2*random.randint(0,1)*self.xmax
            self.ydist_hazard = -self.ymax + 2*random.randint(0,1)*self.ymax
            self.zdist_hazard = -self.zmax + 2*random.randint(0,1)*self.zmax

        self.calc_hazard_ang()
        self.calc_rel_hazard_ang()  
        self.max_tot_dist = self.calc_max_tot_dist()

        state = self.get_state()
        return state
        

    def get_state(self):
        dist1, dist2 = self.calc_dist_to_target()
        target_dist1 = np.linalg.norm(dist1)
        target_dist2 = np.linalg.norm(dist2)
        state = np.array([
            self.vertical_angAcc, self.horizontal_angAcc,
            self.vertical_angVel, self.horizontal_angVel,
            self.rel_vertical_target_ang1, self.rel_horizontal_target_ang1, target_dist1,
            self.rel_vertical_target_ang2, self.rel_horizontal_target_ang2 , target_dist2,
            self.rel_vertical_hazard_ang, self.rel_horizontal_hazard_ang, self.hazard_dist
            ])
        return state

               
    def step(self, action):

        self.old_dist = np.linalg.norm([self.dist1]) #Distance to next target 
        self.update_pos(action)
        self.dist1, self.dist2 = self.calc_dist_to_target()        

        self.calc_target_ang()
        self.calc_rel_target_ang()
        self.calc_hazard_ang()
        self.calc_rel_hazard_ang()

        reward, done = self.get_reward()
        state = self.get_state()
        info = self.get_info(done)

        return state, reward, done, info


    def update_pos(self, action):
        acc = self.actions_dict[action]
        
        if abs(self.vertical_angAcc + acc[0]) <= MAX_ANGACC:
            self.vertical_angAcc += acc[0]
        
        if abs(self.horizontal_angAcc + acc[1]) <= MAX_ANGACC:
            self.horizontal_angAcc += acc[1]
        
        # Update angular velocity
        if abs(self.vertical_angVel + self.vertical_angAcc) <= MAX_ANGVEL:
            self.vertical_angVel += self.vertical_angAcc
        
        if abs(self.horizontal_angVel + self.horizontal_angAcc) <= MAX_ANGVEL:
            self.horizontal_angVel += self.horizontal_angAcc
        
        # Update angle
        self.vertical_ang = (self.vertical_ang + self.vertical_angVel)
        if self.vertical_ang < 0:
            self.vertical_ang = 0
            self.vertical_angVel = 0
        if self.vertical_ang > np.pi:
            self.vertical_ang = np.pi
            self.vertical_angVel = 0

        self.horizontal_ang = (self.horizontal_ang + self.horizontal_angVel) % (2*np.pi)

        # update position
        self.x += STEP_LENGTH * np.sin(self.vertical_ang) * np.cos(self.horizontal_ang)
        self.y += STEP_LENGTH * np.sin(self.vertical_ang) * np.sin(self.horizontal_ang)
        self.z += STEP_LENGTH * np.cos(self.vertical_ang)


    def get_pos(self):
        return np.array([self.x, self.y, self.z])


    def calc_dist_to_target(self):
        """Calculates and update distance to target(s)"""

        targpos, rad = map(self.targets[self.target_hits].get, ['pos', 'radius'])
        dist1 = targpos - self.get_pos() - rad  # distance to next target                        

        if self.target_hits == self.numtargets - 1: #Only one target left
            dist2 = dist1
        else:
            targpos, rad = map(self.targets[self.target_hits+1].get, ['pos', 'radius'])
            dist2 = targpos - self.get_pos() - rad # distance to next target
        
        return dist1, dist2


    def calc_target_ang(self):
        self.vertical_target_ang1 = np.arctan2(np.sqrt(self.dist1[0]**2 + self.dist1[1]**2), self.dist1[2])
        self.horizontal_target_ang1 = np.arctan2(self.dist1[1], self.dist1[0])
        self.vertical_target_ang2 = np.arctan2(np.sqrt(self.dist2[0]**2 + self.dist2[1]**2), self.dist2[2])
        self.horizontal_target_ang2 = np.arctan2(self.dist2[1], self.dist2[0])

    def calc_rel_target_ang(self):
        self.rel_vertical_target_ang1 = self.calc_angle_diff(self.vertical_ang, self.vertical_target_ang1)
        self.rel_horizontal_target_ang1 = self.calc_angle_diff(self.horizontal_ang, self.horizontal_target_ang1)
        self.rel_vertical_target_ang2 = self.calc_angle_diff(self.vertical_ang, self.vertical_target_ang2)
        self.rel_horizontal_target_ang2 = self.calc_angle_diff(self.horizontal_ang, self.horizontal_target_ang2)
    
    def calc_hazard_ang(self):
        self.vertical_hazard_ang = np.arctan2(np.sqrt(self.xdist_hazard**2 + self.ydist_hazard**2),self.zdist_hazard)
        self.horizontal_hazard_ang = np.arctan2(self.ydist_hazard,self.xdist_hazard)

    def calc_rel_hazard_ang(self):
        self.rel_vertical_hazard_ang = self.calc_angle_diff(self.vertical_ang, self.vertical_hazard_ang)
        self.rel_horizontal_hazard_ang = self.calc_angle_diff(self.horizontal_ang, self.horizontal_hazard_ang)

    def calc_hazard_distances(self):
        diff = [(np.array(hazard['pos']) - self.get_pos()) for hazard in self.hazards]
        diffnorms = [np.linalg.norm([element[0], element[1], element[2]]) for element in diff]
        closest_hz = np.argmin(diffnorms)
        self.xdist_hazard = diff[closest_hz][0]
        self.ydist_hazard = diff[closest_hz][1]
        self.zdist_hazard = diff[closest_hz][2]
        return closest_hz

    def calc_max_tot_dist(self):
        self.max_dist = []
        self.min_tot_dist = 0
        prev_p = self.get_pos()
        for i in range(self.numtargets):
            self.min_tot_dist += np.linalg.norm([self.targets[i]['pos'][0]
            -prev_p[0],self.targets[i]['pos'][1]-prev_p[1], self.targets[i]['pos'][2]-prev_p[2]])
            prev_p = np.array(self.targets[i]['pos'])
            self.max_dist.append(self.rel_max_dist*self.min_tot_dist)      
        max_tot_dist = self.rel_max_dist*self.min_tot_dist
        return max_tot_dist
        
    def get_reward(self):
        done = False
        #Check new target distance (reward)
        dist_new = np.linalg.norm(self.dist1)  
        dist_diff = dist_new - self.old_dist
        reward = -3*dist_diff

        reward -= abs(self.rel_vertical_target_ang1) + abs(self.rel_horizontal_target_ang1)
               
        if self.numhazards > 0:
            closest_hz = self.calc_hazard_distances()
            self.hazard_dist = np.linalg.norm([self.xdist_hazard, self.ydist_hazard, self.zdist_hazard])

            if self.hazard_dist < self.hazards[closest_hz]['radius']:
                reward -= 2000
                done = True
            
            if self.hazard_dist < 2*self.hazards[closest_hz]['radius']:
                rel_safe_dist = 2*(self.hazards[closest_hz]['radius'] - self.hazard_dist)/(self.hazards[closest_hz]['radius']) # 0 if dist_hazard = 2*radius_hazard, 1 if dist_hazard = radius_hazard
                reward -= 50*rel_safe_dist**2


        #Check if outside grid (reward)
        if (self.x<self.xmin) or (self.y<self.ymin) or (self.z<self.zmin)or (self.x>self.xmax) or (self.y>self.ymax) or (self.z>self.zmax):
            reward -= 3000
            done = True

        #Check if maximum travel range has been reached
        self.dist_traveled += STEP_LENGTH
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
                self.dist1, self.dist2 = self.calc_dist_to_target()
                self.calc_rel_target_ang()
                self.calc_rel_target_ang()

        return reward, done

    def get_info(self, done):
        #Info for plotting and printing in run-file
        if done == True:
            info = {'x':self.x, 'y':self.y, 'z':self.z,
                'xtargets': [target['pos'][0] for target in self.targets],
                'ytargets': [target['pos'][1] for target in self.targets],
                'ztargets': [target['pos'][2] for target in self.targets],
                't_radius': [target['radius'] for target in self.targets],
                'hits': self.target_hits, 'tot_dist':self.dist_traveled, 
                'min_dist':self.min_tot_dist,
                'xhazards': [hazard['pos'][0] for hazard in self.hazards],
                'yhazards': [hazard['pos'][1] for hazard in self.hazards],
                'zhazards': [hazard['pos'][2] for hazard in self.hazards],
                'h_radius': [hazard['radius'] for hazard in self.hazards]}
        else: 
            info = {'x':self.x, 'y':self.y, 'z':self.z}
        return info


    def init_targets(self):
        """
        Initiates targets that are drawn randomly from equally spaced bins in
        x-direction. Constraint applied to max change in y-direction. Radius
        randomly drawn between self.min_radius and self.max_radius.
        """
        # Separate targets in to equally spaced bins to avoid overlap
        xsep = (self.xmax - self.xmin - 2*200)/self.numtargets
        maxz_change = self.maxdeltaZ_target  # (self.zmax - 200 - 1000)/2
        deltaY = self.deltaY_target

        targets = [None]*(self.numtargets)
        for i in range(self.numtargets):
            radius = random.randint(self.min_radius, self.max_radius)
            # x drawn randomnly within bin edges minus the radius on each side
            x = random.randint(200 + i*xsep + radius, 200 + (i+1)*xsep - radius)
            y = random.randint(self.ymax/2 - deltaY, self.ymax/2 + deltaY)
            if i == 0:
                z = random.randint(1000, self.zmax - 200)
            else: 
                # y drawn randomly within its allowed values, with limit to ychange from previous target
                z = random.randint(np.clip(z-maxz_change, 1000, self.zmax-200),
                                     np.clip(z+maxz_change, 1000, self.zmax-200))
            
            targets[i] = ({'pos': np.array([x, y, z]), 
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
                y = random.randint(self.ymax/2-250, self.ymax/2+250)  # Refine this choice later
                z = random.randint(500, self.zmax)
                pos = np.array([x, y, z])

                # Check if hazard is overlapping with targets (*10% of sum of radii)
                for j in range(self.numtargets):
                    relpos = np.linalg.norm(pos - self.targets[j]['pos'])
                    if relpos > (self.targets[j]['radius'] + radius)*1.1:
                        valid = True
                    else: 
                        valid = False
                        break
            hazards[i] = ({'pos': pos, 'radius': radius})
        return hazards
    
    def reset(self):
        self.init_states()
        return self.state


    def calc_angle_diff(self, ang1, ang2):
        diff = ang2 - ang1
        if diff < -np.pi:
            diff = diff + 2*np.pi
        if diff > np.pi:
            diff = diff - 2*np.pi
        return diff


if __name__ == '__main__' :
    env = DeepWellEnvSpher()
    env.reset()
    for _ in range(10):
        action = 0 #env.action_space.sample()
        print(env.actions_dict[action])
        print("Step: ", _ , " this is what the current state is:")
        print(env.step(action))