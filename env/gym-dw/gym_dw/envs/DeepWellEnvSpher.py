import gym
from gym import spaces
import numpy as np
import random

MAX_ANGVEL = 0.05
MAX_ANGACC = 0.01
ANGACC_INCREMENT = 0.005
STEP_LENGTH = 10.0

def calc_ang_diff(ang1, ang2):
    diff = ang2 - ang1
    if diff < -np.pi:
        diff += 2*np.pi
    if diff > np.pi:
        diff -= 2*np.pi
    return diff

def calc_rel_ang(vec, vertical_ang, horizontal_ang):
    """
    Calculates the relative angle between a vector and the vertical and horizontal angle
    """
    vertical_object_ang = np.arctan2(np.sqrt(vec[0]**2 + vec[1]**2), vec[2])
    horizontal_object_ang = np.arctan2(vec[1], vec[0])
    vertical_rel_ang = calc_ang_diff(vertical_ang, vertical_object_ang)
    horizontal_rel_ang = calc_ang_diff(horizontal_ang, horizontal_object_ang)
    return vertical_rel_ang, horizontal_rel_ang


class DeepWellEnvSpher(gym.Env):

    def __init__(self):
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
            0:[-incr, -incr], 1:[-incr, 0], 2:[-incr, incr],  # Actions:
            3:[0, -incr],     4:[0, 0],     5:[0, incr],      # [0] = vertical acceleration
            6:[incr, -incr],  7:[incr, 0],  8:[incr, incr],   # [1] = horizontal acceleration
            }        
        self.action_space = spaces.Discrete(9)
        max_dist = np.sqrt(self.xmax**2 + self.ymax**2 + self.zmax**2)
        state_high = np.array([
                               MAX_ANGACC, MAX_ANGACC,      # vertical_angAcc, horizontal_angAcc,
                               MAX_ANGVEL, MAX_ANGVEL,      # vertical_angVel, horizontal_angVel,
                               np.pi, np.pi,                # vert_targ_rel_ang1, hori_targ_rel_ang1, target_dist1
                               np.pi, np.pi,                # vert_targ_rel_ang2, hori_targ_rel_ang2, target_dist2
                               np.pi, np.pi,                # vert_haz_rel_ang, hori_haz_rel_ang,
                               max_dist, max_dist, max_dist # hazard_dist
                               ])
        state_low = -state_high.copy()
        state_low[10], state_low[11], state_low[12] = 0, 0, 0
        self.observation_space = spaces.Box(low=state_low, high=state_high, dtype=np.float64)

    def init_states(self):
        # Set starting drill position
        self.dist_traveled = 0
        self.target_hits = 0
        self.x = random.randint(0, 600)
        self.y = self.ymax/2
        self.z = 0 

        # Spherical coordinates
        self.horizontal_ang = random.uniform(0, 2*np.pi)
        self.vertical_ang = random.uniform(0, 5*np.pi/180)
        self.horizontal_angVel = 0
        self.vertical_angVel = 0
        self.horizontal_angAcc = 0
        self.vertical_angAcc = 0

        # Initialize targets and hazards
        self.targets = self.init_targets()
        self.hazards = self.init_hazards()
        self.target1, self.target2 = self.find_vector_to_targets()
        self.vert_targ_rel_ang1, self.hori_targ_rel_ang1 = calc_rel_ang(self.target1,
                                               self.vertical_ang, self.horizontal_ang)
        self.vert_targ_rel_ang2, self.hori_targ_rel_ang2 = calc_rel_ang(self.target2, 
                                               self.vertical_ang, self.horizontal_ang)

        # Set distances to closest hazard
        if self.numhazards > 0:
            closest_hz, self.hazard = self.find_closest_hazard()
            self.hazard_dist = np.linalg.norm(self.hazard)
        else:
            self.hazard = np.array([
                                    np.random.choice([-1, 1])*self.xmax,
                                    np.random.choice([-1, 1])*self.ymax,
                                    np.random.choice([-1, 1])*self.ymax
                                    ])
        self.vert_haz_rel_ang, self.hori_haz_rel_ang = calc_rel_ang(self.hazard,
                                    self.vertical_ang, self.horizontal_ang)
        self.max_dist, self.min_tot_dist = self.calc_max_tot_dist()

        state = self.get_state()
        return state
        
    def get_state(self):
        if self.target_hits < self.numtargets:
            self.target_dist1 = np.linalg.norm(self.target1) - self.targets[self.target_hits]['rad']
            if self.target_hits == self.numtargets - 1:
                self.target_dist2 = self.target_dist1
            else:
                self.target_dist2 = np.linalg.norm(self.target2) - self.targets[self.target_hits+1]['rad']
        hazard_dist = np.linalg.norm(self.hazard)

        state = np.array([
                self.vertical_angAcc, self.horizontal_angAcc,
                self.vertical_angVel, self.horizontal_angVel,
                self.vert_targ_rel_ang1, self.hori_targ_rel_ang1, 
                self.vert_targ_rel_ang2, self.hori_targ_rel_ang2,
                self.vert_haz_rel_ang, self.hori_haz_rel_ang,
                self.target_dist1, self.target_dist2, hazard_dist
                ])
        return state

    def get_pos(self):
        return np.array([self.x, self.y, self.z])

    def step(self, action):
        self.update_pos(action)
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
        self.dist_traveled += STEP_LENGTH

    def get_reward(self):
        done = False
        # Find difference in distance to target from last step
        old_dist = np.linalg.norm(self.target1) 
        self.target1, self.target2 = self.find_vector_to_targets()
        target_dist = np.linalg.norm(self.target1)  
        dist_diff = target_dist - old_dist

        # Find relative angles to the targets
        self.vert_targ_rel_ang1, self.hori_targ_rel_ang1 = calc_rel_ang(self.target1,
                                            self.vertical_ang, self.horizontal_ang)
        self.vert_targ_rel_ang2, self.hori_targ_rel_ang2 = calc_rel_ang(self.target2, 
                                            self.vertical_ang, self.horizontal_ang)

        reward = -3*dist_diff
        reward -= abs(self.vert_targ_rel_ang1) + abs(self.hori_targ_rel_ang1)
               
        if self.numhazards > 0:
            closest_hz, self.hazard = self.find_closest_hazard()
            self.hazard_dist = np.linalg.norm(self.hazard)
            hazard_radius = self.hazards[closest_hz]['rad']

            if self.hazard_dist < hazard_radius:
                reward -= 2000
                done = True
            if self.hazard_dist < 2*hazard_radius:
                rel_safe_dist = 2*(hazard_radius - self.hazard_dist)/(hazard_radius) # 0 if dist_hazard = 2*radius_hazard, 1 if dist_hazard = radius_hazard
                reward -= 50*rel_safe_dist**2
        
        if self.outside_bounds():
            reward -= 3000
            done = True

        #Check if maximum travel range has been reached
        if self.dist_traveled > self.max_dist[self.target_hits]:
            reward -= 3000
            done = True

        #Check if inside target radius (reward)
        if target_dist < self.targets[self.target_hits]['rad']:
            reward += 3000
            self.target_hits += 1

            if self.target_hits == self.numtargets:
                done = True
            else:
                self.target1, self.target2 = self.find_vector_to_targets()
                self.vert_targ_rel_ang1, self.hori_targ_rel_ang1 = calc_rel_ang(self.target1,
                                    self.vertical_ang, self.horizontal_ang)
                self.vert_targ_rel_ang2, self.hori_targ_rel_ang2 = calc_rel_ang(self.target2, 
                                    self.vertical_ang, self.horizontal_ang)
        return reward, done

    def find_vector_to_targets(self):
        """Calculates and update vector to target(s)"""
        targpos = self.targets[self.target_hits]['pos']
        target1 = targpos - self.get_pos()  # distance to next target                        

        if self.target_hits == self.numtargets - 1: #Only one target left
            target2 = target1
        else:
            targpos = self.targets[self.target_hits+1]['pos']
            target2 = targpos - self.get_pos() # distance to second next target
        return target1, target2

    def find_closest_hazard(self):
        rel_vectors = [(hazard['pos'] - self.get_pos()) for hazard in self.hazards]
        distances = [np.linalg.norm(vector) for vector in rel_vectors]
        closest_hz = np.argmin(distances)
        hazard = rel_vectors[closest_hz]
        return closest_hz, hazard

    def calc_max_tot_dist(self):
        max_dist = np.zeros(self.numtargets)
        prev_p = self.get_pos()
        min_tot_dist = 0
        for i in range(self.numtargets):
            min_tot_dist += np.linalg.norm(self.targets[i]['pos'] - prev_p)
            max_dist[i] = self.rel_max_dist*min_tot_dist 
            prev_p = self.targets[i]['pos']
        return max_dist, min_tot_dist

    def outside_bounds(self):
        x = (self.x < self.xmin) or (self.x > self.xmax)
        y = (self.y < self.ymin) or (self.y > self.ymax)
        z = (self.z < self.zmin) or (self.z > self.zmax)
        return x and y and z

    def get_info(self, done):
        #Info for plotting and printing in run-file
        if done == True:
            info = {
                'x': self.x, 'y': self.y, 'z': self.z,
                'xtargets': [target['pos'][0] for target in self.targets],
                'ytargets': [target['pos'][1] for target in self.targets],
                'ztargets': [target['pos'][2] for target in self.targets],
                't_radius': [target['rad'] for target in self.targets],
                'hits': self.target_hits, 'tot_dist':self.dist_traveled, 
                'min_dist':self.min_tot_dist,
                'xhazards': [hazard['pos'][0] for hazard in self.hazards],
                'yhazards': [hazard['pos'][1] for hazard in self.hazards],
                'zhazards': [hazard['pos'][2] for hazard in self.hazards],
                'h_radius': [hazard['rad'] for hazard in self.hazards]
                }
        else: 
            info = {'x': self.x, 'y': self.y, 'z': self.z}
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
                                 'rad': radius})
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
                    if relpos > (self.targets[j]['rad'] + radius)*1.1:
                        valid = True
                    else: 
                        valid = False
                        break
            hazards[i] = ({'pos': pos, 'rad': radius})
        return hazards
    
    def reset(self):
        self.init_states()
        return self.state


if __name__ == '__main__' :
    env = DeepWellEnvSpher()
    env.reset()
    for _ in range(10):
        action = 0 #env.action_space.sample()
        print(env.actions_dict[action])
        print("Step: ", _ , " this is what the current state is:")
        print(env.step(action))
