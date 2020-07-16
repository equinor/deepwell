import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
import random

MAX_ANGVEL = 0.1
MAX_ANGACC = 0.05

# The allowed increment. We either add or remove this value to the angular acceleration
ANGACC_INCREMENT = 0.01
STEP_LENGTH = 30.0


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
        self.min_radius = 50
        self.max_radius = 50
       
        self.numhazards = 8     #==SET NUMBER OF HAZARDS==# 
        self.min_radius_hazard = 100
        self.max_radius_hazard = 100    

        self.state = self.init_states() #[xdist1, ydist1, zdist1, xdist2, ydist2, zdist2 xd, yd, zd, x_hz_dist, y_hz_dist, z_hz_dist]
        
        #Set action and observation space
        incr = ANGACC_INCREMENT
        self.actions_dict = {
            0:[-incr, -incr],
            1:[-incr, 0],
            2:[-incr, incr],          # Actions: [0] = vertical acceleration,
            3:[0, -incr],          # [1] = horizontal acceleration 
            4:[0, 0],           # -1:Decreaase, 0:Maintain, 1:Increase
            5:[0, incr],
            6:[incr, -incr],
            7:[incr, 0],
            8:[incr, incr],}        
        self.action_space = spaces.Discrete(9)

        state_high = np.array([self.xmax, self.ymax, self.zmax,  # xdist1, ydist1, zdist1,
                                self.xmax, self.ymax, self.zmax,  # xdist2, ydist2, zdist2,
                                self.xmax, self.ymax, self.zmax,  # xdist_hazard, ydist_hazard, zdist_hazard,
                                2*np.pi, 2*np.pi,                       # vertical_ang, horizontal_ang,
                                MAX_ANGVEL, MAX_ANGVEL,           # vertical_ang_vel, horizontal_ang_vel,
                                MAX_ANGACC, MAX_ANGACC])          # vertical_ang_acc, horizontal_ang_acc
        state_low = -state_high.copy()
        state_low[9], state_low[10] = 0, 0
        self.observation_space = spaces.Box(low=-state_low, high=state_high, dtype=np.float64)


    def init_states(self):
        #Set starting drill position and velocity
        self.dist_traveled = 0
        self.target_hits = 0
        self.x = random.randint(0, 600)
        self.y = self.ymax/2
        self.z = 0 # Better to start at surface? self.zmax/2 

        # Spherical coordinates
        self.horizontal_ang = 0 # random.uniform(0, np.pi/2)  # change later?
        self.vertical_ang = 0    # starting vertically down      # uniform(np.pi/10,np.pi/2)

        self.horizontal_angVel = 0
        self.vertical_angVel = 0

        self.horizontal_angAcc = 0
        self.vertical_angAcc = 0


        #Initialize target(s)
        self.targets = self.init_targets()
        self.hazards = self.init_hazards()
        self.xdist1 = self.x - self.targets[0]['pos'][0]
        self.ydist1 = self.y - self.targets[0]['pos'][1]
        self.zdist1 = self.z - self.targets[0]['pos'][2]

        if self.numtargets > 1:
            self.xdist2 = self.x - self.targets[1]['pos'][0]
            self.ydist2 = self.y - self.targets[1]['pos'][1]
            self.zdist2 = self.y - self.targets[1]['pos'][2]


        #Set distances to closest hazard
        if self.numhazards > 0:
            diff = [(np.array(hazard['pos']) - [self.x, self.y, self.z]) for hazard in self.hazards]
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
            self.min_tot_dist += np.linalg.norm([self.targets[i]['pos'][0]
            -prev_p[0],self.targets[i]['pos'][1]-prev_p[1], self.targets[i]['pos'][2]-prev_p[2]])
            prev_p = np.array(self.targets[i]['pos'])
            self.max_dist.append(self.rel_max_dist*self.min_tot_dist)
        
        self.max_tot_dist = self.rel_max_dist*self.min_tot_dist

        state = self.get_state()
        return state
        

    def get_state(self):
        state = np.array([
            self.xdist1, self.ydist1, self.zdist1,
            self.xdist2, self.ydist2, self.zdist2,
            self.xdist_hazard, self.ydist_hazard, self.zdist_hazard,
            self.vertical_ang, self.horizontal_ang,
            self.vertical_angVel, self.horizontal_angVel,
            self.vertical_angAcc, self.horizontal_angAcc])
        return state

               
    def step(self, action):

        self.old_dist = np.linalg.norm([self.xdist1, self.ydist1, self.zdist1]) #Distance to next target
        
        self.update_pos(action)
        self.calc_dist_to_target()
        reward, done = self.get_reward()

        state = self.get_state()
        info = self.get_info(done)

        return state, reward, done, info


    def update_pos(self, action):
        
        # update angular acceleration
        if abs(self.vertical_angAcc + self.actions_dict[action][0]) < MAX_ANGACC:
            self.vertical_angAcc += self.actions_dict[action][0]
        
        if abs(self.horizontal_angAcc + self.actions_dict[action][1]) < MAX_ANGACC:
            self.horizontal_angAcc += self.actions_dict[action][1]
        

        # Update angular velocity
        if abs(self.vertical_angVel + self.vertical_angAcc) < MAX_ANGVEL:
            self.vertical_angVel += self.vertical_angAcc
        
        if abs(self.horizontal_angVel + self.horizontal_angAcc) < MAX_ANGVEL:
            self.horizontal_angVel += self.horizontal_angAcc
        

        # Update angle
        self.vertical_ang = (self.vertical_ang + self.vertical_angVel) % (2 * np.pi)
        self.horizontal_ang = (self.horizontal_ang + self.horizontal_angVel) % (2 * np.pi)

        #print('before', self.x, self.y, self.z)
        #print("Vertical ", self.vertical_ang, "horizontal ", self.horizontal_ang)
        # update position
        self.x += STEP_LENGTH * np.sin(self.vertical_ang) * np.cos(self.horizontal_ang)
        self.y += STEP_LENGTH * np.sin(self.vertical_ang) * np.sin(self.horizontal_ang)
        self.z += STEP_LENGTH * np.cos(self.vertical_ang)
        #print('after', self.x, self.y, self.z)


    def calc_dist_to_target(self):
        #Calculate and update distance to target(s)
        self.xdist1 = self.targets[self.target_hits]['pos'][0] - self.x  #x-axis distance to next target
        self.ydist1 = self.targets[self.target_hits]['pos'][1] - self.y  #y-axis distance to next target
        self.zdist1 = self.targets[self.target_hits]['pos'][2] - self.z  #z-axis distance to next target
        if self.target_hits == self.numtargets - 1: #Only one target left
            self.xdist2 = self.xdist1
            self.ydist2 = self.ydist1
            self.zdist2 = self.zdist1
        else:
            self.xdist2 = self.targets[self.target_hits+1]['pos'][0] - self.x  #x-axis distance to second next target
            self.ydist2 = self.targets[self.target_hits+1]['pos'][1] - self.y  #y-axis distance to second next target
            self.zdist2 = self.targets[self.target_hits+1]['pos'][2] - self.z  #z-axis distance to second next target


    def get_reward(self):
        done = False
        #Check new target distance (reward)
        dist_new = np.linalg.norm([self.xdist1,self.ydist1, self.zdist1])  
        dist_diff = dist_new - self.old_dist
        reward = -dist_diff

        #Action reward
        #if acc[0] == acc[1] ==acc[2] == 0:
        #    reward +=5
        #Check new hazard distance (reward)

        if self.vertical_angAcc > MAX_ANGACC:
            reward -= 500
        if self.horizontal_angAcc > MAX_ANGACC:
            reward -= 500
        
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
        maxz_change = (self.zmax - 200 - 1000)/2

        targets = [None]*(self.numtargets)
        for i in range(self.numtargets):
            radius = random.randint(self.min_radius, self.max_radius)
            # x drawn randomnly within bin edges minus the radius on each side
            x = random.randint(200 + i*xsep + radius, 200 + (i+1)*xsep - radius)
            y = random.randint(self.ymax/2-250, self.ymax/2+250)
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


    def render(self, xcoord, ycoord, zcoord, xt, yt, zt, rt, xhz, yhz, zhz, rhz):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(xcoord,ycoord,zcoord)
        fig.gca().invert_zaxis()
        
        for i in range(len(xt)):
            plot_ball(xt[i],yt[i],zt[i],rt[i],'g',ax)
            
        for i in range(len(xhz)):
            plot_ball(xhz[i],yhz[i],zhz[i],rhz[i],'r',ax)

        # Create circles for label
        green_circle = Line2D([0], [0], marker='o', color='w', label='Target',
                        markerfacecolor='g', markersize=15)
        red_circle = Line2D([0], [0], marker='o', color='w', label='Hazard',
                        markerfacecolor='r', markersize=15)
        ax.legend(handles=[green_circle, red_circle])

        ax.set_xlim([self.xmin, self.xmax])
        ax.set_ylim([self.ymin, self.ymax])
        ax.set_zlim([self.zmax, self.zmin])
        ax.set_xlabel("East")
        ax.set_ylabel("North")
        ax.set_zlabel("TVD")
        return fig

    def reset(self):
        self.init_states()
        return self.state


def plot_ball(x0, y0, z0, r, c, ax):
        # Make data
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = x0 + r * np.outer(np.cos(u), np.sin(v))
        y = y0 + r * np.outer(np.sin(u), np.sin(v))
        z = z0 + r * np.outer(np.ones(np.size(u)), np.cos(v))
        # Plot the surface
        ax.plot_surface(x, y, z, color=c)


if __name__ == '__main__' :
    env = DeepWellEnvSpher()
    env.reset()
    for _ in range(10):
        action = 7 #env.action_space.sample()
        print(env.actions_dict[action])
        print("Step: ", _ , " this is what the current state is:")
        print(env.step(action))