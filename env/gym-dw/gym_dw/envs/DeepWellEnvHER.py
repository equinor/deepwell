import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
import random


class DeepWellEnv(gym.GoalEnv):
    
    def __init__(self):
        super(DeepWellEnv,self).__init__()

        self.xmin = 0
        self.xmax = 3000
        self.ymin = 0
        self.ymax = 3000
        self.radius_target = 100
        self.stepsize = 10                   #Number of timesteps between each decision
        self.xtarget = None
        self.ytarget = None
        self.dist_old = None
        self.dist = None

        self.dist_traveled = 0
        self.max_dist = 1000000
    
        
        #Action space
        self.action_space = spaces.Discrete(9)
        self.action_dict = {
            0:np.array([0,0]),1:np.array([0,1]),2:np.array([0,2]),
            3:np.array([1,0]),4:np.array([1,1]),5:np.array([1,2]),
            6:np.array([2,0]),7:np.array([2,1]),8:np.array([2,2]),
        }
        #action_low = np.array([0,0])
        #action_high = np.array([2,2])

        #self.action_space = spaces.Box(low=action_low,high=action_high, dtype=np.float64)


        #Point bounds
        goal_low = np.array([self.xmin,self.ymin])
        goal_high = np.array([self.xmax,self.ymax])

        #Drilling state bounds:
        state_low = np.array([-3000, -3000, -1., -1.])
        state_high = np.array([3000, 3000, 1., 1.])

        self.observation_space = spaces.Dict({
            "achieved_goal":spaces.Box(low=goal_low,high=goal_high, dtype=np.float64),
            "desired_goal":spaces.Box(low=goal_low, high=goal_high, dtype=np.float64),
            "observation":spaces.Box(low=state_low, high=state_high, dtype=np.float64)
        })

        #State representation
        #{ "achieved_goal":np.array([x, y]), "desired_goal":np.array([x, y]), "observation":np.array([xdist, ydist, xd, yd]) }

        self.state = self.init_states()

    '''
    def update_xy(self,action,x,y,xd,yd,stepsize=1):    #Update x,y,xd,xy based on action and set stepsize
        #action = self.action_dict[action]
        acc = (action - 1)/100                                      #Make acceleration input lay in range [-0.1 -> 0.1]
        for _ in range(stepsize):                                   #Calculate next states
            xd = acc[0] + xd                                        #update xd (unnormalized)
            yd = acc[1] + yd                                        #update yd (unnormalized)
            velocity = np.linalg.norm([xd,yd])
            if velocity == 0:
                velocity = 1
            normal_vel = np.array([xd,yd])/velocity
            xd = normal_vel[0]                                      #update normalized vel. vector 
            yd = normal_vel[1]                                      #update normalized vel. vector 
            x = x + xd                                              #update x 
            y = y + yd                                              #update y
        return x,y,xd,yd
    '''

    '''
    def step(self, action):
        x_old = self.state["achieved_goal"][0]                              #Save current state
        y_old = self.state["achieved_goal"][1]
        xd_old = self.state["observation"][2]
        yd_old = self.state["observation"][3]

        self.dist_old = self.current_distance_to_target()                        #Calculate old distance to goal before updating

        x,y,xd,yd = self.update_xy(action,x_old,y_old,xd_old,yd_old)             #Update x,y,xd,xy based on action

        xdist = self.xtarget-x                                                   #Calculate new xdist,ydist for current distance to target
        ydist = self.ytarget-y

        self.state["achieved_goal"] = np.array([x,y])                            #Update to new state
        self.state["observation"] = np.array([xdist,ydist,xd,yd])
        self.dist = self.current_distance_to_target()                            #Store new dist after state update
    
        info = {'done':False,'is_success':False}                            #Save old distance in info for use in reward-function
        reward = self.compute_reward(self.state["achieved_goal"],self.state["desired_goal"],info)
        done = info['done']
        
        return self.state, reward, done, info
    '''

    def _set_action(self,action):
        action = self.action_dict[action]
        acc = (action - 1)/100                                      #Make acceleration input lay in range [-0.1 -> 0.1]
        x = self.state["achieved_goal"][0]                          #Get values from current state
        y = self.state["achieved_goal"][1]
        xd = self.state["observation"][2]
        yd = self.state["observation"][3]

        for _ in range(self.stepsize):                              #Calculate values for next state
            xd = acc[0] + xd                                        #update xd (unnormalized)
            yd = acc[1] + yd                                        #update yd (unnormalized)
            velocity = np.linalg.norm([xd,yd])
            if velocity == 0:
                velocity = 1
            normal_vel = np.array([xd,yd])/velocity
            xd = normal_vel[0]                                      #update normalized vel. vector 
            yd = normal_vel[1]                                      #update normalized vel. vector 
            x = x + xd                                              #update x 
            y = y + yd                                              #update y
        xdist = self.xtarget-x                                      #Calculate new xdist,ydist for current distance to target
        ydist = self.ytarget-y

        self.state["achieved_goal"] = np.array([x,y])               #Update current state
        self.state["observation"] = np.array([xdist,ydist,xd,yd])



    def step(self,action):
        self._set_action(action)
        obs = self._get_obs()
        done = False

        achieved_goal = obs['achieved_goal']
        desired_goal = obs['desired_goal']

        info = { 
            'is_success': self._is_success(achieved_goal, desired_goal),
        }

        self.dist_traveled += self.stepsize
        reward = self.compute_reward(achieved_goal, desired_goal, info)

        if (achieved_goal[0]<self.xmin) or (achieved_goal[1]<self.ymin) or (achieved_goal[0]>self.xmax) or (achieved_goal[1]>self.ymax):
            done = True

        if self.dist_traveled > self.max_dist:
            done = True

        if info['is_success']==1.0:
            done = True

        return obs, reward, done, info


    def _get_obs(self):
        return {
            'achieved_goal': self.state["achieved_goal"].copy(),
            'desired_goal': self.state["desired_goal"].copy(),
            'observation': self.state["observation"].copy(),
        }

    '''
    def current_distance_to_target(self):
        xdist = self.state["observation"][0]
        ydist = self.state["observation"][1]
        return np.linalg.norm([xdist,ydist])
    '''

    '''
    def compute_reward(self, achieved_goal, desired_goal, info):
        dist_diff = self.dist - self.dist_old
        reward = -dist_diff

        #Check if in radius of target (reward)
        if self.dist < self.radius_target:
            reward += 3000
            if not info is None:
                info['done'] = True
                info['is_success'] = True

        #Check if outside grid (reward)
        if (achieved_goal[0]<self.xmin) or (achieved_goal[1]<self.ymin) or (achieved_goal[0]>self.xmax) or (achieved_goal[1]>self.ymax):
            reward -= 3000
            if not info is None: info['done'] = True

        return reward
    '''

    def compute_reward(self, achieved_goal, goal, info):
        reward = 0

        if (achieved_goal[0]<self.xmin) or (achieved_goal[1]<self.ymin) or (achieved_goal[0]>self.xmax) or (achieved_goal[1]>self.ymax):
            reward -= 3000

        if self.dist_traveled > self.max_dist:
            reward -= 3000

        if self._is_success(achieved_goal, goal) == 1.0:
            reward += 3000

        d = self.goal_distance(achieved_goal, goal)

        return -d + reward

        #if self.reward_type == 'sparse':
        #    return -(d > self.distance_threshold).astype(np.float32)
        #else:
        #    return -d


    def init_states(self):
        self.xtarget = random.randint(0,3000)
        self.ytarget = random.randint(1000,3000)
        x = 500
        y = 250

        xdist = x-self.xtarget
        ydist = y-self.ytarget
        xd0 = 0.0
        yd0 = 1.0

        self.state = {}
        self.state["achieved_goal"] = np.array([x,y])
        self.state["desired_goal"] = np.array([self.xtarget,self.ytarget])
        self.state["observation"] = np.array([xdist,ydist,xd0,yd0])
        return self.state
        

    def reset(self):
        self.init_states()
        return self.state


    #Create figure to send to server
    def render(self, xcoord, ycoord):
        fig = plt.figure()
        subplot = fig.add_subplot(111)
        subplot.plot(xcoord,ycoord)
        subplot.scatter(self.xtarget,self.ytarget,s=150)
        plt.xlim([self.xmin,self.xmax])
        plt.ylim([self.ymin,self.ymax])
        plt.xlabel("Horizontal")
        plt.ylabel("Depth")
        return fig

    def goal_distance(self,goal_a, goal_b):
        assert goal_a.shape == goal_b.shape
        return np.linalg.norm(goal_a - goal_b, axis=-1)
    
    def _is_success(self, achieved_goal, desired_goal):
        d = self.goal_distance(achieved_goal, desired_goal)
        return (d < self.radius_target).astype(np.float32)