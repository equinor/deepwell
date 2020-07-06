import gym
from gym_dw import envs
#import env.DeepWellEnv
from stable_baselines import TRPO
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2
import matplotlib
import matplotlib.pyplot as plt
import sys

class run_dw:
    def __init__(self):
        self.env = gym.make('DeepWellEnv-v0')
        self.xcoord = []
        self.ycoord = []
        self.obs = self.env.reset()
        self.xt = 0
        self.yt = 0

        
    #Get model either by training a new one or loading an old one
    def get_model(self):
            if len(sys.argv)>1:
                ######## use TRPO or PPO2
                #model = TRPO(MlpPolicy, env, verbose=1, tensorboard_log='logs/')
                #To train model run script with an argument (doesn't matter what)
                model = PPO2('MlpPolicy', self.env, verbose=1, tensorboard_log="logs/")
                model.learn(total_timesteps = 100000)
                model.save("ppo2_shortpath")
                return model
            else:
                #Else it will load a saved one
                model = PPO2.load("ppo2_shortpath", tensorboard_log="logs/")
                return model

    #Test the trained model, run until done, return list of visited coords
    def test_model(self,model):
        while True:
            action, _states = model.predict(self.obs)
            self.obs, rewards, done, info = self.env.step(action)
            print(self.obs)
            print("reward: ",rewards)
            self.xcoord.append(info['x'])
            self.ycoord.append(info['y'])
            if done:
                self.xt = info['xt']
                self.yt = info['yt']
                break
        return self.xcoord, self.ycoord
