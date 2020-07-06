import gym
from gym_dw import envs
#import env.DeepWellEnv
from stable_baselines import TRPO
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2
from stable_baselines import HER, DQN, SAC, DDPG, TD3
from stable_baselines.her import GoalSelectionStrategy, HERGoalEnvWrapper
from stable_baselines.common.bit_flipping_env import BitFlippingEnv
import matplotlib
import matplotlib.pyplot as plt

class run_dw:
    def __init__(self):
        env = gym.make('DeepWellEnv-v0')
        ######## use TRPO or PPO2
        model = TRPO(MlpPolicy, env, verbose=1, gamma = 0.9)
        #model = PPO2(MlpPolicy, env, verbose=1)
        model.learn(total_timesteps=1000000)
        #model.save("trpo_dw")
        #model = TRPO.load("trpo_dw")

        self.xcoord = []
        self.ycoord = []
        self.obs = env.reset()
        self.xt = 0
        self.yt = 0

        while True:
            action, _states = model.predict(self.obs)
            self.obs, rewards, done, info = env.step(action)
            print(self.obs)
            print("reward: ",rewards)
            self.xcoord.append(info['x'])
            self.ycoord.append(info['y'])
            if done:
                self.xt = info['xt']
                self.yt = info['yt']
                break

    def get_plot(self):
        fig = plt.figure()
        subplot = fig.add_subplot(111)
        subplot.plot(self.xcoord,self.ycoord)
        plt.gca().invert_yaxis()
        subplot.scatter(self.xt,self.yt,s=150)
        plt.xlabel("Cross Section")
        plt.ylabel("TVD")
        return fig
