import gym
from gym_dw import envs
#import env.DeepWellEnv
from stable_baselines import TRPO
#from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2
import matplotlib
import matplotlib.pyplot as plt
import sys

from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.deepq.policies import MlpPolicy

from stable_baselines import HER, DQN, SAC, DDPG, TD3
from stable_baselines.her import GoalSelectionStrategy, HERGoalEnvWrapper

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

            mainpy_path, argument = sys.argv[0], sys.argv[1]        #sys.argv fetches arg when running "deepwellstart.ps1 -r arg". This is to make it possible to load,train or retrain the agent. 
            #mainpy_path is the path of main.py = '/app/main.py'
            #If no argument is given (deepwellstart.ps1 -r), argument = " "
        
            if (argument == "train"):
                ######## use TRPO or PPO2
                #model = TRPO(MlpPolicy, self.env, verbose=1, tensorboard_log='logs/')
                #To train model run script with an argument (doesn't matter what)
                #model = PPO2('MlpPolicy', self.env, verbose=1, tensorboard_log="logs/")
                #model.learn(total_timesteps = 250000)
                #model.save("trpo_dw_250k")
                '''
                model = DQN(MlpPolicy, self.env, verbose=1, tensorboard_log="logs/")
                model.learn(total_timesteps=11000)
                model.save("deepq")
                self.obs = self.env.reset()
                '''
                print("====================== NOW TRAINING MODEL ==========================")
                model_class = DQN
                goal_selection_strategy = 'final'

                model = HER('MlpPolicy', self.env, model_class, n_sampled_goal=4, goal_selection_strategy=goal_selection_strategy,verbose=1, tensorboard_log="logs/")
                model.learn(10000)


                #model.save("./her_bit_env")

                self.obs = self.env.reset()
                

                return model
                

            elif (argument == "load"):
                print("====================== NOW LOADING MODEL ==========================")
                #Else it will load a saved one
                #remove "/app/" if not running with docker
                #model = PPO2.load("/app/ppo2_shortpath", tensorboard_log="logs/")
                #model = DQN.load("/app/deepq")
                model = HER.load('/app/her_bit_env.zip', env=self.env)
                return model
            
            elif (argument == "retrain"):
                print("====================== NOW RETRAINING MODEL ==========================")

            else:
                print("====================== NO ARGUMENT (Just .\deepwellstart.ps1 -r) ==========================")

        

    #Test the trained model, run until done, return list of visited coords
    def test_model(self,model):
        while True:
            action, _states = model.predict(self.obs)
            self.obs, rewards, done, info = self.env.step(action)
            print(self.obs)
            print(info)
            print("reward: ",rewards)
            self.xcoord.append(self.obs["achieved_goal"][0])
            self.ycoord.append(self.obs["achieved_goal"][1])
            if done:
                self.xt = self.obs["achieved_goal"][0]
                self.yt = self.obs["achieved_goal"][1]
                break
        return self.xcoord, self.ycoord
