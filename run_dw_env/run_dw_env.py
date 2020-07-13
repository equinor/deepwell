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
from custom_callback.evalcallback import EvalCallback2

# Filter tensorflow version warnings
import os
# https://stackoverflow.com/questions/40426502/is-there-a-way-to-suppress-the-messages-tensorflow-prints/40426709
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import warnings
# https://stackoverflow.com/questions/15777951/how-to-suppress-pandas-future-warning
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)
import tensorflow as tf
tf.get_logger().setLevel('INFO')
tf.autograph.set_verbosity(0)
import logging
tf.get_logger().setLevel(logging.ERROR)


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
        self.xt = []
        self.yt = []
        self.xhz = []
        self.yhz = []

   
    #Get model either by training a new one or loading an old one
    def get_model(self):
        #sys.argv fetches arg when running "deepwellstart.ps1 -r arg". This is to make it possible to load,train or retrain the agent.
        mainpy_path, text_argument, num_argument = sys.argv[0], sys.argv[1], int(sys.argv[2])
        #mainpy_path is the path of main.py = '/app/main.py'
        #If no argument is given (deepwellstart.ps1 -r), argument = " "


        #Periodically evalute agent, save best model
        eval_callback = EvalCallback2(self.env, best_model_save_path='./model_logs/', 
                        log_path='./model_logs/', eval_freq=1000,
                        deterministic=True, render=False) 


        tensorboard_log = "app/tensorboard_logs/"

        if text_argument == "train":
            # Use TRPO or PPO2
            # To train model run script with an argument train
            #model = TRPO(MlpPolicy, self.env, verbose=1, tensorboard_log='logs/')
            print("====================== NOW TRAINING MODEL ==========================")
            model = PPO2('MlpPolicy', self.env, verbose=1, tensorboard_log=tensorboard_log)
            model.learn(total_timesteps = num_argument, tb_log_name='200k_new')
            model.save("app/trained_models/ppo2_200k_newenv")
            return model
         
        elif text_argument == "retrain":
            # This is for retraining the model, for tensorboard integration load the tensorboard log from your trained model and create a new name in model.learn below.
            print("====================== NOW RETRAINING MODEL ==========================")
            model = PPO2.load("/app/ppo2_200k+400k", tensorboard_log="logs/200k_new_1")
            model.set_env(make_vec_env('DeepWellEnv-v0', n_envs=8))
            model.learn(total_timesteps=num_argument, callback =eval_callback, reset_num_timesteps=False, tb_log_name='PPO2_400_5th')      #Continue training
            model.save("ppo2_200k+500k")                                                                                            #Save the retrained model
            return model
                
        elif text_argument == "load":
            # Load a saved model. Remove "/app/" if not running with docker
            print("====================== NOW LOADING MODEL ==========================")
            model = PPO2.load("/app/ppo2_200k_newenv", tensorboard_log="logs/200k_new_1")              
            model.save("ppo2_100k+240k")
            return model

        else:
            print("====================== NO ARGUMENT OR NO KNOWN ARGUMENT ENTERED ==========================")
            #Code here

        

    #Test the trained model, run until done, return list of visited coords
    def test_model(self,model):
        self.obs = self.env.reset()
        while True:
            action, _states = model.predict(self.obs)
            self.obs, rewards, done, info = self.env.step(action)

            """ print(self.obs)
            print(info)"""
            print("reward: ",rewards) 
            self.xcoord.append(info['x'])
            self.ycoord.append(info['y'])
            if done:
                hits = info['hits']
                self.xt = info['xtargets']
                self.yt = info['ytargets']
                self.xhz = info['xhazards']
                self.yhz = info['yhazards']
                break
        print("Minimum total distance: ",info['min_dist'])
        print("Distance traveled: ",info['tot_dist'])    
        print("Target hits:     ", hits)
        self.env.close()
        return self.xcoord, self.ycoord, self.xt, self.yt, self.xhz, self.yhz
