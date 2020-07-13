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
from custom_callback.evalcallback import EvalCallback2

from datetime import datetime


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
        mainpy_path, text_argument, num_argument, model_name = sys.argv[0], sys.argv[1], sys.argv[2], sys.argv[3]
        #mainpy_path is the path of main.py = '/app/main.py'
        #If no argument is given (deepwellstart.ps1 -r), text_argument = " "
        if model_name == " " : model_name = datetime.now().strftime('%d%m%y-%H%M')


        #Periodically evalute agent, save best model
        eval_callback = EvalCallback2(self.env, best_model_save_path='./model_logs/', 
                        log_path='./model_logs/', eval_freq=1000,
                        deterministic=True, render=False) 


        tensorboard_logs_path = "app/tensorboard_logs/"
        trained_models_path = "app/trained_models/"

        if text_argument == "train":
            # Use TRPO or PPO2
            # To train model run script with an argument train
            #model = TRPO(MlpPolicy, self.env, verbose=1, tensorboard_log='logs/')
            print("====================== NOW TRAINING MODEL ==========================")
            model = PPO2('MlpPolicy', self.env, verbose=1, tensorboard_log=tensorboard_logs_path)
            model.learn(total_timesteps = int(num_argument), tb_log_name="TB_"+datetime.now().strftime('%d%m%y-%H%M'))
            model.save(trained_models_path + model_name)
            return model
         
        elif text_argument == "retrain":
            # This is for retraining the model, for tensorboard integration load the tensorboard log from your trained model and create a new name in model.learn below.
            print("====================== NOW RETRAINING MODEL ==========================")
            model = PPO2.load(trained_models_path + model_name, tensorboard_log=tensorboard_logs_path)
            model.set_env(make_vec_env('DeepWellEnv-v0', n_envs=8))
            model.learn(total_timesteps=int(num_argument), callback=eval_callback, reset_num_timesteps=False, tb_log_name="TB_"+datetime.now().strftime('%d%m%y-%H%M'))      #Continue training
            model.save(trained_models_path + model_name)                                                                                                   #Save the retrained model
            return model
                
        elif text_argument == "load":
            # Load a saved model. Remove "/app/" if not running with docker
            print("====================== NOW LOADING MODEL ==========================")
            model_name = num_argument                                                                    #Use string instead of number
            model = PPO2.load(trained_models_path + model_name, tensorboard_log=tensorboard_logs_path)              
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
                self.rt = info['t_radius']
                self.xhz = info['xhazards']
                self.yhz = info['yhazards']
                self.rhz = info['h_radius']
                break
        print("Minimum total distance: ",info['min_dist'])
        print("Distance traveled: ",info['tot_dist'])    
        print("Target hits:     ", hits)
        self.env.close()
        return self.xcoord, self.ycoord, self.xt, self.yt, self.rt, self.xhz, self.yhz, self.rhz
