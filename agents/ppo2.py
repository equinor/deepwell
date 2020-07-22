import gym
from gym_dw import envs

from stable_baselines import TRPO
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2

import sys
from custom_callback.evalcallback import EvalCallback2

from datetime import datetime
from custom_policies.policies import ThreeOf128NonShared,  OneShared55TwoValueOnePolicy

import numpy as np

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


class ppo2:
    def __init__(self,env):
        self.env = env     #Options: 'DeepWellEnv-v0' 'DeepWellEnv-v2' 'DeepWellEnv3d-v0' 'DeepWellEnvSpher-v0'


    #Get model either by training a new one or loading an old one
    def get_model(self):
        #sys.argv fetches arg when running "deepwellstart.ps1 -r arg". This is to make it possible to load,train or retrain the agent.
        mainpy_path, text_argument, num_argument, model_name = sys.argv[0], sys.argv[1], sys.argv[2], sys.argv[3]
        #mainpy_path is the path of main.py = '/app/main.py'
        #If no argument is given (deepwellstart.ps1 -r), text_argument = " "
        if model_name == " " : model_name = datetime.now().strftime('%d%m%y-%H%M')


        #Periodically evalute agent, save best model
        eval_callback = EvalCallback2(self.env, best_model_save_path='app/model_logs/', 
                        log_path='app/model_logs/', eval_freq=1000,
                        deterministic=True, render=False) 

        #To use custom policy with different layer setup than MlpPolicy([64,64])
        custom_policy = ThreeOf128NonShared

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

        elif text_argument == "leveltrain":
            levels = 5
            # Initiate model
            print("====================== Initiating environment ==========================")
            self.env = gym.make('DeepWellEnvSpherlevel1-v0')
            model = PPO2('MlpPolicy', self.env, verbose=1, tensorboard_log=tensorboard_logs_path)
            model.save(trained_models_path + model_name)

            # Train model for increasingly difficult levels
            for i in range(1, levels+1):
                if i == 1:
                    timesteps = float(num_argument)/(8*levels)  # Divide by 8 as num env = 8
                else: 
                    timesteps = float(num_argument)/(levels)    # When loading it trains the inputed number of timesteps
                model = self.retrain('DeepWellEnvSpherlevel'+str(i)+'-v0',
                                    trained_models_path, model_name, tensorboard_logs_path, timesteps)
                print("====================== Level " + str(i) + "finished ==========================")
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


    def retrain(self, env, trained_models_path, model_name, tensorboard_logs_path, timesteps):
        # This is for retraining the model, for tensorboard integration load the tensorboard log from your trained model and create a new name in model.learn below.
        print("====================== NOW RETRAINING MODEL ==========================")
        self.env = gym.make(env) # For testing model after training
        model = PPO2.load(trained_models_path + model_name, tensorboard_log=tensorboard_logs_path)
        model.set_env(make_vec_env(env, n_envs=8))
        model.learn(total_timesteps=int(timesteps), tb_log_name="TB_"+datetime.now().strftime('%d%m%y-%H%M'))      #Continue training
        model.save(trained_models_path + model_name)                                                                                                   #Save the retrained model
        return model

    


