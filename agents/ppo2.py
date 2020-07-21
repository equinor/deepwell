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
import plotly.graph_objects as go # or plotly.express as px

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
    def __init__(self):
        self.env = gym.make('DeepWellEnvSpher-v0')     #Options: 'DeepWellEnv-v0' 'DeepWellEnv-v2' 'DeepWellEnv3d-v0' 'DeepWellEnvSpher-v0'
   
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



    def get_fig(self,model):
        xcoord_list, ycoord_list, zcoord_list, info = self.get_path_from_model(model)
        
        fig = go.Figure(data=[go.Scatter3d(x=xcoord_list, y=ycoord_list, z=zcoord_list, mode='lines', name="Well path", line=dict(width=10.0))])


        x_targets = info['xtargets']
        y_targets = info['ytargets']
        z_targets = info['ztargets']
        radius_targets = info['t_radius']

        x_hazards = info['xhazards']
        y_hazards = info['yhazards']
        z_hazards = info['zhazards']
        radius_hazards = info['h_radius']



        for i in range(len(x_targets)):
            self.plot_ball(fig, "Target", 'greens', x_targets[i], y_targets[i], z_targets[i], radius_targets[i])
        
        for i in range(len(x_hazards)):
            self.plot_ball(fig, "Hazard", 'reds', x_hazards[i], y_hazards[i], z_hazards[i], radius_hazards[i])


        fig.update_layout(
            scene = dict(
                xaxis = dict(nticks=4, range=[self.env.xmin,self.env.xmax],title_text="East",),
                yaxis = dict(nticks=4, range=[self.env.ymin,self.env.ymax],title_text="North",),
                zaxis = dict(nticks=4, range=[self.env.zmax,self.env.zmin],title_text="TVD",),
            ),
        )

        print("Minimum total distance: ", info['min_dist'])
        print("Distance traveled: ", info['tot_dist'])    
        print("Target hits:     ", info['hits'])
        return fig



    #Test the trained model, run until done, return list of visited coords
    def get_path_from_model(self,model):
        obs = self.env.reset()
        xcoord_list = [self.env.x]      #Initialize list of path coordinates with initial position
        ycoord_list = [self.env.y]
        zcoord_list = [self.env.z]
        info = {}

        while True:
            action, _states = model.predict(obs)
            obs, rewards, done, info = self.env.step(action)
            
            print("reward: ",rewards) 
            xcoord_list.append(info['x'])
            ycoord_list.append(info['y'])
            zcoord_list.append(info['z'])
            
            if done: break

        return xcoord_list, ycoord_list, zcoord_list, info
  


    def plot_ball(self, figure, name, color, x0, y0, z0, radius):
            # Make data
            u = np.linspace(0, 2 * np.pi, 100)
            v = np.linspace(0, np.pi, 100)

            x = x0 + radius * np.outer(np.cos(u), np.sin(v))
            y = y0 + radius * np.outer(np.sin(u), np.sin(v))
            z = z0 + radius * np.outer(np.ones(np.size(u)), np.cos(v))

            # Plot the surface
            figure.add_trace(
                go.Surface(x=x, y=y, z=z, colorscale=color, showscale=False, name=name),)        
        
