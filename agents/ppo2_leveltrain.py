
from agents.agent_super import agent

from datetime import datetime
from stable_baselines.common import make_vec_env

from stable_baselines import PPO2
from custom_callback.evalcallback import EvalCallback2

import gym
from gym_dw import envs

from agents.ppo2 import ppo2


class ppo2leveltrain(ppo2):

    def train(self, env, timesteps ,modelpath, tensorboard_logs_path):

        print("NOTE: No matter what env you pass to this agent, ppo2leveltrain will use DeepWellEnvSpherlevel1-v0 initially, and then increse levels ")
        
        levels = 5
        print("====================== Initiating environment ==========================")
        env = gym.make('DeepWellEnvSpherlevel1-v0')
        model = PPO2('MlpPolicy', env, verbose=1, tensorboard_log=tensorboard_logs_path)
        model.save(modelpath)

        # Train model for increasingly difficult levels
        for i in range(1, levels+1):
            #if i == 1:
            #    timesteps = float(timesteps)/(8*levels)  # Divide by 8 as num env = 8
            #else: 
            #    timesteps = float(timesteps)/(levels)    # When loading it trains the inputed number of timesteps
            
            print("TIMESTEPS: ", timesteps)

            env = gym.make('DeepWellEnvSpherlevel'+str(i)+'-v0')
            model = super().retrain(env, int(timesteps), modelpath, tensorboard_logs_path)
            
            print("====================== Level " + str(i) + " finished ==========================")

            #The models at each level gets saved in super().retrain()
        print("Trained with " + str(levels) + " levels with " + str(timesteps) + " timesteps each.")
        return model




    


    