from agents.agent_super import agent
from datetime import datetime

from stable_baselines.common import make_vec_env

from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines import DQN

import gym
from gym_dw import envs

class dqn(agent):

    def train(self, env, timesteps,modelpath,tensorboard_logs_path):
        model = DQN(MlpPolicy, env, verbose=1)
        model.learn(total_timesteps=timesteps)
        return model

    def load(self, modelpath, tensorboard_logs_path):
        model = DQN.load(modelpath, tensorboard_log=tensorboard_logs_path)
        return model

    def retrain(self, env, timesteps, modelpath, tensorboard_logs_path):
        model = self.load(modelpath, tensorboard_logs_path)
        env_str = self.get_env_str(env)                                     #get_env_str method in superclass agent_super.py
        model.set_env(make_vec_env(env_str, n_envs=1))
        model.learn(total_timesteps=timesteps, reset_num_timesteps=False, tb_log_name="TB_"+datetime.now().strftime('%d%m%y-%H%M'))      #Continue training
        return model


class dqnleveltrain(dqn):
    def __init__(self):
        self.policy_kwargs = dict(layers = [64,64,64,32]) #To change network architecture from the default [64,64]

    def leveltrain(self, from_level, to_level, env, timesteps, level_modelpath, tensorboard_logs_path):
        model = DQN('MlpPolicy', env, verbose=1, policy_kwargs=self.policy_kwargs, prioritized_replay=True, buffer_size=100000,
                     learning_rate=0.0003,exploration_final_eps=0,tensorboard_log=tensorboard_logs_path)
        model.save(level_modelpath)

        for current_level in range(from_level, to_level+1):                               # Train model for increasingly difficult levels      
            env = gym.make('DeepWellEnvSpherlevel'+str(current_level)+'-v0')

            model = self.load(level_modelpath, tensorboard_logs_path)                     # Load previous model
            env_str = self.get_env_str(env)
            model.set_env(make_vec_env(env_str, n_envs=1))
            model.learn(total_timesteps=timesteps, reset_num_timesteps=False, tb_log_name="TB_"+datetime.now().strftime('%d%m%y-%H%M'))      # Continue training previous model
            
            level_modelpath = level_modelpath[0:-1] + str(current_level)                  # Generate new name of newly trained model
            model.save(level_modelpath)                                                   # Save newly trained model
            
            print("====================== Level " + str(current_level) + " finished with "+ str(timesteps) +" timesteps ==========================")
    
        return model
    

    def train(self, env, timesteps, modelpath, tensorboard_logs_path):
        print("NOTE: No matter what env you pass to this agent, dqnleveltrain will use DeepWellEnvSpherlevel1-v0 initially, and then increse levels ")
        env = gym.make('DeepWellEnvSpherlevel1-v0')
        level_modelpath = modelpath + "_level1"
        total_levels = 6                                                                  # Set how many levels you want to train
        return self.leveltrain(1, total_levels, env, timesteps, level_modelpath, tensorboard_logs_path)


    def retrain(self, env, timesteps, level_modelpath, tensorboard_logs_path):
        if not level_modelpath.split('_')[-1][0:-1] == "level":
            raise ValueError("Model name needs to end with '_level1' or level higher than 1")

        try: input_model_level = int(level_modelpath[-1])
        except: print("Could not extract level number from model. Model name needs to end with '_level1' or level higher than 1") 
        
        total_levels = 5
        print("MODEL LEVEL " + str(input_model_level) + " DETECTED. TRAINING TO LEVEL " + str(total_levels))
        return self.leveltrain(input_model_level+1, total_levels, env, timesteps, level_modelpath, tensorboard_logs_path)