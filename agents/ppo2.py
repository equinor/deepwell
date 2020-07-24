from agents.agent_super import agent

from datetime import datetime
from stable_baselines.common import make_vec_env

from stable_baselines import PPO2
from custom_callback.evalcallback import EvalCallback2

import gym
from gym_dw import envs


class ppo2(agent):

    def train(self, env, timesteps, modelpath, tensorboard_logs_path):
        model = PPO2('MlpPolicy', env, verbose=1, tensorboard_log=tensorboard_logs_path)
        model.learn(total_timesteps = timesteps, tb_log_name="TB_"+datetime.now().strftime('%d%m%y-%H%M'))
        model.save(modelpath)
        return model

    def load(self, modelpath, tensorboard_logs_path):
        print("TRY TO LOAD: ", modelpath)
        model = PPO2.load(modelpath, tensorboard_log=tensorboard_logs_path)              
        return model

    def retrain(self, env, timesteps, modelpath, tensorboard_logs_path):
        model = self.load(modelpath, tensorboard_logs_path)
        env_str = self.get_env_str(env)
        model.set_env(make_vec_env(env_str, n_envs=8))
        model.learn(total_timesteps=timesteps, reset_num_timesteps=False, tb_log_name="TB_"+datetime.now().strftime('%d%m%y-%H%M'))      #Continue training
        model.save(modelpath)
        return model


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

            env = gym.make('DeepWellEnvSpherlevel'+str(i)+'-v0')
            model = super().retrain(env, int(timesteps), modelpath, tensorboard_logs_path)
            
            print("====================== Level " + str(i) + " finished with "+ str(timesteps) +" timesteps ==========================")
            
            #The models at each level gets saved in super().retrain()
        return model


class ppo2callback(ppo2):
    def retrain(self, env, timesteps, modelpath, tensorboard_logs_path):
        #Periodically evalute agent, save best model
        eval_callback = EvalCallback2(env, best_model_save_path='app/model_logs/', 
                        log_path='app/model_logs/', eval_freq=1000,
                        deterministic=True, render=False) 

        model = self.load(modelpath, tensorboard_logs_path)
        env_str = self.get_env_str(env)
        model.set_env(make_vec_env(env_str, n_envs=8))
        model.learn(total_timesteps=timesteps, callback=eval_callback, reset_num_timesteps=False, tb_log_name="TB_"+datetime.now().strftime('%d%m%y-%H%M'))      #Continue training
        model.save(modelpath)
        return model

    