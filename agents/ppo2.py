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



class ppo2leveltrain(ppo2):

    def leveltrain(self, from_level, to_level, env, timesteps, level_modelpath, tensorboard_logs_path):
        model = PPO2('MlpPolicy', env, verbose=1, tensorboard_log=tensorboard_logs_path)
        model.save(level_modelpath)

        for current_level in range(from_level, to_level+1):                                  # Train model for increasingly difficult levels      
            if current_level == from_level: new_timesteps = int(timesteps)//(8*to_level)     # Divide by 8 as n_env = 8
            else: new_timesteps = int(timesteps)//to_level

            env = gym.make('DeepWellEnvSpherlevel'+str(current_level)+'-v0')

            model = self.load(level_modelpath, tensorboard_logs_path)                     # Load previous model
            env_str = self.get_env_str(env)
            model.set_env(make_vec_env(env_str, n_envs=8))
            model.learn(total_timesteps=new_timesteps, reset_num_timesteps=False, tb_log_name="TB_"+datetime.now().strftime('%d%m%y-%H%M'))      # Continue training previous model
            
            level_modelpath = level_modelpath[0:-1] + str(current_level)                  # Generate new name of newly trained model
            model.save(level_modelpath)                                                   # Save newly trained model
            
            print("====================== Level " + str(current_level) + " finished with "+ str(new_timesteps) +" timesteps ==========================")
    
        return model
    

    def train(self, env, timesteps, modelpath, tensorboard_logs_path):
        print("NOTE: No matter what env you pass to this agent, ppo2leveltrain will use DeepWellEnvSpherlevel1-v0 initially, and then increse levels ")
        env = gym.make('DeepWellEnvSpherlevel1-v0')
        level_modelpath = modelpath + "_level1"
        total_levels = 5
        return self.leveltrain(1, total_levels, env, timesteps, level_modelpath, tensorboard_logs_path)


    def retrain(self, env, timesteps, level_modelpath, tensorboard_logs_path):
        if not level_modelpath.split('_')[-1][0:-1] == "level":
            raise ValueError("Model name needs to end with '_level1' or level higher than 1")

        try: input_model_level = int(level_modelpath[-1])
        except: print("Could not extract level number from model. Model name needs to end with '_level1' or level higher than 1") 
        
        total_levels = 5
        print("MODEL LEVEL " + str(input_model_level) + " DETECTED. TRAINING TO LEVEL " + str(total_levels))
        return self.leveltrain(input_model_level+1, total_levels, env, timesteps, level_modelpath, tensorboard_logs_path)


    