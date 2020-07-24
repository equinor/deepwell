from agents.agent_super import agent
from datetime import datetime

from stable_baselines.common import make_vec_env

from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines import DQN

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
        #model.set_env(env)
        model.learn(total_timesteps=timesteps, reset_num_timesteps=False, tb_log_name="TB_"+datetime.now().strftime('%d%m%y-%H%M'))      #Continue training
        return model


    