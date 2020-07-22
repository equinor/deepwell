from plot_server import PlotServer                           #Import the server that plots the result in browser
from agents.ppo2 import ppo2
#from <FOLDERNAME>.<FILENAME> import *              #Import your code/agents like this. The star means that you import all classes in the file.

import gym
from gym_dw import envs
from stable_baselines import PPO2

#Env names:
#agent = gym.make('DeepWellEnv-v0')
#agent = gym.make('DeepWellEnv2-v0')

def main():
    ###### INSTANTIATE AND TRAIN YOUR AGENT HERE ######

    #Remember to put your agents in the agents folder, you can use ppo2.py as a template

    #Set up environment
    env = gym.make('DeepWellEnvSpher-v0')
    agent = ppo2(env)
    model = agent.get_model()

    ###### THIS PART STARTS THE WEBSERVER FOR SHOWING PLOT ######
    PlotServer().show_model_3d(env, model)         #The server needs a model and an env to generate a wellpath and plot it


if __name__ == "__main__":
    main()
