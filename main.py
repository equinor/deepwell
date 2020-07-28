from plot_server import PlotServer2d, PlotServer3d                           #Import the server that plots the result in browser
from agents.agent_loader import AgentLoader
#from <FOLDERNAME>.<FILENAME> import *              #Import your code/agents like this. The star means that you import all classes in the file.

import gym
from gym_dw import envs
import sys

def main():
    ###### INSTANTIATE AND TRAIN YOUR AGENT HERE ######

    #Set up environment
    env = gym.make('DeepWellEnvSpher-v0')
    
    #Remember to put your agents in the agents folder, you can use ppo2.py as a template

    model = AgentLoader().get_model(env)

    PlotServer3d().show_model(env, model)         #The server needs a model and an env to generate a wellpath and plot it


if __name__ == "__main__":
    main()
