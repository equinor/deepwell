from plot_server import PlotServer2d, PlotServer3d                           #Import the server that plots the result in browser
from agents.ppo2 import ppo2, ppo2leveltrain, ppo2callback
from agents.dqn import dqn, dqnleveltrain

from agents.agent_loader import AgentLoader
#from <FOLDERNAME>.<FILENAME> import *              #Import your code/agents like this. The star means that you import all classes in the file.

import gym
from gym_dw import envs


def main():
    ###### INSTANTIATE AND TRAIN YOUR AGENT HERE ######

    #Remember to put your agents in the agents folder, you can use ppo2.py as a template

    #Set up environment
    env = gym.make('DeepWellEnvSpher-v0')
    #agent = ppo2()
    #agent = dqn()
    agent = dqnleveltrain()

    model = AgentLoader().get_model(env,agent)

    PlotServer3d().show_model(env, model)         #The server needs a model and an env to generate a wellpath and plot it


if __name__ == "__main__":
    main()
