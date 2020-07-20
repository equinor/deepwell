
from plothandler import Plotter                           #Import the server that plots the result in browser
from agents.ppo2 import ppo2
#from <FOLDERNAME>.<FILENAME> import *              #Import your code/agents like this. The star means that you import all classes in the file.

#import gym
#from gym_dw import envs

#Env names:
#agent = gym.make('DeepWellEnv-v0')
#agent = gym.make('DeepWellEnv2-v0')

def main():
    ###### INSTANTIATE AND TRAIN YOUR AGENT HERE ######

    #Remember to put your agents in the agents folder, you can use ppo2.py as a template
    
    #Set up environment
    agent = ppo2()

    #Train or load model
    model = agent.get_model()

    figure = agent.get_fig(model)

    ###### THIS PART STARTS THE WEBSERVER FOR SHOWING PLOT ######
    try:
        figure
    except NameError:
        raise TypeError("Figure for plotting in main.py is not defined or wrong type. Fix in main.py or ignore if plotting is not relevant.")

    Plotter().show3d(figure, port=8080)         #Here you can specify at which port the plot should appear NOTE: Only 8080 is open in the docker-conainer


if __name__ == "__main__":
    main()
