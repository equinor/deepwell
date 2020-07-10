
from plothandler import *                           #Import the server that plots the result in browser
from run_dw_env.run_dw_env import *
#from <FOLDERNAME>.<FILENAME> import *              #Import your code/agents like this. The star means that you import all classes in the file.

#import gym
#from gym_dw import envs

#Env names:
#agent = gym.make('DeepWellEnv-v0')
#agent = gym.make('DeepWellEnv2-v0')

def main():
    ###### INSTANTIATE AND TRAIN YOUR AGENT HERE ######

    #Remember to move your agent into its own folder (look at the run_dw_env folder), then import it here before commiting and pushing

    
    #Set up environment
    agent = run_dw()
    
    #Train or load model
    model = agent.get_model()
    
    #Test model and get lists of visited coordinates
    xcoord,ycoord = agent.test_model(model)
    print("Point: ", agent.xt, agent.yt)
    print("Target: ",agent.env.xtarget,agent.env.ytarget)

    figure = agent.env.render(xcoord,ycoord) #Enter figure here from agent using agent.fig, agent.get_plot(), agent.close() or agent.render() depending on implementation
    figure.gca().invert_yaxis()
    
    

    '''
    env = gym.make('DeepWellEnv-v0')
    xcoord = []
    ycoord = []
    obs = env.reset()
    xt = 0
    yt = 0

    for _ in range(1000):
        #action = env.action_space.sample()
        action = 1
        obs, rewards, done, info = env.step(action)
        xcoord.append(info['x'])
        ycoord.append(info['y'])
        print(rewards)
    print("Last position: ",xcoord[-1],ycoord[-1])
    figure = env.render(xcoord,ycoord)
    figure.gca().invert_yaxis()
    '''


    ###### THIS PART STARTS THE WEBSERVER FOR SHOWING PLOT ######

    try:
        figure
    except NameError:
        raise TypeError("Figure for plotting in main.py is not defined or wrong type. Fix in main.py or ignore if plotting is not relevant.")

    application = MyApplication(figure)
    http_server = tornado.httpserver.HTTPServer(application)
    http_server.listen(8080)

    print("http://127.0.0.1:8080/")
    print("Press Ctrl+C to quit")

    tornado.ioloop.IOLoop.instance().start()



if __name__ == "__main__":
    main()
