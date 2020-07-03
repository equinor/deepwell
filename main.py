
from plothandler import *                           #Import the server that plots the result in browser
from run_dw_env.run_dw_env import *
#from <FOLDERNAME>.<FILENAME> import *              #Import your code/agents like this. The star means that you import all classes in the file.

def main():
    ###### INSTANTIATE AND TRAIN YOUR AGENT HERE ######

    agent = run_dw()
    figure = agent.get_plot()








    #figure = agent.fig                   #Enter figure here from agent using agent.fig, agent.close() or agent.render() depending on implementation

    ###### THIS PART STARTS THE WEBSERVER FOR SHOWING PLOT ######
    try:
        figure
    except:
        print("Figure for plotting in main.py is not defined or wrong type. \nFix in main.py or ignore if plotting is not relevant.")

    application = MyApplication(figure)
    http_server = tornado.httpserver.HTTPServer(application)
    http_server.listen(8080)

    print("http://127.0.0.1:8080/")
    print("Press Ctrl+C to quit")

    tornado.ioloop.IOLoop.instance().start()



if __name__ == "__main__":
    main()
