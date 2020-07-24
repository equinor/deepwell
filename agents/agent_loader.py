import sys
from datetime import datetime
import numpy as np

# Filter tensorflow version warnings
import os               # https://stackoverflow.com/questions/40426502/is-there-a-way-to-suppress-the-messages-tensorflow-prints/40426709
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

import warnings         # https://stackoverflow.com/questions/15777951/how-to-suppress-pandas-future-warning
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)

import tensorflow as tf
tf.get_logger().setLevel('INFO')
tf.autograph.set_verbosity(0)

import logging
tf.get_logger().setLevel(logging.ERROR)


class AgentLoader:
    
    #Get model either by training a new one or loading an old one
    def get_model(self,env,agent):
        #sys.argv fetches arg when running "deepwellstart.ps1 -r arg". This is to make it possible to load,train or retrain the agent.
        mainpy_path, text_argument, num_argument, model_name = sys.argv[0], sys.argv[1], sys.argv[2], sys.argv[3]
        #mainpy_path is the path of main.py = '/app/main.py'
        #If no argument is given (deepwellstart.ps1 -r), text_argument = " "
        if model_name == " " : model_name = datetime.now().strftime('%d%m%y-%H%M')

        tensorboard_logs_path = "app/tensorboard_logs/"
        trained_models_path = "app/trained_models/"

        if text_argument == "train":
            print("====================== NOW TRAINING MODEL ==========================")
            model = agent.train(env, int(num_argument), trained_models_path + model_name, tensorboard_logs_path)
            #model.save(trained_models_path + model_name)
            return model
         
        elif text_argument == "retrain":                        # This is for retraining the model, for tensorboard integration load the tensorboard log from your trained model and create a new name in model.learn below.
            print("====================== NOW RETRAINING MODEL ==========================")
            model = agent.retrain(env, trained_models_path+model_name, tensorboard_logs_path, int(num_argument))
            #model.save(trained_models_path + model_name)
            return model

        elif text_argument == "load":                           # Load a saved model. Remove "/app/" if not running with docker
            print("====================== NOW LOADING MODEL ==========================")
            model_name = num_argument
            model = agent.load(trained_models_path + model_name, tensorboard_logs_path)
            return model
        else:
            print("====================== NO ARGUMENT OR NO KNOWN ARGUMENT ENTERED ==========================")
            #Code here


    


