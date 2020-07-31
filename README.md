# DeepWell
 
Team repo for Wishing Well team - Virtual summer internship 2020

Using the open source tools OpenAI Gym and Stable Baselines we simulate simple drilling environments and train agents to efficiently find the optimal drilling paths. The environments, which are either 2D or 3D, have included both targets to hit and hazards to avoid. More information on the different environments can be found in [gym_dw/envs](https://github.com/equinor/deepwell/blob/dev/env/gym-dw/gym_dw/envs/README.md). Among the reinforcement learning algorithms we have mostly used the TRPO, PPO2 and DQN. Some pretrained agents using these algorithms can be found in [trained models](https://github.com/equinor/deepwell/tree/dev/trained_models). Below you'll find instructions on how to start and keep track of the agent learning process for a given drilling environment.

## Installation (windows)

1. Make sure you have Docker Desktop installed on your computer
    https://docs.docker.com/docker-for-windows/install/

2. Clone the repository 

    git clone https://github.com/equinor/deepwell
    cd deepwell

3. Build the docker image

    .\deepwellstart.ps1 -build

    All dependencies are installed when building the image for the first time.

4. Train, retrain, leveltrain or load different agents using the arguments below.

## Running with script (Windows/PowerShell)

First, open powershell and make sure you are allowed to run scripts by entering:

    Set-ExecutionPolicy Unrestricted

You only have to do that one time.
Then you can navigate to the deepwell repository on your computer and build the container by running:

    .\deepwellstart.ps1 -build
Then, to train the agent, simply enter:

    .\deepwellstart.ps1 -run train <num timesteps> <model nickname> <agent>

You choose agent by giving one of these arguments:
* ppo2
* ppo2leveltrain
* ppo2callback
* dqn
* dqnleveltrain

*Dqnleveltrain* is chosen by default if no agent is given.

A handy tip is just to write the first couple of letters "de" and press tab to complete.

You can then see the plot at http://localhost:8080 and the tensorboard result from the training session at http://localhost:7007

The trained models will be saved to the trained_models folder. The logs for tensorboard are saved in the tensorboard_logs folder.

Note: You do not need to specify a nickname for the model. If you omit this, it will simply be called the current date and time like: '130720-1530.zip' 



### Examples of how you can use the script:

Rebuild docker image (needed to load changes in env):

    .\deepwellstart.ps1 -b

Train the model with 10000 timesteps and save it as 'myppo2model':

    .\deepwellstart.ps1 -r train 10000 myppo2model

Train the model with 10000 timesteps and save it with a  current date+time name:

    .\deepwellstart.ps1 -r train 10000
  
Retrain the myppo2model model with 10000 timesteps:

    .\deepwellstart.ps1 -r retrain 10000 myppo2model

Load and run model:

    .\deepwellstart.ps1 -r load <model nickname> <agent name>
    
For example:

    .\deepwellstart.ps1 -r load myppo2model ppo2


## Running without script (Non windows)

Instructions if you for some reason cannot launch docker using the script. 
(For example if you are on linux/mac)


First, navigate to the deepwell repository on your computer.

Then build the container using:

  

    docker build -t deepwell-app .

  

Then start it with:

  

    docker run -dit --mount type=bind,source="$(pwd)",target=/app -p 0.0.0.0:7007:6006 -p 8080:8080 --name dwrunning deepwell-app


Then start the tensorboard server:

    docker exec -dit dwrunning tensorboard --logdir /usr/src/app/logs/ --host 0.0.0.0 --port 6006

And lastly, run python with

    docker exec -it dwrunning python /app/main.py <text_arg_for_python> <num_arg_for_python>

For information regarding the paramters, see the "running with script" section above.

It should now display the plot at http://localhost:8080/ and the tensorboard result from the training session at http://localhost:7007
  

  

When you have changed your code, you need to remove the old container
  

    docker rm -f dwrunning

  
Then just run again.
  

Note that if you edit the env, you have to build the container again for the changes to take effect. If you are not seeing your changes being applied, try rebuilding.
