import gym
import env.dw_diffeq_env
from stable_baselines import TRPO
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2
import matplotlib.pyplot as plt


env = gym.make('DeepWellEnv-v2')
######## use TRPO or PPO2
model = TRPO(MlpPolicy, env, verbose=1, tensorboard_log="./logs/")
#model = PPO2(MlpPolicy, env, verbose=1, tensorboard_log="./logs/")
model.learn(total_timesteps=100000)

# To start tensorboard run this in a different command line on the same env:
# "tensorboard --logdir=logs/ --host localhost --port 8088"
#model.save("testPPO2")
#model.load("testPPO2")

xcoord = []
ycoord = []
obs = env.reset()

for i in range(10):
    xcoord = []
    ycoord = []
    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        #print(obs)
        #print("reward: ",rewards)
        xcoord.append(obs[0,0])
        ycoord.append(obs[0,1])
        if done:
            break


    fig, ax = plt.subplots()
    ax.set_xlim([0, 3000])
    ax.set_ylim([0, 3000])
    ax.plot(xcoord, ycoord)
    fig.gca().invert_yaxis()
    circle = plt.Circle(env.targetball['center'], env.targetball['R'], color='g', label='target')
    ax.add_artist(circle)
    ax.set_xlabel("Cross Section")
    ax.set_ylabel("TVD")
    plt.show() 
