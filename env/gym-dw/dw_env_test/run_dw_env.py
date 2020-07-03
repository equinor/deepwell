import gym
import env.DeepWellEnv
from stable_baselines import TRPO
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2
import matplotlib
import matplotlib.pyplot as plt

env = gym.make('DeepWellEnv-v0')
######## use TRPO or PPO2
#model = TRPO(MlpPolicy, env, verbose=1)
model = PPO2(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=100000)

xcoord = []
ycoord = []
obs = env.reset()

while True:
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    print(obs)
    print("reward: ",rewards)
    xcoord.append(obs[0])
    ycoord.append(obs[1])
    if done:
        break

im = plt.subplot()
plt.plot(xcoord,ycoord)
plt.gca().invert_yaxis()
plt.scatter(obs[4],obs[5],s=150)
plt.xlabel("Cross Section")
plt.ylabel("TVD")
plt.show() 
