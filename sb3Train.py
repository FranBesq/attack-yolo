import gym
import attackEnv 
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.ppo import CnnPolicy

env = attackEnv.AttackEnv()

model = PPO(CnnPolicy, env, verbose=1, device='cpu', learning_rate=0.01)
model.learn(total_timesteps=500, log_interval=10)
model.save("ppo_sb3_attack")

del model # remove to demonstrate saving and loading

model = PPO.load("ppo_sb3_attack")

obs = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    if done:
      obs = env.reset()