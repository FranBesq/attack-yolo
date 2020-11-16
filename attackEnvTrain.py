import gym
import attackEnv 
import numpy as np

from stable_baselines.td3.policies import CnnPolicy
#from stable_baselines.common.policies import CnnPolicy
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines.ddpg.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines import TD3, A2C, PPO1, PPO2

log_dir = "./temp/"

if __name__ == '__main__':
    # multiprocess environment
    #num_cpu = 2
    env = attackEnv.AttackEnv()
    env = DummyVecEnv([lambda: env])
    #env = VecNormalize(env, norm_obs=True, norm_reward=False)

    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))


    model = TD3(CnnPolicy, env, action_noise=action_noise, verbose=1)
    model.learn(total_timesteps=2000, log_interval=10)
    model.save("td3_attack_model")

    del model # remove to demonstrate saving and loading

    model = TD3.load("td3_attack_model")

    # Enjoy trained agent
    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        #env.render()
        #env.save_running_average(log_dir)