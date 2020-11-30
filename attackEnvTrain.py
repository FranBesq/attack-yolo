import gym
import attackEnv 
import numpy as np

#from stable_baselines.td3.policies import CnnPolicy
from stable_baselines.sac.policies import CnnPolicy
#from stable_baselines.common.policies import CnnPolicy
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize, VecCheckNan
from stable_baselines.ddpg.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines import TD3, A2C, PPO1, PPO2, SAC

log_dir = "./temp/"

if __name__ == '__main__':
    # multiprocess environment
    #num_cpu = 2
    env = attackEnv.AttackEnv()
    env = DummyVecEnv([lambda: env])
    env = VecCheckNan(env, raise_exception=True)
    #env = VecNormalize(env, norm_obs=True, norm_reward=False)

    #n_actions = env.action_space.shape[-1]
    #action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    model = SAC(CnnPolicy, env, verbose=1, learning_rate=0.01)
    model.learn(total_timesteps=1000, log_interval=20)
    model.save("SAC_attack_pp_model")

    del model # remove to demonstrate saving and loading

    model = SAC.load("sac_attack_pp_model")

    # Enjoy trained agent
    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        print(str(obs.shape))
        #env.render()
        #env.save_running_average(log_dir)
