import gym
import cv2
import core.utils as utils
import numpy as np
from gym import spaces

#Dimensions of input image
HEIGHT = 416
WIDTH = 416

class AttackEnv(gym.Env):
  """Custom Environment for performing adversarial attacks on cnns that follows gym interface"""
  metadata = {'render.modes': ['human']}

  def __init__(self):
    super(AttackEnv, self).__init__()
    # Define action and observation space
    # Action space is also an image
    self.action_space = spaces.Box(low=0, high=255, shape=
                    (HEIGHT, WIDTH, 3), dtype=np.uint8)
    # Example for using image as input:
    self.observation_space = spaces.Box(low=0, high=255, shape=
                    (HEIGHT, WIDTH, 3), dtype=np.uint8)

  def step(self, action):
    # Execute one time step within the environment
    ...
  def reset(self):
    # Reset the state of the environment to
    
    ...
  def render(self, mode='human', close=False):
    # Render the environment to the screen
    ...