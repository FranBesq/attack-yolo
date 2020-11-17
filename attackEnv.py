import gym
import cv2
import core.utils as utils
import numpy as np
from gym import spaces

#Dimensions of input image
HEIGHT = 416
WIDTH = 416
#Patch size
ACTION_H = 64
ACTION_W = 64

class AttackEnv(gym.Env):
  """Custom Environment for performing adversarial attacks on cnns that follows gym interface"""
  metadata = {'render.modes': ['human']}

  def __init__(self, normActions=True):

    #Anyway, its hardcoded in self._get_obs()
    self.imgDir = "img/",
    self.imgName = "senal-stop-416x2.jpg"
    self.targetClass = "nstop sign"
    self.deceiveClass = True
    self.desiredClass = "ntraffic light"
    self.normActionSpace = True
    #Set to false if using AttackEnvTest
    if normActions == False:
      self.normActions = False
    else:
      self.normActions = True

    super(AttackEnv, self).__init__()
    # Define action and observation space
    # Action space is also an image, may be normalized
    if self.normActionSpace is True:
        self.action_space = spaces.Box(low=-1, high=1, shape=
                    (ACTION_H * ACTION_W * 3,), dtype=np.float32)
    else:
      self.action_space = spaces.Box(low=0, high=255, shape=
                    (ACTION_H * ACTION_W * 3,), dtype=np.uint8)
    # Example for using image as input:
    self.observation_space = spaces.Box(low=0, high=255, shape=
                    (HEIGHT, WIDTH, 3), dtype=np.uint8)

  def step(self, action):
    # Execute one time step within the environment
    reward = 0.0
    is_class_found = 0
    class_deceived = 0
    class_score = 0
    imgOr = self._get_obs()

    #Get an image from action vector
    action = np.reshape(action, (ACTION_H, ACTION_W, 3))

    #Normalize action if needed
    if self.normActions == True:
        normalizedImg = np.zeros((ACTION_H, ACTION_W))
        normalizedImg = cv2.normalize(action, normalizedImg, 0, 255, cv2.NORM_MINMAX)
        action = normalizedImg

    #Consider the overlay the action
    utils.mergeImages(imgOr, action, normBack=True)
    res = utils.detectYolo()

    #Compute reward
    for detection in res:
      if self.targetClass in detection:
        is_class_found += 1
        #Convert score to int
        class_score += int(detection[1].replace("%", ""))
      
      if self.desiredClass in detection and self.deceiveClass == True:
        class_deceived = int(detection[1].replace("%", ""))
    
    #Do we care about previous detection?>

    #Focus on not detecting our target class
    if class_score == 0:
      reward = 1000
    #Punish each score point target class got
    else: 
      reward -= class_score * 10

    #Reward detectin wrong class
    reward += class_deceived * 10

    return imgOr, reward, True, {}

    ...
  def reset(self):
    # Reset the state of the environment to
    return self._get_obs()
    
    ...
  def render(self, mode='human', close=False):
    # Render the environment to the screen
    ...

  def _get_obs(self):
    return cv2.imread("img/senal-stop-416x2.jpg") #Remove this hardcoded path