#import yolo
import cv2
import os 
import numpy as np
import attackEnv 
import core.utils as utils

from PIL import Image

def main():

    env = attackEnv.AttackEnv(normActions=False)#Actions normalized by .utils
    
    #Get random image as action
    imgAdv = utils.getRandomImg(width=64, height=64, save=True)

    state, reward, done, info = env.step(imgAdv)

    print("\nObservation shape is: " + str(state.shape))
    print("\nReward given from Env = " + str(reward))

if __name__ == "__main__":
  main()
