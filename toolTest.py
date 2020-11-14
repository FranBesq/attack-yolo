#import yolo
import cv2
import os 
import numpy as np
from core import utils as utils

from PIL import Image

def main():

    imgDir = "img/"
    imgName = "senal-stop-416.jpg"
    #Load target image
    imgOriginal = cv2.imread(imgDir + imgName)

    #Create random image
    imgAdv = utils.getRandomImg(width=64, height=64, save=True)
    
    #Merge both imgOriginal and imgAdv
    imgRes = utils.mergeImages(imgOriginal, imgAdv, save=True, saveName="imgRes.jpg")
    
    #Get detections with yolo, send name of the image inside img/ directory
    detectionsAttck = utils.detectYolo(imgName="imgRes.jpg")
    detectionOr = utils.detectYolo(imgName=imgName)

    print("\nDetections: " + str(detectionsAttck)) 


if __name__ == "__main__":
  main()
