#import yolo
import cv2
import os 
import numpy as np
from core import utils as utils

from PIL import Image

def main():

    imgDir = "img/"
    imgName = "senal-stop-416x2.jpg"
    #Load target image
    imgOriginal = cv2.imread(imgDir + imgName)

    #StackOverflow:
    #  https://stackoverflow.com/questions/43391205/add-padding-to-images-to-get-them-into-the-same-shape
    
    #ht, wd, cc= imgOriginal.shape
    #ww = 416
    #hh = 416
    #color = (255,255,255)
    #result = np.full((hh,ww,3), color, dtype=np.uint8)
#
    ## compute center offset
    #xx = (ww - wd) // 2
    #yy = (hh - ht) // 2
#
    #result[yy:yy+ht, xx:xx+wd] = imgOriginal
#
    #cv2.imwrite("senal-stop-416x2.jpg", result)

    #Create random image
    #imgAdv = utils.getRandomImg(width=64, height=64, save=True)
    
    imgAdv = cv2.imread(imgDir + "adv_img.jpg")

    #Merge both imgOriginal and imgAdv
    imgRes = utils.mergeImages(imgOriginal, imgAdv, save=True, saveName="imgRes.jpg")
    
    #Get detections with yolo, send name of the image inside img/ directory
    detectionsAttck = utils.detectYolo(imgName="imgRes.jpg")
    detectionOr = utils.detectYolo(imgName=imgName)

    print("\nDetections: " + str(detectionsAttck))


if __name__ == "__main__":
  main()
