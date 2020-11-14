#import yolo
import cv2
import core.yolo as yolo
import core.utils as utils
import os 
import numpy as np

from PIL import Image


imgPath = "img/senal-stop-416.jpg"


#Call YOLOv3 Tiny
yolo = yolo.yolo(imagePath=os.path.join(dir_path,"result.jpg"))
detection = yolo.getDetection()
print(str(detection))


#cv2.waitKey(0)

def main():

    #Load target image
    imgOriginal = cv2.imread(img_path)

    #Create random image
    imgAdv = utils.getRandomImg(width=64, height=64, save=True)
    
    #Merge both imgOriginal and imgAdv
    imgRes = utils.mergeImages(imgOriginal, imgAdv, save=True, saveName="img/imgRes.jpg")
    
    #Get detections with yolo, send name of the image inside img/ directory
    detections = utils.detectYolo(imgPath="imgRes.jpg")

if __name__ == "__main__":
  main()
