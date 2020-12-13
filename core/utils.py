#import yolo
import cv2
from core import yolo
import os 
import numpy as np
from PIL import Image

dir_path = os.path.dirname(os.path.realpath(__file__))
#Your image folder path goes here
img_path = "/home/francisco/Documents/PIC/attack-yolo/img/"

# Returns random image of given size
def getRandomImg(height=32, width=32, norm=True,imgName="adv_img.jpg", save=False):
    
    img_adv = np.ones((height, width, 3), dtype=np.uint8)
    bgr = cv2.split(img_adv)
    cv2.randu(bgr[0], 0, 255)
    cv2.randu(bgr[1], 0, 255)
    cv2.randu(bgr[2], 0, 255)
    img_adv = cv2.merge(bgr)

    if norm is True:
        normalizedImg = np.zeros((width, height))
        normalizedImg = cv2.normalize(img_adv, normalizedImg, 0, 255, cv2.NORM_MINMAX)
        img_adv = normalizedImg

    if save == True:
        cv2.imwrite(img_path + imgName, img_adv)
    
    return img_adv

def mergeImages(imgBack, imgOver, offsetH=150, offsetW=150, save=True, normBack=False,saveName="result.jpg"):
    
    overShape = imgOver.shape[:2]
    #Check if offset + sizeOver out of bounds??

    #Assume imgOver is normalized, so we normalize ImgBack only
    if normBack is True:
        normalizedImg = np.zeros((imgBack.shape[0], imgBack.shape[1]))
        normalizedImg = cv2.normalize(imgBack, normalizedImg, 0, 255, cv2.NORM_MINMAX)
        imgBack = normalizedImg

    # Copy imgOver on top of imgBack given an offset
    imgBack[offsetH:offsetH+overShape[0], offsetW:offsetW+overShape[1]] = imgOver
    if save == True:
        cv2.imwrite(img_path + saveName, imgBack)
    
    return imgBack

def detectYolo(imgName = "result.jpg"):
    #Call YOLOv3 Tiny
    model = yolo.YOLO(imagePath=img_path + imgName)
    detection = model.getDetection()
    #print(str(detection))
    return detection

#Func from StackOverflow
def resizeImg(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized


'''
Code from:
https://www.pyimagesearch.com/2014/09/15/python-compare-two-images/
'''
def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	
	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err

#cv2.waitKey(0)

