#import yolo
import cv2
import numpy as np

from PIL import Image

#Reduces img to desired size
def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
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


img_path = "img/senal-stop-416.jpg"
adv_size = 64 #Size of adversarial image

#Load target image
img_original = cv2.imread(img_path)
#img_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)

#Shape original image to fit yolo input size
#img_mod = image_resize(img_original, 416, 416)
#img_mod = cv2.resize(img_original, (416, 416))
#img_mod.astype(np.uint8)

#cv2.imwrite('senal-stop-416.jpg', img_mod)

#Create random image
img_adv = np.ones((adv_size, adv_size, 3), dtype=np.uint8)
bgr = cv2.split(img_adv)
cv2.randu(bgr[0], 0, 255)
cv2.randu(bgr[1], 0, 255)
cv2.randu(bgr[2], 0, 255)
img_adv = cv2.merge(bgr)

cv2.imwrite('img/adv_img.jpg', img_adv)
img_adv = cv2.imread('img/adv_img.jpg')

#Show results
#img_adv = Image.fromarray(img_adv)

img_original[150:150+adv_size, 150:150+adv_size] = img_adv

cv2.imshow("Composited image", img_original)
cv2.imwrite('result.jpg', img_original)
cv2.waitKey(0)

