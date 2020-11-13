#import yolo
import cv2
import numpy as np

from PIL import Image

#Reduces img to desired size
def image_preporcess(image, target_size):

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

    ih, iw    = target_size
    h,  w, _  = image.shape

    scale = min(iw/w, ih/h)
    nw, nh  = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh))

    image_paded = np.full(shape=[ih, iw, 3], fill_value=128.0)
    dw, dh = (iw - nw) // 2, (ih-nh) // 2
    image_paded[dh:nh+dh, dw:nw+dw, :] = image_resized
    image_paded = image_paded / 255.

    return image_paded


img_path = "senal-stop.jpg"


adv_size = 64 #Size of adversarial image


#Load target image
img_original = cv2.imread(img_path)
img_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
#Shape original image to fit yolo input size
img_mod = image_preporcess(np.copy(img_original), [416, 416])

cv2.imwrite('senal-stop-416.jpg', img_mod)

#Create random image
img_adv = np.ones((adv_size, adv_size, 3), dtype=np.uint8)
bgr = cv2.split(img_adv)
cv2.randu(bgr[0], 0, 255)
cv2.randu(bgr[1], 0, 255)
cv2.randu(bgr[2], 0, 255)
img_adv = cv2.merge(bgr)

cv2.imwrite('adv_img.jpg', img_adv)
img_adv = cv2.imread('adv_img.jpg')

#alpha_adv = 1
#alpha_or = 1

#for c in range(0, 3):
#    img_mod[250:250+adv_size, 200:200+adv_size, c] = (alpha_adv * img_adv[:, :, c] + alpha_or * img_mod[250:250+adv_size, 200:200+adv_size, c])

#Show results
#img_adv = Image.fromarray(img_adv)

cv2.imshow("Composited image", img_mod)
#cv2.imwrite('result.jpg', img_mod)
cv2.waitKey(0)

