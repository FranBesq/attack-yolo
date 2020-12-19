# attack-yolo

*(Work in progress)*

Developing adversarial attacks on YOLO algorithm for computer vision 

High level overview

![alt text](https://github.com/FranBesq/attack-yolo/blob/main/img/esquema_high_level.png)

# Requirements

* Darknet - https://github.com/AlexeyAB/darknet#requirements 

* [OpenCV python](https://pypi.org/project/opencv-python/)

* [NumPy](https://pypi.org/project/numpy/)

* [OpenAI Gym](https://pypi.org/project/gym/)

* [Stable Baselines for TF](https://stable-baselines.readthedocs.io/en/master/guide/install.html) or [Stable Baselines 3 for pytorch](https://stable-baselines3.readthedocs.io/en/master/guide/install.html)

* nvidia-docker + [darknet container](https://hub.docker.com/r/takuyatakeuchi/yolo-darknet/) not needed if using Darknet above

# Developing attacks with RL

![alt text](https://github.com/FranBesq/attack-yolo/blob/main/img/esquema_low_level.png)

# TO DO

* normalize rewards - get better mse
* use L_inf?
