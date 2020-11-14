import os
import subprocess
#Your darknet path goes here
darknetPath = "/home/francisco/TFG/darknet"

class YOLO(object):

    def __init__(self, imagePath="data/dog.jpg"):
        self.imagePath = imagePath
        #Some hyperparameters
        self.treshold = 0.2


    def getDetection(self):

        darknetComd = "./darknet detect cfg/yolov3-tiny.cfg yolov3-tiny.weights " + self.imagePath

        process = subprocess.Popen(darknetComd.split(), stdout=subprocess.PIPE, cwd=darknetPath)
        output, error = process.communicate()

        #print("Darknet Output is: " + str(output))
        outputList = str(output).split("\\")
        detecList = [[]]
        detecList.pop()
        for e in outputList:
            print("Elemento del output: "+str(e))
            #Save detection
            if '%' in str(e):
                detecList.append(str(e).split(":"))

        #print(str(detecList))
        return detecList


"""
def main():

    model = yolo(imagePath=)
    detections = model.getDetection()

if __name__ == "__main__":
  main()
"""