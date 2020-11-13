
darknetPath = ""

class yolo:

    def __init__(self, imagePath="data/dog"):
        self.imagePath = os.path.join(darknetPath, imagePath)
        #Some hyperparameters
        self.treshold = 0.2


    def getDetection(self):
        