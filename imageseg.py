import os
import ultralytics
import cv2
from ultralytics import YOLO

HOME = os.getcwd()
ultralytics.checks()

class instanceSegmentationModel:
    def __init__(self, path=None):
        self.model = YOLO(path)

    def trainModel(self, dataPath, epochs, imgSize, weights=None):
        if weights:
            return self.model.train(data=dataPath, epochs=epochs, imgsz=imgSize, weights=weights)
        return self.model.train(data=dataPath, epochs=epochs, imgsz=imgSize)

    def testModel(self):
        pass

    def validateModel(self):
        pass

    def predict(self, img):
        return self.model.predict(img)

model = instanceSegmentationModel('yolov8s-seg.pt')
results = model.trainModel(dataPath=f'{HOME}/datasets/turtle/data.yaml', epochs=1, imgSize=320, weights='yolov8s-seg.pt')

img = cv2.imread("unit_test_images/ANIMALS3.jpg")

results = model.predict(img)