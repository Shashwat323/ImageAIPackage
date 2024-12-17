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

    def validateModel(self):
        return self.model.val()

    def predict(self, img):
        return self.model.predict(img)

    def export(self, Format="onnx"):
        return self.model.export(format=Format)

model = instanceSegmentationModel('weights/yolov8x-seg.pt')
results = model.trainModel(dataPath=f'{HOME}/datasets/branches/data.yaml', epochs=100, imgSize=640)
model.validateModel()
#img = cv2.imread("unit_test_images/ANIMALS3.jpg")
#results = model.predict(img)

model.export(Format="onnx")