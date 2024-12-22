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
        return self.model(img)

    def export(self, Format="onnx"):
        return self.model.export(format=Format)

#model = instanceSegmentationModel('weights/yolov8x-seg.pt')
#results = model.trainModel(dataPath=f'{HOME}/datasets/branches/data.yaml', epochs=250, imgSize=640)
#model.validateModel()
model_onnx = YOLO("weights/best.pt")
img = cv2.imread("unit_test_images/TREE3.jpg")
results = model_onnx(img)
results[0].show()

#model.export(Format="onnx")