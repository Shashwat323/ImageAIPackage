import os
import ultralytics
import cv2
from ultralytics import YOLO

HOME = os.getcwd()
ultralytics.checks()

def createModel(path):
    return YOLO(path)

def trainModel(dataPath, epochs, imgSize):
    return model.train(data=dataPath, epochs=epochs, imgsz=imgSize)


model = createModel(f'{HOME}/weights/yolov8x-seg.pt')
results = model.train(data='C:/Users/cohen/OneDrive/Documents/GitHub/ImageAIPackage/datasets/turtle/data.yaml', epochs=10, imgsz=320)

img = cv2.imread("unit_test_images/ANIMALS3.jpg")

results = model.predict(img)