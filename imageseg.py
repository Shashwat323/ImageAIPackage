import os
import ultralytics
import cv2
from ultralytics import YOLO

HOME = os.getcwd()
ultralytics.checks()

model = YOLO(f'{HOME}/weights/yolov8n-seg.pt')

results = model.train(data='C:/Users/cohen/OneDrive/Documents/GitHub/ImageAIPackage/datasets/jellyfishdataset/data.yaml', epochs=50, imgsz=640)

img = cv2.imread("unit_test_images/CATANDDOG.jpg")

results = model.predict(img)
for result in results:
    result.save(filename="result.jpg")