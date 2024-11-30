import numpy as np
from PIL import Image
import cv2
import rembg
import os

"""
# Image
img = Image.open('Sample.png')
# Convert to Array
numpydata = np.asarray(img)
# Convert from Array
pilImage = Image.fromarray(numpydata)

or
# Save the image using OpenCV
cv2.imwrite('rgb_image_opencv.png', np_image)
"""

def normalise(image_file_path: str,greyscale: bool=False):
    # Read image
    image = cv2.imread(image_file_path)
    gray_image = image
    if (greyscale != True):
        # Image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Normalize grayscale
    #normalized_gray_image = cv2.normalize(gray_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    normalized_gray_image = gray_image / 255.0
    return normalized_gray_image

def normalise(img: np.ndarray, greyscale: bool=False):
    if (greyscale != True):
        img = sum(img) / 3
    return img / 255

# CLAHE (Contrast Limited Adaptive Histogram Equalization)
def histogram_equalisation(image_file_path: str, greyscale: bool=False, clahe: bool=False, gridsize: int=8):
    # Read image
    img = cv2.imread(image_file_path)
    if (clahe == False):
        if (greyscale):
            # Image to grayscale
            grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Apply histogram equalisation
            image = cv2.equalizeHist(grey_img)
        else:
            # Convert to HSV
            img_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            # Histogram equalisation on V-channel
            img_hsv[:, :, 2] = cv2.equalizeHist(img_hsv[:, :, 2])
            # Convert to RGB
            image = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
    else:
        if (greyscale):
            # Image to grayscale
            grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Create and apply CLAHE
            image = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(gridsize,gridsize)).apply(grey_img)
        else:
            # Convert to HSV
            img_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            # Histogram equalisation on V-channel
            img_hsv[:, :, 2] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(gridsize,gridsize)).apply(img_hsv[:, :, 2])
            # Convert to RGB
            image = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
    return image

def histogram_equalisation(img: np.ndarray, greyscale: bool=False, clahe: bool=False, gridsize: int=8):
    cv2.imwrite("temp.jpg",img)
    # Read image
    img = cv2.imread("temp.jpg")
    if (clahe == False):
        if (greyscale):
            # Image to grayscale
            grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Apply histogram equalisation
            image = cv2.equalizeHist(grey_img)
        else:
            # Convert to HSV
            img_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            # Histogram equalisation on V-channel
            img_hsv[:, :, 2] = cv2.equalizeHist(img_hsv[:, :, 2])
            # Convert to RGB
            image = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
    else:
        if (greyscale):
            # Image to grayscale
            grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Create and apply CLAHE
            image = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(gridsize,gridsize)).apply(grey_img)
        else:
            # Convert to HSV
            img_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            # Histogram equalisation on V-channel
            img_hsv[:, :, 2] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(gridsize,gridsize)).apply(img_hsv[:, :, 2])
            # Convert to RGB
            image = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
    os.remove("temp.jpg")
    return image

def zeroMeanOneVar(image_file_path: str):
    image = cv2.imread(image_file_path)
    # Image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Find mean and std
    mean, std_dev = cv2.meanStdDev(gray_image)
    # Normalise image
    normalized_image = (gray_image - mean) / std_dev
    return normalized_image

def zeroMeanOneVar(img: np.ndarray):
    cv2.imwrite("temp.jpg",img)
    # Read image
    image = cv2.imread("temp.jpg")
    # Image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Find mean and std
    mean, std_dev = cv2.meanStdDev(gray_image)
    # Normalise image
    normalized_image = (gray_image - mean) / std_dev
    os.remove("temp.jpg")
    return normalized_image

def minMaxScaling(image_file_path: str):
    # Read image
    image = cv2.imread(image_file_path)
    # Image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Min-Max values
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(gray_image)
    # Normalise image
    normalized_image = (gray_image - min_val) / (max_val - min_val)
    return normalized_image

def removeBackground(image_file_path: str):
    # Read image
    img = Image.open(image_file_path)
    # Remove background
    rem = rembg.remove(img)
    return rem

def removeBackground(img: np.ndarray):
    # Remove background
    output = rembg.remove(img)
    return output