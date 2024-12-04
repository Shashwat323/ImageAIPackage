import numpy as np
import cv2
import rembg

# Image to Array.
# img = cv2.imread('Sample.png')
# Eg. call function normalise(cv2.imread('Sample.png'))

# Remove background.
# removeBackground(): Remove background of an image.

# Adjust the contrast in an image.
# histogramEqualisation() : Adjust contrast in an image using histogram equalisation

# There are four methods of normalising images.
# These will all convert the images to greyscale 
# normalise() : Between 0 and 1
# zeroMeanOneVar() : Z-Score of each pixel
# minMaxScaling() : rescale pixel values to fit in a set range (0-1 by default)
# meanNormalisation() : normalise based on the mean pixel value

def removeBackground(img: np.ndarray):
    """
    Remove background.
    Area outside will become white
    Parameter:
    - img (np.ndarray): cv2 image array in BGR colour. (cv2.imread(path_to_image_file))
    Output:
    - output (np.ndarray): Image with white background in array form
    """
    # Remove background
    output = rembg.remove(img)
    return output

# CLAHE (Contrast Limited Adaptive Histogram Equalization)
def histogramEqualisation(img: np.ndarray, greyscale: bool=False, clahe: bool=False, gridsize: int=8):
    """
    Adjust the contrast in an image using histogram equalisation.
    Parameters:
    - img (np.ndarray): cv2 image array in BGR colour. (cv2.imread(path_to_image_file))
    - greyscale (bool): Set the output to be greyscale (default False)
    - clahe (bool): Use Contrast Limited Adaptive Histogram Equalization (CLAHE) (default False)
    - gridsize (int): gridsize of CLAHE to be applied (default 8).
    Output:
    - image (np.ndarray): output image in array form. 
    """
    if (clahe == False):
        if (greyscale):
            # Image to grayscale
            grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Apply histogram equalisation
            image = cv2.equalizeHist(grey_img)
        else:
            # Convert to HSV
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            # Histogram equalisation on V-channel
            img_hsv[:, :, 2] = cv2.equalizeHist(img_hsv[:, :, 2])
            # Convert to RGB
            image = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
    else:
        if (greyscale):
            # Image to grayscale
            grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Create and apply CLAHE
            image = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(gridsize,gridsize)).apply(grey_img)
        else:
            # Convert to HSV
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            # Histogram equalisation on V-channel
            img_hsv[:, :, 2] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(gridsize,gridsize)).apply(img_hsv[:, :, 2])
            # Convert to RGB
            image = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
    return image

def normalise(img: np.ndarray,greyscale: bool=False):
    """
    Normalise pixel values to between 0 and 1.
    This will convert the image to greyscale
    Parameters:
    - img (np.ndarray): cv2 image array. (cv2.imread(path_to_image_file))
    - greyscale (bool): Is the image already in greyscale (default False).
    Output:
    - normalized_grey_image (np.ndarray): Normalised image in array form (greyscale)
    """
    if (greyscale == False):
        # Image to grayscale
        grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        grey_img = img
    # Normalize grayscale
    normalized_gray_image = cv2.normalize(grey_img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return normalized_gray_image

def zeroMeanOneVar(img: np.ndarray,greyscale: bool=False):
    """
    Get Z-score of pixel values.
    This will convert the image to greyscale
    Parameters:
    - img (np.ndarray): cv2 image array. (cv2.imread(path_to_image_file))
    - greyscale (bool): Is the image already in greyscale (default False).
    Output:
    - normalized_image (np.ndarray): Normalised image in array form (greyscale)
    """
    if (greyscale == False):
        # Image to grayscale
        grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        grey_img = img
    # Find mean and std
    mean, std_dev = cv2.meanStdDev(grey_img)
    # Normalise image
    normalized_image = (grey_img - mean) / std_dev
    return normalized_image

def minMaxScaling(img: np.ndarray,greyscale: bool=False,min: int=0, max: int=1):
    """
    Normalise pixel values by rescaling them to within a range (default to 0-1 inclusive).
    This will convert the image to greyscale
    Parameters:
    - img (np.ndarray): cv2 image array. (cv2.imread(path_to_image_file))
    - greyscale (bool): Is the image already in greyscale (default False).
    - min (int): Minimum value of the normalised data (OPTIONAL)
    - max (int): Maximum value of the normalised data (OPTIONAL)
    Output:
    - normalized_grey_image (np.ndarray): Normalised image in array form (greyscale)
    """
    if (greyscale == False):
        # Image to grayscale
        grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        grey_img = img
    # Min-Max values
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(grey_img)
    # Normalise image
    return min + ((grey_img - min_val) * (max - min)) / (max_val - min_val)

def meanNormalisation(img: np.ndarray,greyscale: bool=False):
    """
    Normalise pixel values based on the mean.
    This will convert the image to greyscale
    Parameters:
    - img (np.ndarray): cv2 image array. (cv2.imread(path_to_image_file))
    - greyscale (bool): Is the image already in greyscale (default False).
    Output:
    - normalized_grey_image (np.ndarray): Normalised image in array form (greyscale)
    """
    if (greyscale == False):
        # Image to grayscale
        grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        grey_img = img
    # Find mean and std
    mean, std_dev = cv2.meanStdDev(grey_img)
    # Min-Max values
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(grey_img)
    # Normalize grayscale
    mean_normalized = (grey_img - mean) / (max_val - min_val)
    return mean_normalized