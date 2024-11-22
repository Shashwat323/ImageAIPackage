import numpy


def resize(image_file_path: str, crop: bool, width: int, height: int) -> str:
    pass

from PIL import Image
import os

def gray_scale(image_file_path: str, mode: int = 1):
    # mode 1 = gray scale 0-255
    # mode 2 = 8 bit gray scale
    # mode 3 = black and white
    image = Image.open(image_file_path)
    if mode == 1:
        gray_scaled_image = image.convert("L")
    elif mode == 2:
        gray_scaled_image = image.quantize(colors=256)
        gray_scaled_image = image.convert("L")
    elif mode == 3:
        gray_scaled_image =image.convert("1")
    else:
        print("Error, incorrect mode")
    # return image here. (Return model to be decided)
import os

import cv2
import numpy as np

def save_image(img: np.ndarray, output_file_path: str) -> None:
    if img is None:
        raise ValueError("Cannot save an image that is None.")

    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    # Save the image
    cv2.imwrite(output_file_path, img)

def resize(image_file_path: str, crop: bool = False, new_width: int = 224) -> np.ndarray:
    """
    Resize or crop an image based on the given parameters.

    This function reads an image from the specified file path and either resizes
    or crops it to the provided new width.

    Args:
        image_file_path (str): The path to the input image file.
        crop (bool): Whether to crop to square or resize the image. Default is False.
        new_width (int): The new width and height of the image if not cropping. Default is 224.

    Returns:
        np.ndarray: The resized or cropped image as a NumPy array. If the image
        cannot be loaded, the function returns None.

    Raises:
        ValueError: If the new width is not a positive integer.
    """
    img = cv2.imread(image_file_path)
    if crop:
        height, width = img.shape[:2]
        size = min(height, width)
        x = int((width - size) / 2)
        y = int((height - size) / 2)
        new_img = img[y:y + size, x:x + size]
    else:
        new_img = cv2.resize(img, (new_width, new_width))
    return new_img

def box_blur(img_input, kernel_size=3) -> numpy.ndarray:
    """
        box_blur is a standard blur in which the current pixels value is changed to be the average of it's nxn neighbours

    Args:
        img_input(str/np.ndarray): Can be input as a np.ndarray or a file path, is used as the image to be blurred
        kernel_size(int): Is the size of the kernel used for blurring

    Returns:
        np.ndarray: The blurred image is returned as a np.ndarray

    Raises:
        ValueError: If kernel_size is not an int, or the img is not inputted as a file path or np.ndarray

    """
    img = img_to_numpy_array(img_input)
    if not isinstance(kernel_size, int):
        raise ValueError("kernel_size must be an int")
    return cv2.blur(img, (kernel_size, kernel_size))
def bilateral_blur(img_input, diameter=9, sigma_color=75, sigma_space=75) -> numpy.ndarray:
    """
    bilateral_blur is used to maintain edges whilst smoothing the image

    Args:
        img_input(str/np.ndarray): Can be input as a np.ndarray or a file path, is used as the image to be blurred
        diameter(int): Size of the kernel
        sigma_color(float): The number of colors to consider in the pixel range
        sigma_space(float): The space between the pixel and neighbouring pixels, higher value means pixels further out impact
                        the pixel value more

    Returns:
        np.ndarray: The blurred image is returned as a np.ndarray

    Raises:
        ValueError: If diameter is not an int, sigma_color or sigma_space are not a numeric value, also img_input

    """
    img = img_to_numpy_array(img_input)
    if not isinstance(diameter, int) or not isinstance(sigma_space, int) or not isinstance(sigma_color, int):
        raise ValueError("diameter, sigma_color and sigma_space must be an int")
    return cv2.bilateralFilter(img, diameter, sigma_color, sigma_space)
def gaussian_blur(img_input, kernel_size=3, sigma=3) -> numpy.ndarray:
    """
       gaussian_blur is used to favour closer neighbours more then further neighbours

       Args:
           img_input(str/np.ndarray): Can be input as a np.ndarray or a file path, is used as the image to be blurred
           kernel_size(int): Size of the kernel used for blurring
           sigma(float): The variance in the values

       Returns:
           np.ndarray: The blurred image is returned as a np.ndarray

       Raises:
           ValueError: If kernel_size is not an int or sigma is not a float, also img_input

       """
    img = img_to_numpy_array(img_input)
    if not isinstance(kernel_size, int) or not (isinstance(sigma, int) or isinstance(sigma, float)):
        raise ValueError("kernel_size must be an int and sigma must be a numeric value")
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)
def median_blur(img_input, kernel_size=3) -> numpy.ndarray:
    """
           median_blur is used to make the current pixel value the median value in all neighbours

           Args:
               img_input(str/np.ndarray): Can be input as a np.ndarray or a file path, is used as the image to be blurred
               kernel_size(int): Size of the kernel used for blurring

           Returns:
               np.ndarray: The blurred image is returned as a np.ndarray

           Raises:
               ValueError: If kernel_size is not an int, also img_input

           """
    img = img_to_numpy_array(img_input)
    if not isinstance(kernel_size, int):
        raise ValueError("kernel_size must be an int")
    return cv2.medianBlur(img, kernel_size)
def img_to_numpy_array(image_input):
    """
           img_to_numpy_array checks whether the input is a file path or a np.ndarray, if it's a file path then it
           gets converted to a np.ndarray then returned, if it's already a np.ndarray then gets returned as is.
           If it's an invalid value then a ValueError is raised and nothing is returned

           Args:
               image_input(str/np.ndarray): The image to be used

           Returns:
               np.ndarray: The image is returned as a np.ndarray

           Raises:
               ValueError: If the image_input is not a valid file path or np.ndarray

           """
    if isinstance(image_input, numpy.ndarray):
        return image_input
    elif isinstance(image_input, str):
        img = cv2.imread(image_input, cv2.IMREAD_GRAYSCALE)
        if isinstance(img, numpy.ndarray):
            return img
    if isinstance(image_input, str):
        raise ValueError("File path must be a valid file path")
    else:
        raise ValueError("Image must be either a valid file path or a numpy array")
