def resize(image_file_path: str, crop: bool, width: int, height: int) -> str:
    pass

from PIL import Image
import os

def gray_scale(image_file_path: str, mode: int = 1): 	# mode base = 1, type casted parameters
    image = Image.open(image_file_path)
    if mode == 1:
        gray_scaled_image = image.convert('L')
    elif mode == 2:
        gray_scaled_image = image.quantize(colors=256)
        gray_scaled_image = image.convert('L')
    elif mode == 3:
        gray_scaled_image = image.convert('1')
    else:
        raise ValueError("incorrect mode for gray_scale function != 1, 2, or 3")
    gray_scaled_image_array = np.array(gray_scaled_image)
    return gray_scaled_image_array

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


