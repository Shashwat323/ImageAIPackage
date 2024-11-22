def resize(image_file_path: str, crop: bool, width: int, height: int) -> str:
    pass

from PIL import Image
import os
import random
from PIL import Image, ImageOps, ImageEnhance
import random
import numpy as np

def overlay_images(image_path: str, overlay_path: str, transparency: float = 1.0):
    if not (0.0 <= transparency <= 1.0):
        raise ValueError('Transparency must be between 0.0 and 1.0')
    base_image = Image.open(image_path).convert('RGBA')
    overlay_image = Image.open(image_path).convert('RGBA')
    if base_image.size != overlay_image.size:
        raise ValueError('Base image and overlay image must be the same size')
    overlay_image = Image.blend(Image.new("RGBA", overlay_image.size, (0, 0, 0, 0)), overlay_image, transparency)
    result = Image.alpha_composite(base_image, overlay_image)
    return np.array(result)

def adjust_contrast(image_file_path: str, factor: float=1.0, chance: float=1.0):
    if not (0 <= chance <= 1.0):
        raise ValueError('Chance must be between 0.0 and 1.0')
    if factor < 0:
        raise ValueError('Factor must be > 0')
    image = Image.open(image_file_path)
    if random.random() > chance:
        return np.array(image)
    enhancer = ImageEnhance.Contrast(image)
    enhanced_image = enhancer.enhance(factor)
    return np.array(enhanced_image)

def adjust_hue(image_file_path: str, factor: float=1.0, chance: float=1.0):
    if not (0 <= chance <= 1.0):
        raise ValueError('Chance must be between 0.0 and 1.0')
    if factor < 0:
        raise ValueError('Factor must be > 0')
    image = Image.open(image_file_path)
    if random.random() > chance:
        return np.array(image)
    hsv_image = image.convert("HSV")
    hsv_array = np.array(hsv_image)
    # print(hsv_array.shape)  # 500=height, 600=width, 3=channels of HSV
    for i in range(hsv_array.shape[0]):		# height
        for j in range(hsv_array.shape[1]):	# width
            h, s, v = hsv_array[i, j]      	 # get hue, saturation, value
            h = int(h)
            # h is an 8 bit integer holding val's between 0-255. Convert to int.
            h = (h*factor)%255
            # hue is measured in degrees so it wraps around thus we use %
            hsv_array[i, j] = [h, s, v]  	# update the pixel
    adjusted_image = Image.fromarray(hsv_array, mode="HSV").convert("RGB")
    # converts an HSV array back to an RGB image using PIL library
    return np.array(adjusted_image)


def adjust_brightness(image_file_path: str, factor: int =1.0, chance: float=1.0):
    if not (0 <= chance <= 1.0):
        raise ValueError('Chance must be between 0.0 and 1.0')
    if factor < 0:
        raise ValueError('Factor must be > 0')
        image = Image.open(image_file_path)
    if random.random() > chance:
        return np.array(image)
    enhancer = ImageEnhance.Brightness(image)
    enhanced_image = enhancer.enhance(factor)
    return np.array(enhanced_image)


def square_rotate(image_file_path: str, mode: int=0, chance: float=1.0):
    if not (0 <= chance <= 1.0):
        raise ValueError('Chance must be between 0.0 and 1.0')
    if mode not in {0, 1, 2, 3}:
        raise ValueError('Mode must be 0, 1, 2, or 3')
    image = Image.open(image_file_path)
    if random.random() > chance:
        return np.array(image)
    angle = random.choice([90, 180, 270]) if mode == 3 else (mode+1) * 90
    rotated_image = image.rotate(angle)
    return np.array(rotated_image)


def mirror_image(image_file_path: str, chance: float=1.0):
    if chance != 1.0:
        chance = chance % 1.0
    roll = random.random()
    image = Image.open(image_file_path)
    if (roll <= chance):
        mirrored_image = ImageOps.mirror(image)
    else:
        mirrored_image = image
    #mirrored_image.show() #for testing purposes
    return np.array(mirrored_image)


def random_rotate(image_file_path: str):
    angle = random.randint(0, 360)
    image = Image.open(image_file_path)
    rotated_image = image.rotate(angle, expand=True, fillcolor=(0, 255, 0))
    rotated_image.show()
    return np.array(rotated_image)

def random_crop(image_file_path: str):
    image = Image.open(image_file_path)
    width, height = image.size
    smallest_edge = width if width < height else height
    x_min = random.randint(0, width - smallest_edge) if width > smallest_edge else 0
    y_min = random.randint(0, height - smallest_edge) if height > smallest_edge else 0
    box = (x_min, y_min, x_min + smallest_edge, y_min + smallest_edge)
    cropped_image = image.crop(box)
    #print(x_min) this code is to ensure randint() is working correctly
    #print(y_min) this code is to ensure randint() is working correctly
    cropped_image_array = np.array(cropped_image)
    #cropped_image.show()
    return cropped_image_array

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


