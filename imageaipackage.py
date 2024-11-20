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

