import os
import cv2
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify

app = Flask(__name__)

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
        gray_scaled_image = image.convert("1")
    else:
        print("Error, incorrect mode")
    # return image here. (Return model to be decided)


def save_image(img: np.ndarray, output_file_path: str) -> None:
    if img is None:
        raise ValueError("Cannot save an image that is None.")

    # Save the image
    cv2.imwrite(output_file_path, img)

def image_path_to_np_array(image_file_path: str) -> np.ndarray:
    img = cv2.imread(image_file_path)
    return img


def resize(img: np.ndarray, crop: bool = False, new_width: int = 224) -> np.ndarray:
    """
    Resize or crop an image based on the given parameters.

    This function reads an image from the specified file path and either resizes
    or crops it to the provided new width.

    Args:
        img (np.ndarray): The image file as a numpy array.
        crop (bool): Whether to crop to square or resize the image. Default is False.
        new_width (int): The new width and height of the image if not cropping. Default is 224.

    Returns:
        np.ndarray: The resized or cropped image as a NumPy array. If the image
        cannot be loaded, the function returns None.

    Raises:
        ValueError: If the new width is not a positive integer.
    """
    if crop:
        height, width = img.shape[:2]
        size = min(height, width)
        x = int((width - size) / 2)
        y = int((height - size) / 2)
        new_img = img[y:y + size, x:x + size]
    else:
        new_img = cv2.resize(img, (new_width, new_width))
    return new_img

@app.route('/api/resize', methods=['POST'])
def api_resize():
    try:
        # Log the incoming request
        print("Form data:", request.form)
        print("Files:", request.files)

        # Get the image file from the request
        file = request.files['image']
        crop = request.form.get('crop', 'false').lower() == 'true'
        new_width = int(request.form.get('new_width', 224))
        output_path = request.form.get('output_path', '')

        # Convert the file to a NumPy array
        np_img = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        # Resize the image
        resized_img = resize(img, crop, new_width)

        # Save the resized image to a file
        output_path = output_path + 'resized_image.png'
        save_image(resized_img, output_path)

        return jsonify({"message": "Image resized successfully", "output_path": output_path}), 200

    except Exception as e:
        return jsonify({"message": "An error occurred", "error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)

