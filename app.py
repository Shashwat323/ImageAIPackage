from io import BytesIO

import torch
from flask import Flask, request, jsonify, send_file
import imageaipackage as iap
from PIL import Image
import os
import pickle
import adjustibleresnet as resnet
import json
app = Flask(__name__)

# Map of supported preprocessing techniques
PREPROCESSING_TECHNIQUES = {
    "random_crop": iap.random_crop,
    "convert_to_grey": iap.convert_to_grey,
}

@app.route('/', methods=['GET'])
def predict_image():
    if 'model_name' not in request.files:
        return jsonify({"error": "No model uploaded"}), 400
    if 'model_weights' not in request.files:
        return jsonify({"error": "No model uploaded"}), 400
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    model_name_file = request.files['model_name']
    model_weights_file = request.files['model_weights']
    image_file = request.files['image']

    if not model_weights_file:
        return jsonify({"error": "Invalid model file"}), 400
    if not image_file:
        return jsonify({"error": "Invalid image file"}), 400
    if not model_name_file:
        return jsonify({"error": "Invalid image file"}), 400

    try:
        model_path = os.path.join("weights", model_weights_file)
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        model = None
        if model_name_file == 'resnet50_cifar10':
            model = resnet.ResNet50(3, 10).to(device)
        model.load_state_dict(torch.load(model_path, weights_only=False))

        image = Image.open(image_file)
        image_array = iap.img_to_numpy_array(image)
        image_tensor = torch.from_numpy(image_array)
        prediction = model(image_tensor)

        _, predicted = torch.max(prediction.data, 1)
        return jsonify({"message": f"Prediction: {str(predicted)}"})

    except Exception as e:
        return jsonify({"error": f"Failed to load the model: {str(e)}"}), 500


# Flask route for image upload and preprocessing
@app.route('/upload-image', methods=['POST'])
def upload_image():
    # Check if the request has a file
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_file = request.files['image']
    if not image_file:
        return jsonify({"error": "Invalid image file"}), 400

    # Load preprocessing technique
    technique = request.form.get('technique')
    if technique not in PREPROCESSING_TECHNIQUES:
        return jsonify({"error": f"Unknown preprocessing technique '{technique}'"}), 400

    # Open the image and preprocess it
    try:
        image = Image.open(image_file)
        image_array = iap.img_to_numpy_array(image)

        processed_image_array = PREPROCESSING_TECHNIQUES[technique](image_array)
        processed_image = Image.fromarray(processed_image_array)

        # Save processed image to bytes
        img_io = BytesIO()
        processed_image.save(img_io, 'JPEG')
        img_io.seek(0)

        # Send the processed image back to the client
        return send_file(img_io, mimetype='image/jpeg')
    except Exception as e:
        return jsonify({"error": f"Failed to process the image: {str(e)}"}), 500

@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "Welcome to the Image AI Package REST API!"})


if __name__ == "__main__":
    app.run(debug=True)