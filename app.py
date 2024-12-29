import os
import uuid
from io import BytesIO

import torch
import torchvision.transforms
from flask import Flask, request, jsonify, send_file
from torch.cuda.nccl import unique_id

import imageaipackage as iap
from PIL import Image
import os
import pickle
import adjustibleresnet as resnet
import json
import demo
import loader

app = Flask(__name__)
app.config['UPLOAD_FOLDER']='api-uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Map of supported preprocessing techniques
PREPROCESSING_TECHNIQUES = {
    "random_crop": iap.random_crop,
    "convert_to_grey": iap.convert_to_grey,
}

@app.route('/predict-image', methods=['GET'])
def predict_image():
    if 'model_weights' not in request.files:
        return jsonify({"error": "No model weights uploaded"}), 400

    model_name_file = request.form.get('model_name')
    model_weights_file = request.files['model_weights']
    image_path = get_file_path(request.form.get('image_id'))

    if not model_weights_file:
        return jsonify({"error": "Invalid model weights file"}), 400
    if not model_name_file:
        return jsonify({"error": "Invalid model name file"}), 400

    try:

        predicted = demo.test_and_show(image_path, model_weights_file, model="resnet50", to_tensor=loader.tensor, label_transform=loader.cifar_index_to_label)

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
    unique_id = str(uuid.uuid4())
    filename = f"{unique_id}_{image_file.filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image_file.save(filepath)
    return jsonify({'message': 'Image file uploaded', 'image_id': unique_id})

@app.route('/preprocess-image', methods=['POST'])
def preprocess_image():
    # Load preprocessing technique
    technique = request.form.get('technique')
    if technique not in PREPROCESSING_TECHNIQUES:
        return jsonify({"error": f"Unknown preprocessing technique '{technique}'"}), 400

    # Open the image and preprocess it
    try:
        image_id = request.form.get('image_id')
        if not image_id:
            return jsonify({"error": "Image ID not provided"}), 400

        # Find the corresponding file
        uploaded_images = os.listdir(app.config['UPLOAD_FOLDER'])
        filepath = next(
            (
                os.path.join(app.config['UPLOAD_FOLDER'], f)
                for f in uploaded_images if f.startswith(image_id)
            ),
            None,
        )

        if not filepath:
            return jsonify({"error": f"No image found with ID '{image_id}'"}), 404
        image = Image.open(filepath)
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

def get_file_path(image_id):
    if not image_id:
        return jsonify({"error": "Image ID not provided"}), 400

    # Find the corresponding file
    uploaded_images = os.listdir(app.config['UPLOAD_FOLDER'])
    filepath = next(
        (
            os.path.join(app.config['UPLOAD_FOLDER'], f)
            for f in uploaded_images if f.startswith(image_id)
        ),
        None,
    )

    if not filepath:
        return jsonify({"error": f"No image found with ID '{image_id}'"}), 404
    return filepath

@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "Welcome to the Image AI Package REST API!"})


if __name__ == "__main__":
    app.run(debug=True)