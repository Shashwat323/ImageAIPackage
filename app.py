import os
import uuid
from io import BytesIO

import torch
from flask import Flask, request, jsonify, send_file
from ray import tune

import imageaipackage as iap
from PIL import Image
import os
import pickle
import adjustibleresnet as resnet
import json

from hyperparameteroptimizer import CIFAR10Trainer

app = Flask(__name__)
app.config['UPLOAD_FOLDER']='api-uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Map of supported preprocessing techniques
PREPROCESSING_TECHNIQUES = {
    "random_crop": iap.random_crop,
    "convert_to_grey": iap.convert_to_grey,
    "adjust_contrast": iap.adjust_contrast,
    "adjust_hue": iap.adjust_hue,
    "adjust_brightness": iap.adjust_brightness,
    "square_rotate": iap.square_rotate,
    "mirror_image": iap.mirror_image,
    "random_rotate": iap.random_rotate,
    "remove_background": iap.removeBackground,
    "histogram_equalisation": iap.histogramEqualisation,
    "normalise": iap.normalise,
    "zero_mean_one_var": iap.zeroMeanOneVar,
    "min_max_scaling": iap.minMaxScaling,
    "mean_normalisation": iap.meanNormalisation,
    "crop": iap.crop,
    "resize": iap.resize,
    "region_grow": iap.region_grow,
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
@app.route('/upload-file', methods=['POST'])
def upload_file():
    # Check if the request has a file
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['image']
    if not file:
        return jsonify({"error": "Invalid file"}), 400
    unique_id = str(uuid.uuid4())
    filename = f"{unique_id}_{file.filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    return jsonify({'message': 'File uploaded', 'file_id': unique_id})

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

@app.route('/hyperparameter-tuning', methods=['POST'])
def hyperparameter_optimiser():
    """
        API endpoint to perform hyperparameter optimization using CIFAR10Trainer class.
        """
    try:
        # Parse configuration settings from the JSON payload
        data = request.json
        if not data:
            return jsonify({"error": "No configuration provided"}), 400

        # Extract parameters or use defaults if not provided
        root = data.get("root", "D:\\Other\\Repos\\ImageAIPackage")
        batch_size = data.get("batch_size", 64)
        fraction = data.get("fraction", 1.0)
        num_samples = data.get("num_samples", 100)
        iterations = data.get("iterations", 5)
        use_progress_bar = data.get("use_progress_bar", True)

        # Initialize the trainer with the provided configurations
        trainer = CIFAR10Trainer(
            root=root,
            batch_size=batch_size,
            fraction=fraction,
            use_progress_bar=use_progress_bar
        )

        # Define search space for tuning (can come from the request or use a default)
        search_space = data.get("search_space", {
            "initial_out": tune.randint(32, 128),
            "dropout": tune.uniform(0.2, 0.5),
            "augmentations": tune.randint(5, 20),
            "lr": tune.loguniform(1e-5, 1e-2),
        })

        # Start hyperparameter tuning
        results = trainer.tune_model(
            num_samples=num_samples,
            iterations=iterations,
            search_space=search_space
        )

        # Fetch the best configuration
        best_config = results.get_best_result().config
        return jsonify({
            "message": "Hyperparameter tuning completed successfully",
            "best_config": best_config
        })

    except Exception as e:
        # Catch errors and return an appropriate message
        return jsonify({"error": f"Failed to perform hyperparameter tuning: {str(e)}"}), 500

@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "Welcome to the Image AI Package REST API!"})


if __name__ == "__main__":
    app.run(debug=True)