import os
import uuid
from io import BytesIO

import torch
import torchvision.transforms
from flask import Flask, request, jsonify, send_file
from ray import tune
import tensorflow.keras.datasets as datasets
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score

import imageaipackage as iap
from PIL import Image
import os
import pickle
import adjustibleresnet as resnet
import json
import demo
import loader
import models.CNNmodels as cnn

import hyperparameteroptimizer

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

@app.route('/predict-image', methods=['GET'])
def predict_image():
    model_name = request.form.get('model_name')
    model_path = get_file_path(request.form.get('model_id'))
    image_path = get_file_path(request.form.get('image_id'))

    try:
        predicted = demo.test_and_show(image_path, model_path, model=model_name, to_tensor=loader.tensor, label_transform=loader.cifar_index_to_label)

        return jsonify({"message": f"Prediction: {str(predicted)}"})

    except Exception as e:
        return jsonify({"error": f"Failed to load the model: {str(e)}"}), 500

@app.route('/segment-image', methods=['GET'])
def segment_image():

    model_weights_path = get_file_path(request.form.get('model_id'))
    image_path = get_file_path(request.form.get('image_id'))

    try:
        result = demo.segment_test_and_show(image_path, model_weights_path)
        #return result
        return jsonify({"message": f"Prediction: {str(result)}"})

    except Exception as e:
        return jsonify({"error": f"Failed to load the model: {str(e)}"}), 500



# Flask route for image upload and preprocessing
@app.route('/upload-file', methods=['POST'])
def upload_file():
    # Check if the request has a file
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
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

@app.route('/cnn-training',methods=['POST'])
def trainModel():
    datasets = ["numbers","fashion_mnist", "cifar10"]
    models = ["leNet5", "alexNet", "vgg16C", "vgg16", "vgg19"]
    dataset = request.form.get("dataset")
    cnn_model = request.form.get("cnn_model")
    model_name = request.form.get("model_name")
    epo = request.form.get("epoch")
    bat = request.form.get("batch")
    if dataset not in datasets:
        return jsonify({"error": "UNDEFINED DATASET\nSupported datasets are the fashion dataset (fashion_mnist), CIFAR10 (cifar10) and the mnist numbers dataset (nubers)"}), 400
    if cnn_model not in models:
        return jsonify({"error": "UNDEFINED CNN_MODEL\nThe supported models are LeNet-5 (leNet5), AlexNet (alexNet), VGG-16 Configuration C (vgg16C), VGG-16 Configuration D (vgg16) and VGG-19 (vgg19)"}), 400
    try:
        epoch = int(epo)
    except Exception as e:
        epoch = 100
    try:
        batch = int(bat)
    except Exception as e:
        batch = 2048
    
    try:
        # Load dataset
        match dataset:
            case "numbers":
                (X_train,y_train),(X_test,y_test) = datasets.mnist.load_data()
                class_count = 10
            case "fashion_mnist":
                (X_train,y_train),(X_test,y_test) = datasets.fashion_mnist.load_data()
                class_count = 10
            case "cifar10":
                (X_train,y_train),(X_test,y_test) = datasets.cifar10.load_data()
                class_count = 10
            case _:
                print("UNDEFINED DATASET\nSupported datasets are the fashion dataset (fashion_mnist), CIFAR10 (cifar10) and the mnist numbers dataset (nubers).")
                return

        # Dataset to categorical
        y_train_cat = to_categorical(y_train, num_classes=class_count)
        y_test_cat = to_categorical(y_test, num_classes=class_count)

        if(cnn_model == "leNet5"):
            # Normalise data
            X_train_norm = X_train / 255
            X_test_norm = X_test / 255
            # Reshape data
            X_train_norm = X_train_norm.reshape(X_train_norm.shape[0], X_train_norm.shape[1], X_train_norm.shape[2], 1)
            X_test_norm = X_test_norm.reshape(X_test_norm.shape[0], X_test_norm.shape[1], X_test_norm.shape[2], 1)
        else:
            # Reshape data
            X_train_norm = X_train_norm.reshape(X_train_norm.shape[0], X_train_norm.shape[1], X_train_norm.shape[2], 3)
            X_test_norm = X_test_norm.reshape(X_test_norm.shape[0], X_test_norm.shape[1], X_test_norm.shape[2], 3)

        # Select model
        match cnn_model:
            case "leNet5":
                model = cnn.leNet5(class_count)
            case "alexNet":
                model = cnn.alexNet(class_count)
            case "vgg16C":
                model = cnn.vgg16C(class_count)
            case "vgg16":
                model = cnn.vgg16(class_count)
            case "vgg19":
                model = cnn.vgg19(class_count)
            case _:
                print("UNDEFINED CNN_MODEL\nThe supported models are LeNet-5 (leNet5), AlexNet (alexNet), VGG-16 Configuration C (vgg16C), VGG-16 Configuration D (vgg16) and VGG-19 (vgg19)")
                return

        # Compile model
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Define callback
        callback = [EarlyStopping(monitor='val_loss', patience=10)]

        # Fit the model
        model.fit(x=X_train_norm, y=y_train_cat, validation_data=(X_test_norm,y_test_cat), epochs=epoch, batch_size=batch, callbacks=callback)

        # Save model
        filename = model_name + ".keras"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        model.save(filepath)

        # Get last accuracy scores
        batch_size = 1024
        y_pred_train = to_categorical(model.predict(X_train_norm,batch_size=batch_size).argmax(axis=1), num_classes=10)
        y_pred_test = to_categorical(model.predict(X_test_norm,batch_size=batch_size).argmax(axis=1), num_classes=10)
        return jsonify({
            "message": f"Training Accuracy:{accuracy_score(y_pred_train, y_train_cat)}\nTesting Accuracy:{accuracy_score(y_pred_test, y_test_cat)}"
        })
    except Exception as e:
        return jsonify({"error": "Failed to train model"}), 400

@app.route('/predict-model',methods=['GET'])
def predictImage():
    model_name = request.form.get("model_name")
    image = request.form.get("image")
    if(".keras" not in model_name):
        model_name += ".keras"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], model_name)
    try:
        # Load model
        model = load_model(filepath)
        return jsonify({"message": f"Predicted class:{to_categorical(model.predict(image))}"})
    except Exception as e:
        return jsonify({"error": "Failed to load model or to predict class"}), 400


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
        use_progress_bar = data.get("use_progress_bar", 'True')
        search_space = data.get("search_space", None)
        num_samples = data.get("num_samples", 20)
        iterations = data.get("iterations", 2)

        # Fetch the best configuration
        best_config = hyperparameteroptimizer.main(root, batch_size, fraction, use_progress_bar, search_space, num_samples, iterations)

        return jsonify({
            "message": "Hyperparameter tuning completed successfully",
            "best_config": best_config
        })

    except Exception as e:
        # Catch errors and return an appropriate message
        return jsonify({"error": f"Failed to perform hyperparameter tuning: {str(e)}"}), 500

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