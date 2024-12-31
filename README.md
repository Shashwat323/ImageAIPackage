# Image AI Package REST API

## Description

The **Image AI Package REST API** is a Flask-based tool designed to facilitate various machine learning-related operations. It provides endpoints for tasks such as prediction, image segmentation, preprocessing, and hyperparameter optimization, making it an excellent tool for integrating image processing workflows.

---

## Features

1. **Image Prediction**  
   Predicts the outcome for an uploaded image using pre-trained models.

2. **Image Segmentation**  
   Performs image segmentation to identify or categorize different regions in an image.

3. **File Upload**  
   Efficiently uploads files and assigns a unique ID to manage them.

4. **Image Preprocessing**  
   Applies various image preprocessing techniques, including cropping, resizing, and normalization.

5. **Hyperparameter Tuning**  
   Optimizes hyperparameters for training machine learning models.

---

## API Endpoints

### 1. `/predict-image` (GET)
Predicts outcomes for uploaded images using pre-trained models.  
#### **Request Parameters**
- `model_name` (form data): The name of the model.  
- `model_weights` (file): The weights file for the model.  
- `image_id` (form data): Unique ID of the image.  

#### **Response**
- Success: JSON containing model prediction results.  
- Failure: JSON with error details.

---

### 2. `/segment-image` (GET)
Segments the given image into meaningful regions using a segmentation model.  
#### **Request Parameters**
- `model_weights` (form data): Path to the model weights file.  
- `image_id` (form data): Unique ID of the image.

#### **Response**
- Success: JSON containing segmentation results.  
- Failure: JSON with error details.

---

### 3. `/upload-file` (POST)
Uploads files to the server.  
#### **Request Parameters**
- `file` (file): The file to be uploaded.

#### **Response**
- Success: JSON containing the unique file ID.  
- Failure: JSON with error details.

---

### 4. `/preprocess-image` (POST)
Applies preprocessing techniques to uploaded images.  
#### **Request Parameters**
- `technique` (form data): The preprocessing technique to apply (e.g., "resize", "random_crop").  
- `image_id` (form data): Unique ID of the image.

#### **Response**
- Success: Preprocessed image returned as a downloadable file.  
- Failure: JSON with error details.

#### **Supported Preprocessing Techniques**
- `random_crop`
- `convert_to_grey`
- `adjust_contrast`
- `adjust_hue`
- `adjust_brightness`
- `square_rotate`
- `mirror_image`
- `random_rotate`
- `remove_background`
- `histogram_equalisation`
- `normalise`
- `zero_mean_one_var`
- `min_max_scaling`
- `mean_normalisation`
- `crop`
- `resize`
- `region_grow`

---

### 5. `/hyperparameter-tuning` (POST)
Performs hyperparameter tuning for CIFAR10 models using the provided configurations.  
#### **Request Body** (JSON)
- `root` (string): Root directory path.
- `batch_size` (integer): Batch size for training. Default: `64`.
- `fraction` (float): Data sampling fraction. Default: `1.0`.
- `use_progress_bar` (boolean): Whether to use a progress bar. Default: `True`.
- `search_space` (object): Hyperparameter search space configuration. Default: `None`.
- `num_samples` (integer): Number of samples to try. Default: `20`.
- `iterations` (integer): Number of iterations. Default: `2`.

#### **Response**
- Success: JSON containing the best configuration.  
- Failure: JSON with error details.

---

### 6. `/` (GET)
Homepage of the API.  
#### **Response**
- A welcome message: `"Welcome to the Image AI Package REST API!"`

---

## Setup Instructions

### Prerequisites
- Python 3.12.6 or later
- Flask
- Necessary packages listed below

### Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Start the Server
Run the Flask application:
```bash
python <app-file-name>.py
```
Ensure the server is running at `http://127.0.0.1:5000`.

---

## File Uploads
All uploaded files are stored in the directory:  
`api-uploads` (created automatically).

---

## Technologies Used
- **Flask**: For building the REST API.
- **PyTorch**: For model handling and computations.
- **Pillow (PIL)**: For image processing.
- **Ray Tune**: For hyperparameter optimization.

---

## Contact
For queries, feedback, or issues, please [contact us](mailto:jfernando020202@gmail.com).