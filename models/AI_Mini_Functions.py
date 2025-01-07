import math
import matplotlib.pyplot
import numpy as np
import csv
import cv2
import matplotlib.pyplot as plt

def visualize_weights(weights):
    if len(weights.shape) != 1:
        raise ValueError("Input must be a 1D array of a nxn square")
    square_test = len(weights)
    square_test = math.sqrt(square_test)
    if square_test != int(square_test):
        raise ValueError("Input must be a 1D array of a nxn square")
    square_test = int(square_test)
    reshaped_weights = weights.reshape(square_test, square_test)
    weightsNormalized = (reshaped_weights - reshaped_weights.min()) / (reshaped_weights.max() - reshaped_weights.min())
    matplotlib.pyplot.figure(figsize=(5, 5))
    matplotlib.pyplot.imshow(weightsNormalized, cmap="RdYlGn")
    matplotlib.pyplot.show()

def exponentialDecay(learning_rate: float, epoch: int, decay_rate: float = 0.01):
    if learning_rate <= 0:
        raise ValueError("Learning rate must be above 0")
    if epoch < 0:
        raise ValueError("Epoch must be non-negative")
    return learning_rate * math.exp(-decay_rate * epoch)

def polynomialDecay(learning_rate: float, epoch: int, total_epochs: int, power: float = 2):
    if learning_rate <= 0:
        raise ValueError("Learning rate must be above 0")
    if epoch < 0:
        raise ValueError("Epoch must be non-negative")
    if total_epochs <= 0:
        raise ValueError("Total epochs must be above 0")
    return learning_rate * ((1 - epoch / total_epochs) ** power)

def stepDecay(learning_rate: float, epoch: int, step_size: int, decay_rate: float = 0.1):
    if learning_rate <= 0:
        raise ValueError("Learning rate must be above 0")
    if epoch < 0:
        raise ValueError("Epoch must be non-negative")
    if step_size <= 0:
        raise ValueError("Step size must be above 0")
    return learning_rate * (decay_rate ** (epoch // step_size))

def cosineAnnealing(learning_rate: float, epoch: int, total_epochs: int, min_lr: float = 0):
    if min_lr < 0:
        raise ValueError("Minimum learning rate must be non-negative")
    if epoch > total_epochs:
        return min_lr
    if epoch < 0:
        raise ValueError("Epoch must be non-negative")
    if total_epochs <= 0:
        raise ValueError("Total epochs must be above 0")
    cosine_term = (1 + math.cos(math.pi * epoch / total_epochs)) / 2
    return min_lr + (learning_rate - min_lr) * cosine_term

def open_csv_file(file_path: str, label: str):
    with open(file_path, 'r') as file_object:
        csv_reader = csv.reader(file_object, delimiter=',')
        first_row = next(csv_reader)
        if label not in first_row:
            raise ValueError(f"'{label}' was not found in the file.")
        label_index = first_row.index(label)
        x_values = []
        y_values = []
        for row in csv_reader:
            y_values.append(row[label_index])
            x_values.append([row[i] for i in range(len(row)) if i != label_index])
        x_values = np.array(x_values, dtype=object)
        y_values = np.array(y_values, dtype=object)
        return x_values, y_values

def Otsu_Threshold_Pipeline(image: np.ndarray):
    final = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, final = cv2.threshold(final, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    images = [image, final]
    plt.figure(figsize=(10, 5))
    for i in range(2):
        plt.subplot(1, 2, i + 1)
        plt.imshow(images[i], cmap='gray' if i == 1 else None)
    plt.tight_layout()
    plt.show()





