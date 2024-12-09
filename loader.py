import gzip
import os
from functools import partial

import numpy as np
import idx2numpy
import cv2
import torch
import torchvision
from torchvision import transforms

import imageaipackage as iap

from PIL import Image
from matplotlib import pyplot as plt

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split


resize_to = partial(iap.resize, new_width=224)
number_resize_to = partial(iap.resize, new_width=28)
demo_resize_to = partial(iap.resize, new_width=32)
flower_resize_to = partial(iap.resize, new_width=518)

normalize = iap.TransformPipeline([
    iap.img_to_numpy_array,
    iap.crop,
    resize_to
])

augment = iap.TransformPipeline([
    iap.img_to_numpy_array,
    iap.mirror_image,
    iap.adjust_brightness,
    iap.adjust_contrast,
    iap.square_rotate
])

tensor = iap.TransformPipeline([
    iap.img_to_numpy_array,
    iap.crop,
    resize_to,
    transforms.ToTensor(),  # Converts PIL Image or NumPy ndarray to tensor and scales to [0, 1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet statistics
])

demo_tensor = iap.TransformPipeline([
    iap.img_to_numpy_array,
    iap.crop,
    demo_resize_to,
    transforms.ToTensor(),  # Converts PIL Image or NumPy ndarray to tensor and scales to [0, 1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet statistics
])

number_tensor = iap.TransformPipeline([
    iap.img_to_numpy_array,
    iap.crop,
    number_resize_to,
    iap.convert_to_grey,
    transforms.ToTensor(),  # Converts PIL Image or NumPy ndarray to tensor and scales to [0, 1]
    transforms.Normalize(mean=[0.5,], std=[0.5,])  # Normalize with ImageNet statistics
])

flower_tensor = iap.TransformPipeline([
    iap.img_to_numpy_array,
    iap.crop,
    flower_resize_to,
    transforms.ToTensor(),  # Converts PIL Image or NumPy ndarray to tensor and scales to [0, 1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet statistics
])

def flower_label_to_index(x):
    classes = ["sunflower", "dandelion", "daisy", "tulip", "rose"]

    if x in classes:
        # Return the index of the class
        return classes.index(x)
    else:
        raise ValueError(f"Unknown flower label: {x}")

def flower_index_to_label(index):
    classes = ["sunflower", "dandelion", "daisy", "tulip", "rose"]

    if 0 <= index < len(classes):
        # Return the class corresponding to the index
        return classes[index]
    else:
        raise IndexError(f"Index out of range: {index}")

def cifar_index_to_label(index):
    classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    if 0 <= index < len(classes):
        # Return the class corresponding to the index
        return classes[index]
    else:
        raise IndexError(f"Index out of range: {index}")



class UbyteImageDataset(Dataset):
    def __init__(self, root_dir, img_file, label_file, transform=None, label_transform=None):
        self.images = idx2numpy.convert_from_file(root_dir + img_file)
        self.labels = idx2numpy.convert_from_file(root_dir + label_file)
        self.transform = transform
        self.label_transform = label_transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.label_transform:
            label = self.label_transform(label)

        if self.transform:
            image = self.transform(image)

        return image, label

# the original dataset
class FolderImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, label_transform=None):
        """
        Args:
            root_dir (string): Directory with all the images organized in subfolders per class.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}
        self.transform = transform
        self.label_transform = label_transform

        # Traverse the directory to collect image paths and labels
        for idx, class_folder in enumerate(os.listdir(root_dir)):
            class_path = os.path.join(root_dir, class_folder)
            if os.path.isdir(class_path):
                self.class_to_idx[class_folder] = idx
                for image_name in os.listdir(class_path):
                    image_path = os.path.join(class_path, image_name)
                    if os.path.isfile(image_path):
                        self.image_paths.append(image_path)
                        self.labels.append(idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path)

        # Ensure image is RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')

        label = os.path.basename(os.path.dirname(image_path))

        if self.label_transform:
            label = self.label_transform(label)

        if self.transform:
            image = self.transform(image)

        return image, label

# the augmented dataset
class AugmentedImageDataset(Dataset):
    def __init__(self, original_dataset, transform = None, augmentations = 5):
        self.original_dataset = original_dataset
        self.transform = transform
        self.augmentations = augmentations

    def __len__(self):
        return len(self.original_dataset) * self.augmentations

    def __getitem__(self, idx):
        image, y = self.original_dataset[idx // self.augmentations]

        if idx % self.augmentations != 0:
            image = self.transform(image)

        return image, y

class TransformedImageDataset(Dataset):
    def __init__(self, original_dataset, transform = None, label_transform = None):
        self.original_dataset = original_dataset
        self.transform = transform
        self.label_transform = label_transform

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        image, y = self.original_dataset[idx]
        if self.transform:
            image = self.transform(image)
        if self.label_transform:
            y = self.label_transform(y)
        return image, y

# show 5 sample images
def show_sample_image(dataset):
    fig, axs = plt.subplots(nrows=1, ncols=5, figsize=(15, 3))
    for i, ax in enumerate(axs):
        image, label = dataset.__getitem__(i)
        ax.imshow(image, cmap='gray')
        ax.set_title(str(label))
        ax.axis('off')  # Hide axes
    plt.show()

# split dataset and (optionally) augment and/ or transform it for vit
def train_val_split(dataset):
    val_size = int(0.1 * len(dataset))
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    return train_dataset, val_dataset



# get dataloaders
def get_dataloaders(batch_size=16, root="", dataset_type = "default", augmentations=5, fraction=1.0):
    train_val_dataset = None
    test_dataset = None
    if dataset_type == "default":
        train_val_dataset = FolderImageDataset(root + '/dataset/train', normalize, flower_label_to_index)
        test_dataset = FolderImageDataset(root + '/dataset/test', normalize, flower_label_to_index)
        train_val_dataset = AugmentedImageDataset(train_val_dataset, augment)
        train_val_dataset = TransformedImageDataset(train_val_dataset, tensor)
        test_dataset = TransformedImageDataset(test_dataset, tensor)
    elif dataset_type == "cifar10":
        train_val_dataset = torchvision.datasets.CIFAR10(
            root=root,
            train=True,
            download=True
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root=root,
            train=False,
            download=True
        )
        train_val_dataset = TransformedImageDataset(train_val_dataset)
        test_dataset = TransformedImageDataset(test_dataset)
        train_val_dataset = AugmentedImageDataset(train_val_dataset, augment, augmentations)
        test_dataset = AugmentedImageDataset(test_dataset, augment, augmentations)
        train_val_dataset = TransformedImageDataset(train_val_dataset, tensor)
        test_dataset = TransformedImageDataset(test_dataset, tensor)

    elif dataset_type == "mnist":
        train_val_dataset = UbyteImageDataset(root, '/numbers_dataset/train-images.idx3-ubyte', '/numbers_dataset/train-labels.idx1-ubyte')
        test_dataset = UbyteImageDataset(root, '/numbers_dataset/t10k-images.idx3-ubyte', '/numbers_dataset/t10k-labels.idx1-ubyte')
        train_val_dataset = AugmentedImageDataset(train_val_dataset, augment)
        train_val_dataset = TransformedImageDataset(train_val_dataset, number_tensor)
        test_dataset = TransformedImageDataset(test_dataset, number_tensor)

    train_dataset, val_dataset = train_val_split(train_val_dataset)

    def get_fraction_indices(dataset, fraction):
        num_samples = int(len(dataset) * fraction)
        indices = torch.randperm(len(dataset))[:num_samples]
        return indices

    # Subset datasets based on fraction
    train_indices = get_fraction_indices(train_dataset, fraction)
    val_indices = get_fraction_indices(val_dataset, fraction)
    test_indices = get_fraction_indices(test_dataset, fraction)
    train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(val_dataset, val_indices)
    test_dataset = torch.utils.data.Subset(val_dataset, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,  shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,  shuffle=False)

    return train_loader, test_loader, val_loader

# for test
if __name__ == "__main__":
    #dataset = ImageDataset("D:\\Other\\Repos\\ImageAIPackage\\dataset\\train", transform=normalize, label_transform=flower_label_to_index)
    dataset = UbyteImageDataset("D:/Other/Repos/ImageAIPackage", "/numbers_dataset/train-images.idx3-ubyte", "/numbers_dataset/train-labels.idx1-ubyte")
    augmented_dataset = AugmentedImageDataset(dataset)
    #augmented_dataset = TransformedImageDataset(augmented_dataset, transform=tensor)
    show_sample_image(augmented_dataset)