import os
from functools import partial

import imageaipackage as iap

from PIL import Image
from matplotlib import pyplot as plt

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split

resize_to = partial(iap.resize, new_width=64)

augment = iap.TransformPipeline([
    iap.img_to_numpy,
    iap.mirror_image,
    iap.adjust_brightness,
    iap.adjust_contrast,
    iap.adjust_hue,
    iap.square_rotate
])

normalize = iap.TransformPipeline([
    iap.img_to_numpy,
    iap.crop,
    resize_to
])

# the original dataset
class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images organized in subfolders per class.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}
        self.transform = transform

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

        if self.transform:
            image = self.transform(image)

        return image, label

# the augmented dataset
class AugmentedImageDataset(Dataset):
    def __init__(self, original_dataset, transform = None):
        self.original_dataset = original_dataset
        self.transform = transform

    def __len__(self):
        return 5 * len(self.original_dataset)

    def __getitem__(self, idx):
        image, y = self.original_dataset[idx // 5]

        if idx % 5 != 0:
            image = self.transform(image)

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
def get_dataloaders(batch_size=16):
    train_val_dataset = ImageDataset('../dataset/train', normalize)
    train_val_dataset = AugmentedImageDataset(train_val_dataset, augment)
    test_dataset = ImageDataset('../dataset/test', normalize)

    train_dataset, val_dataset = train_val_split(train_val_dataset)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,  shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,  shuffle=False)

    return train_loader, test_loader, val_loader

# for test
if __name__ == "__main__":
    dataset = ImageDataset("D:\\Other\\Repos\\ImageAIPackage\\dataset\\train", transform=normalize)
    augmented_dataset = AugmentedImageDataset(dataset, transform=augment)
    show_sample_image(augmented_dataset)