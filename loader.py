import os

import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.transforms import ToTensor

# the original dataset
class ImageDataset(Dataset):
    def __init__(self, csv_path, image_folder, y_col_name, transform=None):
        self.csv = pd.read_csv(csv_path)
        self.image_folder = image_folder

        # Drop the rows where the image does not exist
        images = os.listdir(image_folder)
        self.csv = self.csv[self.csv['name'].isin(images)]
        self.csv.reset_index(drop=True, inplace=True)

        self.y_col_name = y_col_name
        self.transform = transform

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_folder, self.csv.iloc[idx, 4])
        image = Image.open(image_path)

        # check the channel number
        if image.mode != 'RGB':
            image = image.convert('RGB')

        y = self.csv.loc[idx, self.y_col_name]

        if self.transform:
            image = self.transform(image)

        return image, y



# the augmented dataset
class AugmentedImageDataset(Dataset):
    def __init__(self, original_dataset, transforms=None):
        self.original_dataset = original_dataset
        self.transforms = transforms

    def __len__(self):
        return 5 * len(self.original_dataset)

    def __getitem__(self, idx):
        image, y = self.original_dataset[idx // 5]

        if self.transforms and (idx % 5 != 0):
            image = self.transforms(image)

        return image, y



# the dataset transformed (by default for vit inputs)
class TransformedDataset(Dataset):
    def __init__(self, original_dataset, transforms=None):
        self.original_dataset = original_dataset
        self.transforms = transforms

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        image, y = self.original_dataset[idx]

        if self.transforms:
            image = self.transforms(image)

        return image, y


# show 5 sample images
def show_sample_image(dataset):
    fig, axs = plt.subplots(nrows=1, ncols=5, figsize=(15, 3))
    for i, ax in enumerate(axs):
        image, label = dataset.__getitem__(i)
        ax.imshow(image.detach().cpu().permute(1, 2, 0), cmap='gray')
        ax.set_title('gt BMI: ' + str(label))
        ax.axis('off')  # Hide axes
    plt.show()

# split dataset and (optionally) augment and/ or transform it for vit
def train_val_test_split(dataset, augmented=True, vit_transformed=True, detection=None):
    val_size = int(0.1 * len(dataset))
    test_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size - test_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    if augmented:
        train_dataset = AugmentedImageDataset(train_dataset)

    if vit_transformed:
        train_dataset = TransformedDataset(train_dataset)
        val_dataset = TransformedDataset(val_dataset)
        test_dataset = TransformedDataset(test_dataset)

    return train_dataset, val_dataset, test_dataset



# get dataloaders
def get_dataloaders(batch_size=16, augmented=True, vit_transformed=True, show_sample=False, detection=None):
    image_dataset = ImageDataset('../data/data.csv', '../data/Images', 'bmi', ToTensor())
    if show_sample:
        train_dataset, val_dataset, test_dataset = train_val_test_split(image_dataset, augmented, vit_transformed, detection)
        show_sample_image(train_dataset)
    train_dataset, val_dataset, test_dataset = train_val_test_split(image_dataset, augmented, vit_transformed, detection)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,  shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,  shuffle=False)

    return train_loader, test_loader, val_loader



# for test
if __name__ == "__main__":
    get_dataloaders(augmented=False, vit_transformed=False, show_sample=True, detection="none")