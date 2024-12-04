import kagglehub
import os
import shutil
import torchvision
from torchvision import transforms

def dataset_importer(dataset_name, dest_folder="dataset"):
    current_dir = os.getcwd()
    dest_path = os.path.join(current_dir, dest_folder)

    # Remove the existing dataset folder if it exists
    if os.path.exists(dest_path):
        shutil.rmtree(dest_path)

    # Create a new empty dataset folder
    os.makedirs(dest_path, exist_ok=True)

    # Download the latest version to the current directory
    dataset_path = kagglehub.dataset_download(dataset_name)
    print("Downloaded dataset to:", dataset_path)

    # Move the downloaded dataset to the destination folder
    for item in os.listdir(dataset_path):
        s = os.path.join(dataset_path, item)
        d = os.path.join(dest_path, item)
        if os.path.isdir(s):
            shutil.move(s, dest_path)
        else:
            shutil.move(s, d)

    # Clean up the now-empty intermediate directory, if needed
    if os.path.exists(dataset_path):
        shutil.rmtree(dataset_path)

    print("Dataset moved to:", dest_path)

def download_cifar10_dataset(download_path):
    """
    Downloads the CIFAR-10 dataset to the specified path.

    :param download_path: The local directory path where the dataset should be downloaded.
    """

    # Download the training set
    torchvision.datasets.CIFAR10(
        root=download_path,
        train=True,
        download=True
    )

    # Download the test set
    torchvision.datasets.CIFAR10(
        root=download_path,
        train=False,
        download=True
    )

    print(f"CIFAR-10 dataset downloaded to {download_path}")

if __name__ == "__main__":
    download_cifar10_dataset('cifar10-dataset')
    #dataset_importer("imsparsh/flowers-dataset")


