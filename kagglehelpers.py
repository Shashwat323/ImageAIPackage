import argparse

import kagglehub
import os
import shutil

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="", help="imsparsh/flowers-dataset, navoneel/brain-mri-images-for-brain-tumor-detection, hojjatk/mnist-dataset")
    parser.add_argument('--dest_folder', type=str, default="dataset")
    args = parser.parse_args()
    dataset_importer(args.dataset, args.dest_folder)


