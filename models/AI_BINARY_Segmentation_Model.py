import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

image_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

def binary_mask_transform(mask):
    mask = np.array(mask)
    mask[mask != 15] = 0
    mask[mask == 15] = 1
    return torch.tensor(mask, dtype=torch.float32)

mask_transform = transforms.Compose([
    transforms.Resize((256, 256), interpolation=Image.NEAREST),
    transforms.Lambda(lambda mask: binary_mask_transform(mask))
])

def load_and_filter_voc_dataset(image_set, image_transform, mask_transform):
    dataset = datasets.VOCSegmentation(
        root="./data",
        year="2012",
        image_set=image_set,
        download=True,
        transform=image_transform,
        target_transform=mask_transform
    )
    filtered_data = [(image, mask) for image, mask in dataset if torch.any(mask == 1)]
    return filtered_data

train_data = load_and_filter_voc_dataset("train", image_transform, mask_transform)
val_data = load_and_filter_voc_dataset("val", image_transform, mask_transform)
trainval_data = load_and_filter_voc_dataset("trainval", image_transform, mask_transform)

all_data = train_data + val_data + trainval_data
print(f"Total images with 'person' class: {len(all_data)}")

train_loader = DataLoader(all_data, batch_size=4, shuffle=True)

class SimpleUNet(nn.Module):
    def __init__(self):
        super(SimpleUNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def train_model(self, train_loader, num_epochs, device):
        foreground_weight = 6.0
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(foreground_weight).to(device))
        optimizer = optim.Adam(self.parameters(), lr=5e-3)
        self.to(device)

        for epoch in range(num_epochs):
            self.train()
            epoch_loss = 0
            total_correct = 0
            total_pixels = 0
            foreground_correct = 0
            foreground_pixels = 0

            for images, masks in train_loader:
                images, masks = images.to(device), masks.to(device)
                masks = masks.unsqueeze(1)
                outputs = self(images)
                loss = criterion(outputs, masks)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                preds = torch.sigmoid(outputs) > 0.5
                total_correct += (preds == masks).sum().item()
                total_pixels += masks.numel()
                foreground_correct += ((preds == masks) & (masks == 1)).sum().item()
                foreground_pixels += (masks == 1).sum().item()

            epoch_accuracy = total_correct / total_pixels
            foreground_accuracy = (foreground_correct / foreground_pixels if foreground_pixels > 0 else 0.0)

            print(
                f"Epoch {epoch + 1}/{num_epochs}, "
                f"Loss: {epoch_loss:.4f}, "
                f"Accuracy: {epoch_accuracy:.4%}, "
                f"Foreground Accuracy: {foreground_accuracy:.4%}"
            )

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = SimpleUNet()
model.train_model(train_loader, num_epochs=25, device=device)
model.eval()

with torch.no_grad():
    sample_image, sample_mask = all_data[1]
    sample_image = sample_image.unsqueeze(0).to(device)
    predicted_mask = model(sample_image).squeeze(0).squeeze(0).cpu().numpy()
    binary_predicted_mask = (predicted_mask > 0.5).astype(np.uint8)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.title("Input Image")
    plt.imshow(sample_image.cpu().squeeze().permute(1, 2, 0))
    plt.subplot(1, 3, 2)
    plt.title("Label Mask")
    plt.imshow(sample_mask, cmap="gray")
    plt.subplot(1, 3, 3)
    plt.title("Predicted Mask")
    plt.imshow(binary_predicted_mask, cmap="gray")
    plt.show()
