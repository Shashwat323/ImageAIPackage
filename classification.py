import cv2
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.utils.data as data
import matplotlib.pyplot as plt
from tqdm import tqdm
import loader
import models

device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 64
img_channels = 3
num_classes = 10
start_lr = 0.1
num_epochs = 30
model_save_path = 'resnet101_cifar10.pth'

train_dataset = datasets.CIFAR10(root='/root/RESNET/dataSet', train=True, download=True)
train_dataset = loader.AugmentedImageDataset(train_dataset, loader.augment_cnn)
train_dataset = loader.TransformedImageDataset(train_dataset, loader.tensor)
valid_dataset = datasets.CIFAR10(root='/root/RESNET/dataSet', train=False, download=False)
valid_dataset = loader.TransformedImageDataset(valid_dataset, loader.tensor)

train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

# Function to update learning rate
def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# Training step function
def train_step(model, device, loader, optimizer, loss_fn, epoch, lr):
    model.train()
    total = 0
    correct = 0
    loss_log = []

    progress_bar = tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {epoch}/{num_epochs}")

    for i, (x, y) in progress_bar:
        x, y = x.to(device), y.to(device)

        # Forward pass
        y_hat = model(x)
        loss = loss_fn(y_hat, y)

        # Accuracy calculation
        _, predicted = torch.max(y_hat.data, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_log.append(loss.item())

        # Update progress bar
        progress_bar.set_postfix(loss=loss.item(), accuracy=100 * correct / total)

    avg_loss = sum(loss_log) / len(loss_log)
    accuracy = 100 * correct / total
    print(f'Epoch {epoch} || Training Loss: {avg_loss:.4f} || Lr: {lr} || Training Accuracy: {accuracy:.2f}%')
    return avg_loss, accuracy

# Validation step function
def validate(model, device, loader, loss_fn):
    model.eval()
    total = 0
    correct = 0
    loss_log = []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            # Forward pass
            y_hat = model(x)
            loss = loss_fn(y_hat, y)

            # Accuracy calculation
            _, predicted = torch.max(y_hat.data, 1)

            total += y.size(0)
            correct += (predicted == y).sum().item()

            loss_log.append(loss.item())

    avg_loss = sum(loss_log) / len(loss_log)
    accuracy = 100 * correct / total
    print(f'Validation Loss: {avg_loss:.4f} || Validation Accuracy: {accuracy:.2f}%')
    return avg_loss, accuracy

# Main training loop
if __name__ == "__main__":
    model = models.ResNet101(img_channels, num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=start_lr)
    loss_fn = nn.CrossEntropyLoss()

    train_losses = []
    train_accuracies = []
    valid_losses = []
    valid_accuracies = []

    for e in range(1, num_epochs + 1):
        train_loss, train_acc = train_step(model, device, train_loader, optimizer, loss_fn, e, lr=start_lr)
        valid_loss, valid_acc = validate(model, device, valid_loader, loss_fn)

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        valid_losses.append(valid_loss)
        valid_accuracies.append(valid_acc)

        # Dynamic learning rate adjustment
        if e > 5:
            if train_loss >= (sum(train_losses[-3:]) / 3):
                start_lr /= 10
                update_lr(optimizer, start_lr)


    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    # Save the trained model
    #torch.save(model.state_dict(), model_save_path)
    #print(f"Model saved to {model_save_path}")

    # Plotting training and validation loss
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, num_epochs + 1), valid_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()

    # Plotting training and validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), train_accuracies, label='Training Accuracy')
    plt.plot(range(1, num_epochs + 1), valid_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy Over Epochs')
    plt.legend()

    # Save the plots
    plt.savefig('training_validation_plots.png')
    print("Training and validation plots saved as 'training_validation_plots.png'")

    plt.show()