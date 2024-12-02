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

class block(nn.Module):
    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
        super(block, self).__init__()
        self.expansion = 4
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels*self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
        x += identity
        x = self.relu(x)
        return x

class resnet(nn.Module):
    def __init__(self, block, layers, image_channels, num_classes):
        super(resnet, self).__init__()
        self.in_channels = 64

        self.conv = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        #residual layers
        #conv2
        self.layer1 = self.make_layer(block, layers[0], out_channels=64, stride=1)
        #conv3
        self.layer2 = self.make_layer(block, layers[1], out_channels=128, stride=2)
        #conv4
        self.layer3 = self.make_layer(block, layers[2], out_channels=256, stride=2)
        #conv5
        self.layer4 = self.make_layer(block, layers[3], out_channels=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512*4, num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

    def make_layer(self, block, num_residual_blocks, out_channels, stride):
        identity_downsample = None
        layers = []

        if stride != 1 or self.in_channels != out_channels * 4:
            identity_downsample = nn.Sequential(nn.Conv2d(self.in_channels, out_channels*4,
                                                          kernel_size=1, stride=stride, bias=False),
                                                nn.BatchNorm2d(out_channels*4))
        layers.append(block(self.in_channels, out_channels, identity_downsample, stride))
        self.in_channels = out_channels*4

        for i in range(num_residual_blocks - 1):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

def ResNet50(img_channels=3, num_classes=1000):
    return resnet(block, [3, 4, 6, 3], img_channels, num_classes)

def ResNet101(img_channels=3, num_classes=1000):
    return resnet(block, [3, 4, 23, 3], img_channels, num_classes)

def ResNet152(img_channels=3, num_classes=1000):
    return resnet(block, [3, 8, 36, 3], img_channels, num_classes)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 64
img_channels = 3
num_classes = 10
start_lr = 0.1
num_epochs = 50
model_save_path = 'resnet101_cifar10.pth'

# Load the dataset
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.CIFAR10(root='/root/RESNET/dataSet', train=True, download=True, transform=transform)
valid_dataset = datasets.CIFAR10(root='/root/RESNET/dataSet', train=False, download=False, transform=transform)

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
    model = ResNet101(img_channels, num_classes)#.to(device)
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