import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10

import run
from models import resnet

device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 1
img_channels = 3
num_classes = 10
start_lr = 0.1
num_epochs = 60
model_save_path = 'weights/ckpt.pth'

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

train_dataset = CIFAR10(root='/root/RESNET/dataSet', train=True, download=True, transform=transform_train)
valid_dataset = CIFAR10(root='/root/RESNET/dataSet', train=False, download=False, transform=transform_test)

train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

# Function to update learning rate
def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# Main training loop
if __name__ == "__main__":
    model = resnet.ResNet50(3, 10)
    print("here")
    model.load_state_dict(torch.load('weights/ckpt.pth', weights_only=False, map_location='cpu')['net'])
    optimizer = optim.Adam(model.parameters(), lr=start_lr)
    loss_fn = nn.CrossEntropyLoss()
    print("here")
    valid_loss, valid_acc = run.validate(model, device, valid_loader, loss_fn)
    print("here")
    """train_losses = []
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
    print(f"Model saved to {model_save_path}")"""

