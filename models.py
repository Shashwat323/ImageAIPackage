import argparse

import torch.nn as nn
import torch.optim as optim
from torchvision.models import vit_h_14
import torch.nn.functional as F

class ImageHead(nn.Module):
    def __init__(self):
        super(ImageHead, self).__init__()
        self.linear1 = nn.Linear(1280, 640)
        self.linear2 = nn.Linear(640, 5)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10, downsample=7, in_channels=1):
        super(SimpleCNN, self).__init__()
        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, padding=1)
        # Second convolutional layer
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        # Fully connected layer
        self.fc1 = nn.Linear(in_features=64 * downsample * downsample, out_features=128)
        # Output layer
        self.fc2 = nn.Linear(in_features=128, out_features=num_classes)

    def forward(self, x):
        # Apply first convolution with ReLU and max pooling
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        # Apply second convolution with ReLU and max pooling
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        # Flatten the output
        x = x.view(x.size(0), -1)
        # Apply the first fully connected layer with ReLU
        x = F.relu(self.fc1(x))
        # Apply the output layer
        x = self.fc2(x)
        return x

class ModularSimpleCNN(nn.Module):
    def __init__(self, num_classes=10, downsample=7, in_channels=1, hidden_neurons=128, num_conv_layers=2, expansion=2, dropout=0.5):
        super(ModularSimpleCNN, self).__init__()
        self.layers = nn.ModuleList()

        # Define convolutional layers
        prev_channels = in_channels
        out_channels = 32
        for _ in range(num_conv_layers):
            conv_layer = nn.Conv2d(in_channels=prev_channels, out_channels=out_channels, kernel_size=3, padding=1)
            self.layers.append(conv_layer)
            prev_channels = out_channels
            out_channels = out_channels * expansion

        # Define fully connected layers
        self.fc1 = nn.Linear(in_features=int(out_channels * ((32/(2**num_conv_layers))**2))//expansion, out_features=hidden_neurons)
        self.fc2 = nn.Linear(in_features=hidden_neurons, out_features=num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Apply each convolution layer with ReLU and max pooling
        for conv_layer in self.layers:
            x = F.relu(conv_layer(x))
            x = F.max_pool2d(x, 2)

        # Flatten the output
        x = x.view(x.size(0), -1)
        # Apply the first fully connected layer with ReLU
        x = F.relu(self.fc1(x))
        # Apply the output layer
        x = self.fc2(x)
        return x

def get_model(model_type="default"):
    model = None
    heads = None
    match model_type:
        case "default":
            model = vit_h_14(weights='IMAGENET1K_SWAG_E2E_V1')
            heads = ImageHead()
            for param in model.parameters():
                param.requires_grad = False
        case "number_simple_cnn":
            model = SimpleCNN()
            for param in model.parameters():
                param.requires_grad = True
        case "cifar10_simple_cnn":
            model = SimpleCNN(downsample=8, in_channels=3)
            for param in model.parameters():
                param.requires_grad = True
    model.heads = heads
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default="squeezenet", help="vit_h_14, squeezenet")
    args = parser.parse_args()
    m = get_model(args.model_type)
    print(m)
