import argparse

import torch.nn as nn
from torchvision.models import vit_h_14
import torch.nn.functional as F
import flexiblemodel

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
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        # Second convolutional layer
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        # Fully connected layer
        self.fc1 = nn.Linear(in_features=64 * 7 * 7, out_features=128)
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

def get_model(model_type="default"):
    model = None
    heads = None
    match model_type:
        case "default":
            model = vit_h_14(weights='IMAGENET1K_SWAG_E2E_V1')
            heads = ImageHead()
            for param in model.parameters():
                param.requires_grad = False
        case "simple_cnn":
            model = SimpleCNN()
            for param in model.parameters():
                param.requires_grad = True
    model.heads = heads
    return model

class block(nn.Module):
    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
        super(block, self).__init__()
        self.expansion = 4
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels*self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample

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

        self.conv = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)
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
                                                          kernel_size=1, stride=stride, padding=0),
                                                nn.BatchNorm2d(out_channels*4))
        layers.append(block(self.in_channels, out_channels, identity_downsample, stride))
        self.in_channels = out_channels*4

        for i in range(num_residual_blocks - 1):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)


#FEATURE PYRMAID NETWORK
class FPNBlock(nn.Module):
    def __init__(self, in_channels, out_channels=256, is_highest_block=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.is_highest_block = is_highest_block

    def forward(self, x, y):
        x = self.conv(x)
        if not self.is_highest_block:
            x += nn.functional.interpolate(y, scale_factor=2, mode='bilinear', align_corners=True)
        out = self.conv2(x)
        return x, out

class FPN(nn.Module):
    def __init__(self, expansion=4, out_channels=256):
        super().__init__()
        self.P2 = FPNBlock(64*expansion, out_channels=out_channels)
        self.P3 = FPNBlock(128 * expansion, out_channels=out_channels)
        self.P4 = FPNBlock(256 * expansion, out_channels=out_channels)
        self.P5 = FPNBlock(512 * expansion, out_channels=out_channels, is_highest_block=True)
        self.P6 = nn.MaxPool2d(kernel_size=1, stride=2, padding=0)

    def forward(self, C2, C3, C4, C5):
        x, P5 = self.P5(C5, None)
        x, P4 = self.P4(C4, x)
        x, P3 = self.P3(C3, x)
        _, P2 = self.P2(C2, x)
        P6 = self.P6(P5)

        return P2, P3, P4, P5, P6


class ResnetFPN(nn.Module):
    def __init__(self, resnet=None, fpn=None):
        super().__init__()
        # Create ResNet
        self.resnet = resnet

        # Create FPN
        self.FPN = fpn

    def forward(self, x):
        C2, C3, C4, C5, _ = self.resnet(x)
        P2, P3, P4, P5, P6 = self.FPN(C2, C3, C4, C5)
        return P2, P3, P4, P5, P6


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default="squeezenet", help="vit_h_14, squeezenet")
    args = parser.parse_args()
    m = get_model(args.model_type)
    print(m)
