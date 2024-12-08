import torch.nn as nn

class block(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
        super(block, self).__init__()
        #self.expansion = 4
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels*self.expansion)
        #self.relu = nn.functional.relu()
        self.shortcut = nn.Sequential()
        #self.identity_downsample = identity_downsample
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(nn.Conv2d(in_channels, self.expansion*out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(self.expansion*out_channels))


    def forward(self, x):
        #identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = nn.functional.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = nn.functional.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out += self.shortcut(x)
        out = nn.functional.relu(out)
        """if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
        x += identity
        x = self.relu(x)"""
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, image_channels, num_classes):
        super(ResNet, self).__init__()
        self.in_channels = 64

        #self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        #self.relu = nn.ReLU()
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

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
        #self.fc = nn.Linear(512*4, num_classes)
        self.linear = nn.Linear(512 * block.expansion, num_classes)
        #self.softmax = nn.Softmax(dim=num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = nn.functional.relu(out)
        #x = self.maxpool(x)

        C1 = self.layer1(out)
        C2 = self.layer2(C1)
        C3 = self.layer3(C2)
        C4 = self.layer4(C3)

        #out = nn.functional.avg_pool2d(C4, 4)
        #out = out.view(out.size(0), -1)
        #out = self.linear(out)
        out = self.avgpool(C4)
        out = out.reshape(out.shape[0], -1)
        out = self.linear(out)
        #x = self.fc(x)
        #x = self.softmax(x)
        return out

    def feature_extraction(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        C1 = self.layer1(x)
        C2 = self.layer2(C1)
        C3 = self.layer3(C2)
        C4 = self.layer4(C3)
        return C1, C2, C3, C4

    def make_layer(self, block, num_residual_blocks, out_channels, stride):
        #identity_downsample = None
        strides = [stride] + [1]*(num_residual_blocks - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        """self.in_channels = out_channels*4
        for i in range(num_residual_blocks - 1):
            layers.append(block(self.in_channels, out_channels))"""
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

def ResNet50(image_channels, num_classes):
    return ResNet(block, [3, 4, 6, 3], image_channels, num_classes)


def ResNet101(image_channels, num_classes):
    return ResNet(block, [3, 4, 23, 3], image_channels, num_classes)


def ResNet152(image_channels, num_classes):
    return ResNet(block, [3, 8, 36, 3], image_channels, num_classes)