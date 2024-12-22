import torch.nn as nn

class Block(nn.Module):
    expansion = 2
    def __init__(self, in_channels, out_channels, stride=1):
        super(Block, self).__init__()
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
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, image_channels, num_classes, dropout=0.5, initial_out = 64):
        super(ResNet, self).__init__()
        self.in_channels = initial_out
        self.dropout = nn.Dropout(dropout)

        self.conv1 = nn.Conv2d(image_channels, initial_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(initial_out)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        #residual layers
        #conv2
        self.layer1 = self.make_layer(block, layers[0], out_channels=initial_out, stride=1)
        #conv3
        self.layer2 = self.make_layer(block, layers[1], out_channels=initial_out*2, stride=2)
        #conv4
        self.layer3 = self.make_layer(block, layers[2], out_channels=initial_out*4, stride=2)
        #conv5
        self.layer4 = self.make_layer(block, layers[3], out_channels=initial_out*8, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(initial_out*16, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = nn.functional.relu(out)
        out = self.maxpool(out)

        C1 = self.layer1(out)
        C2 = self.layer2(C1)
        C3 = self.layer3(C2)
        C4 = self.layer4(C3)

        out = self.avgpool(C4)
        out = out.reshape(out.shape[0], -1)
        out = self.dropout(out)
        out = self.linear(out)
        return out

    def make_layer(self, block, num_residual_blocks, out_channels, stride):
        strides = [stride] + [1]*(num_residual_blocks - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)



def ResNet50(image_channels, num_classes, dropout, initial_out):
    return ResNet(Block, [3, 4, 6, 3], image_channels, num_classes, dropout, initial_out)


def ResNet101(image_channels, num_classes):
    return ResNet(Block, [3, 4, 23, 3], image_channels, num_classes)


def ResNet152(image_channels, num_classes):
    return ResNet(Block, [3, 8, 36, 3], image_channels, num_classes)