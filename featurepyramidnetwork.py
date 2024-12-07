import torch.nn as nn

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