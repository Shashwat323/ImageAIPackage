
import torch.nn as nn
from torchvision.models import vit_h_14

class ImageHead(nn.Module):
    def __init__(self):
        super(ImageHead, self).__init__()
        self.linear1 = nn.Linear(1280, 640)
        self.linear2 = nn.Linear(640, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


def get_model():
    model = vit_h_14(weights='IMAGENET1K_SWAG_E2E_V1')
    for param in model.parameters():
        param.requires_grad = False
    heads = ImageHead()
    model.heads = heads
    return model


if __name__ == "__main__":
    model = get_model()
    print(model)
