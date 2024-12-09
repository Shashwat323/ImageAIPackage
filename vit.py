import torch.nn as nn
from torchvision.models import vit_h_14


class ModularLinear(nn.Module):
    def __init__(self, hidden_layers, hidden_neurons, in_features, out_features=10, dropout=0.5):
        super(ModularLinear, self).__init__()
        self.layers = nn.ModuleList()
        self.initial_layer = nn.Linear(in_features=in_features, out_features=hidden_neurons)
        self.final_layer = nn.Linear(in_features=hidden_neurons, out_features=out_features)
        for _ in range(hidden_layers):
            linear_layer = nn.Linear(in_features=hidden_neurons, out_features=hidden_neurons)
            self.layers.append(linear_layer)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.gelu(self.initial_layer(x))
        for linear_layer in self.layers:
            x = self.gelu(linear_layer(x))
        x = self.gelu(self.final_layer(x))
        return x

def get_model(**kwargs):
    model = vit_h_14(weights='IMAGENET1K_SWAG_LINEAR_V1')
    heads = ModularLinear(**kwargs)
    for param in model.parameters():
        param.requires_grad = False
    model.heads = heads
    return model