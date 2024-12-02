# import libraries
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image

# class
class FlexibleModel(nn.Module):  # inherit from nn.Module
    # Why do we inherit from nn.Module? I know why we’d need to (obviously) but what changes does it do specifically
    def __init__(self, input_size, hidden_size, output_size):
        super(FlexibleModel, self).__init__()
        # what is this “super()” function doing?
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, output_size)
        # Are these functions or variables?

    def forward(self, x):
        # we should probably ensure x is an image of size input_size each time.
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out
        # Based on this we can assume the above self.l1 = nn.Linear() calls are functions?

    def train_model(self, train_loader, num_epochs, learning_rate):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        for epoch in range(num_epochs):
            for images, labels in train_loader:
                images = torch.flatten(images, start_dim=1)
                if images.shape[1] != self.l1.in_features:
                    raise ValueError('incorrect input size')
                outputs = self.forward(images)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print(f'Epoch [{epoch + 1}/{num_epochs}]')

    def test_model(self, test_loader):
        self.eval()  # Set model to evaluation mode
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images = torch.flatten(images, start_dim=1)
                outputs = self.forward(images)
                _, predicted = torch.max(outputs, 1)
                for p, l in zip(predicted, labels):
                    # How is this for loop checking if labels = predictions? What does zip function do?
                    if p == l:
                        correct += 1
                    total += 1
        accuracy = 100 * correct / total
        print(f'Test Accuracy: {accuracy}%')
        return accuracy