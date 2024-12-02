import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import DETECTORS


@DETECTORS.register_module()
class CustomEventPointHead(nn.Module):
    def __init__(self, input_channels=1, output_points=500):
        super(CustomEventPointHead, self).__init__()
        
        # Define CNN layers (simplified architecture)
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layer to predict the (x, y) coordinates
        self.fc = nn.Linear(128 * 64 * 64, output_points * 2)  # Output 500 points, each with 2 coordinates (x, y)
    
    def forward_train(self, x):
        # Pass through CNN layers
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        # Flatten the output for the fully connected layer
        x = x.view(x.size(0), -1)

        # Output (batch_size, 500, 2) coordinates for up to 500 event points
        x = self.fc(x)
        x = x.view(-1, 500, 2)
        
        return x
