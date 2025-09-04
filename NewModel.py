import torch
import torch.nn as nn
import torch.nn.functional as F

class DiscreteCenterPredictorCNN(nn.Module):
    def __init__(self, num_x_coords=48, num_y_coords=48):
        super().__init__()
        # Layer 1: Conv -> ReLU -> MaxPool
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, padding=1) # 48x48x2
        self.bn1 = nn.BatchNorm2d(2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)                                # 24x24x2

        # Layer 2: Conv -> ReLU -> MaxPool
        self.conv2 = nn.Conv2d(in_channels=2, out_channels=4, kernel_size=3, padding=1) # 24x24x4
        self.bn2 = nn.BatchNorm2d(4)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)                                # 12x12x4

        # Layer 3: Conv -> ReLU -> MaxPool
        self.conv3 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, padding=1) # 12x12x8
        self.bn3 = nn.BatchNorm2d(8)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)                                # 6x6x8

        # Layer 4: Conv -> ReLU
        self.conv4 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1) # 6x6x16
        self.bn4 = nn.BatchNorm2d(16)

        # Layer 5: Conv -> ReLU
        self.conv5 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1, stride=2) # 3x3x32
        self.bn5 = nn.BatchNorm2d(32)

        # Calculate the flattened size after the last convolutional layer
        self.flattened_size = 3 * 3 * 32

        # Layer 6: Fully Connected layers - now with two heads
        self.fc_shared = nn.Linear(self.flattened_size, 48*48 + 1) # Shared FC layer - size is 48*48 for (x,y) coords + 1 for "no defect"


    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))

        # Flatten
        x = x.view(-1, self.flattened_size)

        # Shared fully connected layer
        output = self.fc_shared(x)

        return output
