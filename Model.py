import torch
import torch.nn as nn
import torch.nn.functional as F

class DiscreteCenterPredictorCNN(nn.Module):
    def __init__(self, num_x_coords=48, num_y_coords=48):
        super().__init__()
        # Layer 1: Conv -> ReLU -> MaxPool
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1) # 48x48x32
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)                                # 24x24x32

        # Layer 2: Conv -> ReLU -> MaxPool
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1) # 24x24x64
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)                                # 12x12x64

        # Layer 3: Conv -> ReLU -> MaxPool
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1) # 12x12x128
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)                                # 6x6x128

        # Layer 4: Conv -> ReLU
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1) # 6x6x256
        self.bn4 = nn.BatchNorm2d(256)

        # Layer 5: Conv -> ReLU
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1) # 6x6x512
        self.bn5 = nn.BatchNorm2d(512)

        # Calculate the flattened size after the last convolutional layer
        self.flattened_size = 6 * 6 * 512

        # Layer 6: Fully Connected layers - now with two heads
        self.fc_shared = nn.Linear(self.flattened_size, 48*48) # Shared FC layer


    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))

        x = x.view(-1, self.flattened_size)

        output = self.fc_shared(x)

        return output
