import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        
        # Block 1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False)  # (96,96,1) -> (96,96,32)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # (96,96,32) -> (48,48,32)

        # Block 2
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # (48,48,64) -> (24,24,64)

        # Block 3
        self.conv5 = nn.Conv2d(64, 96, kernel_size=3, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(96)
        
        self.conv6 = nn.Conv2d(96, 96, kernel_size=3, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(96)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # (24,24,96) -> (12,12,96)

        # Block 4
        self.conv7 = nn.Conv2d(96, 128, kernel_size=3, padding=1, bias=False)
        self.bn7 = nn.BatchNorm2d(128)
        
        self.conv8 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False)
        self.bn8 = nn.BatchNorm2d(128)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)  # (12,12,128) -> (6,6,128)

        # Block 5
        self.conv9 = nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False)
        self.bn9 = nn.BatchNorm2d(256)
        
        self.conv10 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.bn10 = nn.BatchNorm2d(256)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)  # (6,6,256) -> (3,3,256)

        # Block 6
        self.conv11 = nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False)
        self.bn11 = nn.BatchNorm2d(512)
        
        self.conv12 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False)
        self.bn12 = nn.BatchNorm2d(512)

        # Fully Connected Layers
        self.fc1 = nn.Linear(512 * 3 * 3, 512)  # Flattened size: 512 * 3 * 3
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(512, 30)  # Output size 30 for landmark prediction

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        
        # Block 1
        x = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.1)
        x = F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.1)
        x = self.pool1(x)

        # Block 2
        x = F.leaky_relu(self.bn3(self.conv3(x)), negative_slope=0.1)
        x = F.leaky_relu(self.bn4(self.conv4(x)), negative_slope=0.1)
        x = self.pool2(x)

        # Block 3
        x = F.leaky_relu(self.bn5(self.conv5(x)), negative_slope=0.1)
        x = F.leaky_relu(self.bn6(self.conv6(x)), negative_slope=0.1)
        x = self.pool3(x)

        # Block 4
        x = F.leaky_relu(self.bn7(self.conv7(x)), negative_slope=0.1)
        x = F.leaky_relu(self.bn8(self.conv8(x)), negative_slope=0.1)
        x = self.pool4(x)

        # Block 5
        x = F.leaky_relu(self.bn9(self.conv9(x)), negative_slope=0.1)
        x = F.leaky_relu(self.bn10(self.conv10(x)), negative_slope=0.1)
        x = self.pool5(x)

        # Block 6
        x = F.leaky_relu(self.bn11(self.conv11(x)), negative_slope=0.1)
        x = F.leaky_relu(self.bn12(self.conv12(x)), negative_slope=0.1)

        # Flatten and Fully Connected layers
        x = torch.flatten(x, 1)  # Flatten all dimensions except the batch
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

if __name__ == "__main__":
    model = CNNModel()
    x = torch.randn(1,1,96,96)
    pred = model(x)
    print(x.shape)
    print(pred.shape)