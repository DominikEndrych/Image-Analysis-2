import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 11) # 3 32 3
        self.pool = nn.AvgPool2d(3, 2)
        self.conv2 = nn.Conv2d(32, 64, 5) # 32 64 3
        self.pool = nn.AvgPool2d(3, 2)
        self.fc1 = nn.LazyLinear(120)
        self.fc2 = nn.LazyLinear(84)
        self.fc3 = nn.LazyLinear(2)

        self.batchNorm8 = nn.BatchNorm2d(8)
        self.batchNorm16 = nn.BatchNorm2d(16)
        self.batchNorm64 = nn.BatchNorm2d(64)

        self.dropout = nn.Dropout(p=0.4)

    def forward(self, x):
        #x = self.batchNorm1(x)
        x = self.pool(F.relu(self.conv1(x)))
        #x = self.batchNorm1(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.batchNorm64(x)
        x = torch.flatten(x, 1)     # flatten all dimensions except batch
        x = F.relu(self.dropout(self.fc1(x)))
        x = F.relu(self.dropout(self.fc2(x)))
        x = self.fc3(x)
        return x

