
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
#import cv2 as cv
import torchvision
import matplotlib.pyplot as plt
from torchsummary import summary

class NetSimple(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 1, 3) # 3 32 3
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(1, 1, 3) # 3 64 3
        self.fc1 = nn.LazyLinear(8)
        self.fc2 = nn.LazyLinear(8)
        self.fc3 = nn.LazyLinear(2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = NetSimple()
tensor = torch.randn(1, 1, 28, 28)
net(tensor)

for p in net.named_parameters():
    #print(p)
    pass

n_params = 0
#for p in net.parameters():
#    if p.requires_grad:
#        n_params = n_params + p.numel()
#print(f"n_params {n_params}")

print(summary(net, (1,28,28)))

"""
prob = 0.3
dropout = nn.Dropout(p=prob)
x = torch.ones(10)
print(x)
y = dropout(x)
print(y)

def fx(x):
    return x**2 - 6*x + 1

def deriv(x):
    return 2*x - 6

localmin = 18
x = np.linspace(-14, 20, 2000)  # Generuje 2000 bodů v rozsahu -14 a 20
plt.plot(x, fx(x))
plt.plot(localmin, fx(localmin), 'ro')
plt.show()

epoch = 1000
lr = 0.01   # Jak moc se budu posouvat kazdou epochu

for i in range(epoch):
    grad = deriv(localmin)
    move = lr * grad
    localmin = localmin - move

print(f"localmin: {localmin}")


numpy_img = cv.imread("train_images/free/free_57.png")
print(numpy_img.shape)
tensor = torch.Tensor(numpy_img)    # transfer to Tensor
tensor = tensor.permute(2, 0, 1)    # Move channels to first position
tensor = tensor.unsqueeze(0)        # Add value to first position
print(f"Tensor shape: {tensor.shape}")

my_conv = nn.Conv2d(3,4,3,stride = 1, padding = 1)   # 3 kanaly, 4 kernely, 3x3 maska, 1 posun, padding aby se neorezal okraj (nemusi byt)
out = my_conv(tensor)
print(f"Shape after convolution: {out.shape}")



transform = torchvision.transforms.Compose([
         torchvision.transforms.ToPILImage(),
        torchvision.transforms.Resize(size=(120,80)),
        torchvision.transforms.ToTensor(),
    ])

numpy_img = cv.imread("train_images/free/free_57.png")
tensor = transform(numpy_img)
print(f"Tenshor shape after transform: {tensor.shape}")
"""