from torch.utils.data import Dataset, DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision



mobilenet = models.mobilenet_v2(pretrained = True)

for name in mobilenet.parameters():
    name.requires_grad = False
    #print(f"{name}")

print(f"{mobilenet.classifier[1].in_features}")
num_ftrs = mobilenet.classifier[1].in_features
mobilenet.classifier[1] = nn.Linear(num_ftrs, 2)

mobilenet.classifier[1].requires_grad = True


'''
#vgg16 = models.vgg16(pretrained = True)
resnet18 = models.resnet18(pretrained = True)

# Info
for name, child in resnet18.named_children():
    print(f"{name}")

for name in resnet18.named_parameters():
    #print(f"{name}")
    pass
#print(resnet18)

# Zamrazeni pocitani
for name in resnet18.parameters():
    name.requires_grad = False

print(resnet18.fc)

num_ftrs = resnet18.fc.in_features
resnet18.fc = nn.Linear(num_ftrs, 2)

for name, param in resnet18.named_parameters():
    print("\t", name)
    if( ("fc" in name) or ("layer4" in name) ):
        param.requires_grad = True

params_to_update = [param for param in resnet18.parameters() if param.requires_grad == True]

data_transforms = {
    'train': transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

optimizer = optim.SGD(params_to_update, lr=0.001, momentum = 0.9)
'''








