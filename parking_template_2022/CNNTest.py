import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
import numpy
import CNN
import torch.optim as optim

if __name__ == '__main__':

    transform = transforms.Compose(
        [
            #transofrms.Resize(227),
            transforms.ToTensor(),
            #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
         ])

    batch_size = 4

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    dataset_size = len(trainloader.dataset)
    print(f"Dataset size: {dataset_size}")

    #dataiter = iter(trainloader)
    #images, labels = dataiter.next()
    #cv2.namedWindow("img", 0)

    #for i, img in enumerate(images):
    #    print('label', classes[labels[i]])
    #    cv2.imshow('img', img.numpy().transpose(1,2,0))
    #    cv2.waitKey(0)
    

    net = CNN.Net()
    #print(net)
    PATH = './cifar_net.pth'

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    train = False

    if train:
        # Training loop
        for epoch in range(2):  # loop over the dataset multiple times
            print(f"Epoch: {epoch} \n")

            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                #if i % 2000 == 1999:    # print every 2000 mini-batches
                if i % 500 == 0:
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 500:.3f}') # puvodne misto 500 bylo 2000
                    running_loss = 0.0

        print('Finished Training')

        torch.save(net.state_dict(), PATH)
    else:
        net = CNN.Net()
        net.load_state_dict(torch.load(PATH))

    dataiter = iter(testloader)
    images, labels = dataiter.next()
    outputs = net(images)

    _, predicted = torch.max(outputs, 1)

    #for i, img in enumerate(images):
    #    print(classes[predicted[i]])
    #    cv2.imshow('img', img.numpy().transpose(1,2,0))
    #    cv2.waitKey(0)

    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')