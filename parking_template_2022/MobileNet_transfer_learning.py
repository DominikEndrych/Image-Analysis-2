#!/usr/bin/python

import sys
import cv2
import numpy as np
import math
import struct
from datetime import datetime
import glob

import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
import numpy
import CNN
import torch.optim as optim
import DatasetParking
from torchsummary import summary

from torch.utils.data import Dataset, DataLoader

import torchvision.models as models

def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype = "float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect

# Obdelnik z bodu
def four_point_transform(image, one_c):
    #https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
    
    pts = [((float(one_c[0])), float(one_c[1])),
            ((float(one_c[2])), float(one_c[3])),
            ((float(one_c[4])), float(one_c[5])),
            ((float(one_c[6])), float(one_c[7]))]
    
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(np.array(pts))
    (tl, tr, br, bl) = rect
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
	    [0, 0],
	    [maxWidth - 1, 0],
	    [maxWidth - 1, maxHeight - 1],
	    [0, maxHeight - 1]], dtype = "float32")
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    # return the warped image
    return warped


def OpenTestFile(filename):
    result = []
    with open(filename, "r") as file:
        for line in file:
            if line == "0\n":
                result.append(False)
            else: result.append(True)

    return result

# Evaluate text/real
def EvaluateResults(testValues, realValues):
    correct = 0
    all = 0
    for test, result in zip(testValues, realValues):
        if result == test: correct = correct + 1
        all = all + 1

    return correct, all

# Final evaluation in percentage
def FinalEvaluation(nMistakes, nAll):
    percentage = 100 * float(nAll - nMistakes)/float(nAll)
    return percentage/100.0


    
def main(argv):

    pkm_file = open('parking_map_python.txt', 'r')  # Nacteny mapy parkoviste
    pkm_lines = pkm_file.readlines()
    pkm_coordinates = []
   
    for line in pkm_lines:
        st_line = line.strip()
        sp_line = list(st_line.split(" "))
        pkm_coordinates.append(sp_line)
    
      
    test_images = [img for img in glob.glob("test_images/*.jpg")]
    test_files = [file for file in glob.glob("test_images/*.txt")]
    test_images.sort()
    test_files.sort()
    #print(pkm_coordinates)

    train_images_full = [cv2.imread(img) for img in glob.glob("train_images/full/*.png")]     # Reading images in grayscale
    train_images_free = [cv2.imread(img) for img in glob.glob("train_images/free/*.png")]
    labels_full = [1] * len(train_images_full)
    labels_free = [0] * len(train_images_free)

    train_imgs = train_images_full + train_images_free
    labels = labels_full + labels_free

    print("********************************************************")     
    
    #cv2.namedWindow("image", 1)     # Muzu menit velikost
    file_idx = -1                    # Index for evaluation files, just so I dont have to do some heavy changes

    # For final evaluation
    result_mistakes_sum = 0
    result_samples_sum = 0


    # ---- Training CNN --------

    transform = {
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

    trainset = DatasetParking.DatasetParking(train_imgs, labels, transform['train'])
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=2,
                                              shuffle=True, num_workers=2)

    print("Dataset loaded")
    dataset_size = len(trainloader.dataset)
    print(f"Dataset size: {dataset_size}")

    #net = CNN.Net()
    #print(net)
    PATH = './MyMobilenet.pth'

    #criterion = nn.CrossEntropyLoss()
    #optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    train = False

    if train:
        net = models.mobilenet_v2(pretrained = True)
        criterion = nn.CrossEntropyLoss()

        for name in net.parameters():
            name.requires_grad = False
            #print(f"{name}")

        num_ftrs = net.classifier[1].in_features
        net.classifier[1] = nn.Linear(num_ftrs, 2)

        net.classifier[1].requires_grad = True

        params_to_update = [param for param in net.parameters() if param.requires_grad == True]

        optimizer = optim.SGD(params_to_update, lr=0.001, momentum = 0.9)

        net.train()
        # Training loop
        for epoch in range(3):  # loop over the dataset multiple times
            print(f"Epoch: {epoch}")

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
                if i % 200 == 199:
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 200:.3f}') # puvodne misto 500 bylo 2000
                    running_loss = 0.0

        print('Finished Training')

        torch.save(net.state_dict(), PATH)
    else:
        net = CNN.Net()
        #net.load_state_dict(torch.load(PATH))
    # ---- Training CNN end ----

    net.eval()
    #print(summary(net, (3,224,224)))

    for img_name in test_images:
        file_idx = file_idx + 1

        test_file_values = OpenTestFile(test_files[file_idx])

        #image = cv2.imread(img_name, 0)
        image = cv2.imread(img_name)
        image_paint = image.copy()      # Kopie pro kresleni

        #print(img_name)
        results = []

        for coord in pkm_coordinates:
            #print("coord", coord)

            one_place = four_point_transform(image, coord)      # Jedno parkovaci misto
            one_place = cv2.resize(one_place, (80, 80))

            one_place_blur = cv2.medianBlur(one_place, 7)
            canny_img = cv2.Canny(one_place_blur, 70, 180)

            rows,cols = canny_img.shape

            pt_1 = (int(coord[0]), int(coord[1]))
            pt_2 = (int(coord[4]), int(coord[5]))
            pt_3 = (int(coord[2]), int(coord[3]))
            pt_4 = (int(coord[6]), int(coord[7]))

            place_color = (255,255,255)

            occupied = False

            #if label_predict == 1:
            #    occupied = True

            # ---------------------Testing the NET here----------------------------

            #net.eval()
            # since we're not training, we don't need to calculate the gradients for our outputs
            with torch.no_grad():
                sample = transform['val'](one_place)
                sample = sample.unsqueeze(0)

                output = net(sample)
               
                _, predicted = torch.max(output.data, 1)
                #print(f"{output} --- {predicted.data}")
                if predicted == 1:
                    occupied = True
                else: occupied = False

            if occupied: 
                place_color = (0,0,0)
                results.append(True)
            else: results.append(False)

            cv2.line(image_paint, pt_1, pt_2, place_color, 5)
            cv2.line(image_paint, pt_3, pt_4, place_color, 5)

           
            #v2.imshow("points", image_paint)
            #cv2.imshow("canny_img", canny_img)
            #cv2.imshow("one_place", one_place)
            #cv2.imshow("one_place_blur", one_place_blur)

            #cv2.waitKey(0)

        #cv2.imshow("image", image)

        # Evaluation
        res_correct, res_all = EvaluateResults(test_file_values, results)
        print(f"{res_correct} out of {res_all}")

        result_mistakes_sum += res_all - res_correct
        result_samples_sum += res_all

        cv2.imshow("points", image_paint)
        print(test_files[file_idx])
        cv2.waitKey(0)

    final_percentage = FinalEvaluation(result_mistakes_sum, result_samples_sum)
    print(f"Result: {final_percentage}")
    

if __name__ == "__main__":
   main(sys.argv[1:])     


