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

from torchvision.io.image import read_image
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image





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


def DrawBox(image, box, label):
    start_point = (int(box[0]), int(box[1]))
    start_point_label = (int(box[0] + 3), int(box[1] + 15))
    end_point = (int(box[2]), int(box[3]))
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.7
    color = (0,0,255)
    image = cv2.rectangle(image, start_point, end_point, color, 2)
    image = cv2.putText(image, label, start_point_label, font, fontScale, color, 2, cv2.LINE_AA)
    return image

    
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
    coco_names = [ '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase',
'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'diningtable', 
'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush' ]



    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_tresh=0.9)
    model.eval()

    preprocess = weights.transforms()

    print("Waiting for key")
    cv2.waitKey(0)
    print("Done waiting for key")

    for img_name in test_images:
        img = cv2.imread(img_name)
        batch = [preprocess(to_pil_image(img))]

        prediction = model(batch)
        predicted_labels = [coco_names[i] for i in prediction[0]['labels'].cpu().numpy()]
        predicted_boxes = prediction[0]['boxes'].detach().cpu().numpy()
        predicted_scores = prediction[0]['scores'].detach().cpu().numpy()

        for box, label, score in zip(predicted_boxes, predicted_labels, predicted_scores):
            if label == 'car':
                im = DrawBox(img, box, label)

        cv2.imshow("Parking", im)

        print(f"file: {img_name}")
        cv2.waitKey(0)

if __name__ == "__main__":
   main(sys.argv[1:])     
