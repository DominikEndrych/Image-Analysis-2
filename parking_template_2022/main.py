#!/usr/bin/python

import sys
import cv2
import numpy as np
import math
import struct
from datetime import datetime
import glob

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

def Tresholding_detection(place, treshold):
    rows,cols = place.shape
    nBorder = 0                 # Number of border pixel
    for i in range(rows):
        for j in range(cols):
            pixel = place[i,j]
            if pixel > 0:
                nBorder = nBorder + 1
    
    if(nBorder > treshold):
        return True
    else: return False

def LBP_detection(place, lbp):
    label_predict, confidence_predict = lbp.predict(place)      # Prediction from LBP
    
    if label_predict == 1:
        return True
    else: return False

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

    # ---- Training LBP ----
    train_images_full = [cv2.imread(img,0) for img in glob.glob("train_images/full/*.png")]     # Reading images in grayscale
    train_images_free = [cv2.imread(img,0) for img in glob.glob("train_images/free/*.png")]
    labels_full = [1] * len(train_images_full)
    labels_free = [0] * len(train_images_free)

    #lbp.setGridX(4)
    #lbp.setGridY(4)

    train_imgs = train_images_full + train_images_free
    labels = labels_full + labels_free

    lbp = cv2.face.LBPHFaceRecognizer_create()    # It is Face Rocignizer but it works with other things
    lbp.train(train_imgs, np.array(labels))

    # ---- Training LBP end ----

    print("********************************************************")     
    
    #cv2.namedWindow("image", 1)     # Muzu menit velikost
    file_idx = -1                       # Index for evaluation files, just so I dont have to do some heavy changes

    # For final evaluation
    result_mistakes_sum = 0
    result_samples_sum = 0


    for img_name in test_images:
        file_idx = file_idx + 1

        test_file_values = OpenTestFile(test_files[file_idx])

        image = cv2.imread(img_name, 0)
        image_paint = image.copy()      # Kopie pro kresleni

        #print(img_name)
        results = []

        for coord in pkm_coordinates:
            #print("coord", coord)

            one_place = four_point_transform(image, coord)      # Jedno parkovaci misto
            one_place = cv2.resize(one_place, (80, 80))

            one_place_blur = cv2.medianBlur(one_place, 7)       # Gauss nefungoval tak dobře
            canny_img = cv2.Canny(one_place_blur, 70, 180)

            rows,cols = canny_img.shape

            # Drawing points
            #treshold = 130
            pt_1 = (int(coord[0]), int(coord[1]))
            pt_2 = (int(coord[4]), int(coord[5]))
            pt_3 = (int(coord[2]), int(coord[3]))
            pt_4 = (int(coord[6]), int(coord[7]))

            #label_predict, confidence_predict = lbp.predict(one_place)      # Prediction from LBP

            place_color = (255,255,255)

            occupied = Tresholding_detection(canny_img, 130)
            #occupied = LBP_detection(one_place, lbp)

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

        #cv2.imshow("points", image_paint)
        print(test_files[file_idx])
        #cv2.waitKey(0)

    final_percentage = FinalEvaluation(result_mistakes_sum, result_samples_sum)
    print(f"Result: {final_percentage}")

if __name__ == "__main__":
   main(sys.argv[1:])     
