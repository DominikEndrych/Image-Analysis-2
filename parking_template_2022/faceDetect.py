import cv2 as cv

cap = cv.VideoCapture(0)

face_cascade = cv.CascadeClassifier("haar/haarcascade_frontalface_default.xml")     # Can use also LBP file from /lbp

while(True):
    ret, image = cap.read()
    image_paint = image.copy()
    faces_rects = face_cascade.detectMultiScale(image, 1.3, 3)  # Detect the faces

    # Iterate over faces and draw them
    for rect in faces_rects:
        cv.rectangle(image_paint, rect, (0,255,0), 8)

    cv.imshow("image", image_paint)
    cv.waitKey(2)
