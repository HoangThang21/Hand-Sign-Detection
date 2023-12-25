
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model2/keras_model.h5", "Model2/labels.txt")

offset = 20
imgSize = 300

# folder = "Data/C"
counter = 0

labels = ["A", "B", "C"]

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    confidence_threshold =0.85
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        imgCropShape = imgCrop.shape

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            if wCal > 0:
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                if not imgResize.size == 0:
                    imgResizeShape = imgResize.shape
                    wGap = math.ceil((imgSize - wCal) / 2)
                    imgWhite[:, wGap:wCal + wGap] = imgResize
                    prediction, index = classifier.getPrediction(imgWhite, draw=False)
                    print("hinh ok",prediction, index)
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            if hCal > 0:
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                if not imgResize is None and imgResize.size > 0: 
                    imgResizeShape = imgResize.shape
                    hGap = math.ceil((imgSize - hCal) / 2)
                    imgWhite[hGap:hCal + hGap, :] = imgResize
                    prediction, index = classifier.getPrediction(imgWhite, draw=False)
                    print('vo',index)

        if prediction[index] >= confidence_threshold:
            label = labels[index]
        else:
            label = 'Unknown'

        cv2.rectangle(imgOutput, (x - offset, y - offset-50),
                      (x - offset+90, y - offset-50+50), (255, 0, 255), cv2.FILLED)
        # if prediction[0]<=0.0001 :
        #     cv2.putText(imgOutput, 'none', (x, y - 26),
        #                 cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
        # else :
        cv2.putText(imgOutput, label, (x, y - 26),
                    cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
        cv2.rectangle(imgOutput, (x-offset, y-offset),
                      (x + w+offset, y + h+offset), (255, 0, 255), 4)

        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", imgOutput)
    cv2.waitKey(1)
