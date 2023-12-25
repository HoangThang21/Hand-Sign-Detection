import cv2
from cvzone.HandTrackingModule import HandDetector

import numpy as np
import math
import time

cap = cv2.VideoCapture(0) # kết nói với cam mặt định
detector = HandDetector(maxHands=3) # tối đa 1 bàn tay
offset = 20 # khoãng cách giữa khung bà tay với hình cắt
imgSize = 300 # kích thước img cuối

folder = "./Data/C" # chọn folder 
counter = 0 # đếm sl ảnh chụp

while True: # vòng lập vô tận lấý data từ cam
    success, img = cap.read() # Success kiểu boolean đọc có thành công không, img là frame hình hảnh từ cam | sucess = true --> img là 1 numby
    hands, img = detector.findHands(img) # tìm bàn tay trong frame
    if hands: 
        hand = hands[0] # lấy thông tin về bà tay đầu trong frame
        x, y, w, h = hand['bbox'] # trả về khung chứa bàn tay.
 
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255 # tạo mãng 3 chiều numby kích thước imageSize x imageSize x 3, uint 8 bit ==> tạo ra ảnh màu trắng mỗi kênh màu tối đa là 255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset] # cắt phần frame theo boudinng box với khoảng lề là offset

        imgCropShape = imgCrop.shape # trả thog6 tin về kích thước hình ảnh cắt (h, w, số kênh màuz)

        aspectRatio = h / w # tỉ lệ chiều cao và chiều rộng của boudinng box

        if aspectRatio > 1: # kiểm tra tỉ lệ khunng hình  nếu h > w
            k = imgSize / h #tính tóann tĩ lệ k để chỉnh chiều rộng.
            wCal = math.ceil(k * w) #làm tròn tích k với w
            if imgCropShape[0] > 0 and imgCropShape[1] > 0:
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                imgResizeShape = imgResize.shape # trả về thông tin kích thước về (wcal, imgSize)
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize

        else: # h < w
            k = imgSize / w # tính k để xác định tỉn lệ cần thiết để resize thành imageSize (kích thước yêu cầu).
            hCal = math.ceil(k * h)
            if imgCropShape[0] > 0 and imgCropShape[1] > 0:
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize
        if imgCrop.shape[0] > 0 and imgCrop.shape[1] > 0:
            cv2.imshow("ImageCrop", imgCrop)
            cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(counter)

