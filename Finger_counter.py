import cv2
import time
import os
import HandTrackingModule as htm

wCam , hCam = 640, 480
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)


# To store the images
folderPath = 'Finger Images'
myList = os.listdir(folderPath)
print(myList)
overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    # print(f'{folderPath}/{imPath}')
    overlayList.append(image)
print(len(overlayList))
pTime = 0

detector = htm.handDetector(detectionconf=0.75)
tipIds = [4, 8, 12, 16, 20]

while True:
    success, img = cap.read()
    imgResize = cv2.resize(img, (1000, 1000))
    imgResize = detector.findHands(imgResize)
    lmList = detector.findPosition(imgResize, draw=False)
    # print(lmList)

    if len(lmList) != 0:
        fingers = []
        # Thumb
        if lmList[tipIds[0]][1] < lmList[tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        # 4 Fingers
        for id in range(1, 5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        # print(fingers)
        TotalFingers = fingers.count(1)
        print(TotalFingers)

        # It depends on the size of the image
        h, w, c = overlayList[TotalFingers].shape
        imgResize[0:h, 0:w] = overlayList[TotalFingers]

        cv2.rectangle(imgResize, (20, 225), (170, 425), (0, 255, 0), cv2.FILLED)
        cv2.putText(imgResize, str(TotalFingers), (45, 375), cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 0), 25)

    # To display the framerate
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (400, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cv2.imshow('Image', imgResize)
    cv2.waitKey(1)