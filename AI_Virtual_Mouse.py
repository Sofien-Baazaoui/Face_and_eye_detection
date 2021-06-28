import cv2
import HandTrackingModule as htm
import numpy as np
import time
import autopy

###################
wcam, hcam = 640, 480
FrameRe = 100  # Frame Reduction
###################

cap = cv2.VideoCapture(0)
cap.set(3, wcam)
cap.set(4, hcam)
pTime = 0
detector = htm.handDetector(Maxhands=1)
wScr, hScr = autopy.screen.size()
# print(wScr, hScr)

while True:
    # 1. Find Hand Landmarks
    success, img = cap.read()
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)

    # 2. Get the tip of the index and the middle fingers
    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]
        # print(x1, y1, x2, y2)

        # 3. Check which fingers are up
        fingers = detector.fingersUp()
        # print(fingers)
        cv2.rectangle(img, (FrameRe, FrameRe), (wcam - FrameRe, hcam - FrameRe), (255, 0, 255), 2)

        # 4. Only index finger : Moving mode
        if fingers[1] == 1 and fingers[2] == 0:

            # 5. Convert coordinates
            x3 = np.interp(x1, (FrameRe, wcam - FrameRe), (0, wScr))
            y3 = np.interp(y1, (FrameRe, hcam - FrameRe), (0, hScr))

            # 6. Smoothen values

            # 7. Move mouse
            autopy.mouse.move(wScr - x3, y3)
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)

        # 8. Both index and middle fingers are up: Clicking mode
        if fingers[1] == 1 and fingers[2] == 1:
            # 9. Find distance between fingers
            length, img, Lineinfo = detector.findDistance(8, 12, img)
            print(length)
            # 10. Click mouse if distance is short
            if length < 40:
                cv2.circle(img, (Lineinfo[4], Lineinfo[5]), 15, (0, 255, 0), cv2.FILLED)
                autopy.mouse.click()

    # 11. Frame Rate
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    # 12. Display
    cv2.imshow('Image', img)
    cv2.waitKey(1)











