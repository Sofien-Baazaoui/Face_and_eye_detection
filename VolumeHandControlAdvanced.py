import cv2
import numpy as np
import time
import HandTrackingModule as htm
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

##########################
wCam, hCam = 640, 480
##########################
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0
detector = htm.handDetector(detectionconf=0.75, Maxhands=1)

# Use pycaw library to control the volume
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
# volume.GetMute()
# volume.GetMasterVolumeLevel()
volumeRange = volume.GetVolumeRange()
minVol = volumeRange[0]
maxVol = volumeRange[1]

vol = 0
volBar = 400
volPercentage = 0
area = 0

while True:
    success, img = cap.read()

    # Find Hand
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img, draw=True)
    if len(lmList) != 0:

        # Filter based on size
        wB, hB = bbox[2] - bbox[0], bbox[3] - bbox[1]
        area = (wB * hB) // 100
        if 250 < area < 1000 :

            # Find distance between index and Thumb
            length, img, LineInfo = detector.findDistance(4, 8, img)
            # print(length)

            # Convert volume
            volBar = np.interp(length, [50, 300], [400, 150])
            volPercentage = np.interp(length, [50, 300], [0, 100])

            # Reduce resolution to make it smoother
            smoothness = 10
            volPercentage = smoothness * round(volPercentage / smoothness)

            # Check fingers up
            fingers = detector.fingersUp()
            # print(fingers)

            # If pinky is down set volume
            if not fingers[4]:
                volume.SetMasterVolumeLevelScalar(volPercentage/100, None)
                cv2.circle(img, (LineInfo[4], LineInfo[5]), 15, (0, 255, 0), cv2.FILLED)

    # Drawings
    cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
    cv2.rectangle(img, (50, int(volBar)), (85, 400), (0, 255, 0), cv2.FILLED)
    cv2.putText(img, f'{int(volPercentage)} %', (40, 450), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    # Frame rate
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (40, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cv2.imshow('Image', img)
    cv2.waitKey(1)