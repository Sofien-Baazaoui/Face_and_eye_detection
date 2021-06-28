import mediapipe as mp
import cv2
import time


class PoseEstimation():
    def __init__(self, mode=False, complexity=1, smoothness=True, detectionconf=0.5, trackingconf=0.5):
        self.mode = mode
        self.complexity = complexity
        self.smoothness = smoothness
        self.detectionconf = detectionconf
        self.trackingconf = trackingconf

        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.complexity, self.smoothness, self.detectionconf, self.trackingconf)
        self.mpDraw = mp.solutions.drawing_utils

    def findPosition(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img

    def getPosotion(self, img, draw=True):
        lmlist1 = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                # print(id, lm)
                cx, cy = int(lm.x*w), int(lm.y*h)
                lmlist1.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return lmlist1




def main():
    cap = cv2.VideoCapture(0)
    pTime = 0
    detector = PoseEstimation()
    while True:
        success, img = cap.read()
        img = detector.findPosition(img)
        lmlist1 = detector.getPosotion(img, draw=False)
        if len(lmlist1) != 0:
            print(lmlist1[14])
            cv2.circle(img, (lmlist1[14][1], lmlist1[14][2]), 15, (255, 0, 0), cv2.FILLED)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow('Image', img)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()