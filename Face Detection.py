import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
pTime = 0

# To intialize the dectection
mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection()


while True:
    success, img = cap.read()

    # Before we send it to Mediapipe we have to convert the image
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)
    print(results)

    if results.detections:
        for id, detection in enumerate(results.detections):
            mpDraw.draw_detection(img, detection)
            print(id, detection)
            # # To get the score
            print(detection.score)
            # # To get in bbox coordinates
            print(detection.location_data.relative_bounding_box)
            # if we want to extract x min
            # bboxClasse = detection.location_data.relative_bounding_box
            # iw, ih, ic = img.shape
            # bbox = int(bboxClasse.xmin * iw), int(bboxClasse.ymin * ih), \
            #        int(bboxClasse.width * iw), int(bboxClasse.height * ih)
            # cv2.rectangle(img, bbox, (255, 0, 255), 2)
            # cv2.putText(img, f'{int(detection.score[0] * 100)}%', (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN, 2,
            #             (255, 0, 255), 2)

    # To display the framerate value
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2)

    cv2.imshow('image', img)
    cv2.waitKey(1)

