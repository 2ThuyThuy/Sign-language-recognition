import cv2
import mediapipe as mp
from PIL import Image
import numpy as np

class HandDetect:

    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, minTrackCon=0.5):
        self.mpHands = mp.solutions.hands
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.minTrackCon = minTrackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.mode, max_num_hands=self.maxHands,
                                        min_detection_confidence=self.detectionCon,
                                        min_tracking_confidence=self.minTrackCon)

    def findHands(self, img, draw=True, flipType=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        allHands = []
        h, w, c = img.shape
        if self.results.multi_hand_landmarks:
            for handType, handLms in zip(self.results.multi_handedness, self.results.multi_hand_landmarks):
                myHand = {}
                ## lmList
                mylmList = []
                xList = []
                yList = []

                for id, lm in enumerate(handLms.landmark):
                    px, py, pz = int(lm.x * w), int(lm.y * h), int(lm.z * w)
                    mylmList.append([px, py, pz])
                    xList.append(px)
                    yList.append(py)

                ## bbox
                xmin, xmax = min(xList), max(xList)
                ymin, ymax = min(yList), max(yList)
                boxW, boxH = xmax - xmin, ymax - ymin
                bbox = xmin, ymin, boxW, boxH
                
                allHands.append(myHand)



                ## draw
                if draw:
                    #vẽ box trên tay
                    cv2.rectangle(img, (bbox[0] - 20, bbox[1] - 20),(bbox[0] + bbox[2] + 20, bbox[1] + bbox[3] + 20),(255, 0, 255), 2)
                    cv2.putText(img, 'hand', (bbox[0] - 30, bbox[1] - 30), cv2.FONT_HERSHEY_PLAIN,2, (255, 0, 255), 2)
        if draw:
            return allHands, img
        else:
            return allHands
        



def main():
    cap = cv2.VideoCapture(0)
    detector = HandDetect(detectionCon=0.8,maxHands=2)
    while True:
        success, img = cap.read()
        hands, img = detector.findHands(img)
       
        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()