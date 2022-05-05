import cv2
import mediapipe as mp
import math
from keras.models import load_model 
from PIL import Image
import numpy as np

class HandDetect:

    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, minTrackCon=0.5):
        self.mpHands = mp.solutions.hands
        #self.mpDraw = mp.solutions.drawing_utils
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.minTrackCon = minTrackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.mode, max_num_hands=self.maxHands,
                                        min_detection_confidence=self.detectionCon,
                                        min_tracking_confidence=self.minTrackCon)
        self.model = load_model('my_model.h5')
        self.labelSign = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']


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
                cx, cy = bbox[0] + (bbox[2] // 2), \
                         bbox[1] + (bbox[3] // 2)

                myHand["lmList"] = mylmList
                myHand["bbox"] = bbox
                myHand["center"] = (cx, cy)

                if flipType:
                    if handType.classification[0].label == "Right":
                        myHand["type"] = "Left"
                    else:
                        myHand["type"] = "Right"
                else:
                    myHand["type"] = handType.classification[0].label
                allHands.append(myHand)



                if allHands :
                    cv2.rectangle(img, (bbox[0] - 20, bbox[1] - 20),
                                  (bbox[0] + bbox[2] + 20, bbox[1] + bbox[3] + 20),
                                  (255, 0, 255), 2)
                    pre_hand = img[bbox[1] - 20: bbox[1] + bbox[3] + 20,bbox[0] - 20:bbox[0] + bbox[2] + 20]
                   
                    try :
                        pre_hand = cv2.resize(pre_hand,(224,224))
                        imgHand = Image.fromarray(pre_hand, 'RGB')
                        imgHandArray = np.array(imgHand)
                        imgHandArray = np.expand_dims(imgHandArray, axis=0)
                        pred = self.model.predict(imgHandArray)[0]
                        labelPred = self.labelSign[pred.argmax()]
                    except:
                        labelPred='Lỗi'
                    # cv2.putText(img, labelPred, (bbox[0] - 30, bbox[1] - 30), cv2.FONT_HERSHEY_PLAIN,2, (255, 0, 255), 2)
                    cv2.putText(img,labelPred, (bbox[0] - 30, bbox[1] - 30), cv2.FONT_HERSHEY_PLAIN,2, (255, 0, 255), 2)
                ## draw
                if draw:
                    pass
                    #self.mpDraw.draw_landmarks(img, handLms,self.mpHands.HAND_CONNECTIONS)
                    # cv2.putText(img, myHand["type"], (bbox[0] - 30, bbox[1] - 30), cv2.FONT_HERSHEY_PLAIN,
                    #             2, (255, 0, 255), 2)
        # if len(allHands) == 0:
        #             print("Không có tay")

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