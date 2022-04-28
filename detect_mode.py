import cv2
import numpy as np
from PIL import Image
from keras.models import load_model
import time
import cvzone

model = load_model('my_model.h5')
class_labels = []

cap = cv2.VideoCapture(0)

cap = cv2.VideoCapture(0)

pTime = 0
cTime = 0

while True:
  
    # Get image frame
    success, img = cap.read()
    # Find the hand and its landmarks
    hands, img = detector.findHands(img)  # with draw
    #hands = detector.findHands(img, draw=False)  # without draw

    if hands:
        # Hand 1
        lmList = None
        hand1 = hands[0]        
        fingers1 = detector.fingersUp(hand1)

        if len(hands) == 2:
            # Hand 2
            hand2 = hands[1]
            fingers2 = detector.fingersUp(hand2)

         
    # Display
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img,str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()