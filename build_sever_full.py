from charset_normalizer import detect
from cv2 import destroyWindow, waitKey
from keras.models import load_model 
from PIL import Image
from flask import Flask, request, jsonify, make_response
import numpy as np
import cv2
import io
import mediapipe as mp


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
                        labelPred='Không phát hiện được kết quả'
                

        return labelPred



app = Flask(__name__)
global model 


labelSign = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']

@app.route("/", methods=["GET"])
def _start_api():
    return "api running:  sign language recognition ......."

@app.route("/upload", methods=["POST"])
def upload():
    data = {"success" : False}
    ans = ""
    data["predict"] = {"ans":ans}
    if request.files.get('image'):
        try :
            
            detector = HandDetect(detectionCon=0.8, maxHands=2)
            image = request.files['image'].read()
            image = Image.open(io.BytesIO(image)).convert('RGB')
            cvImage = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            ans = detector.findHands(cvImage)
        
            data['success'] = True

        except:
            ans='Lỗi'
        return jsonify({"success":str(data["success"]),"predict": ans})

    else :
        return jsonify({"success":str(data["success"]),"predict": "Lỗi không nhận được ảnh"})
        


if __name__ == "__main__":
    print("app running....")    
    #model = load_model('my_model.h5')
    app.run(debug=False, host="192.168.1.8", threaded=False)
