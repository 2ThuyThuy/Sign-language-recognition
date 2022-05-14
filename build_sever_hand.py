from cv2 import destroyWindow, waitKey
from keras.models import load_model 
from PIL import Image
from flask import Flask, request, jsonify, make_response
import numpy as np
import cv2
import io

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
            image = request.files['image'].read()
            image = Image.open(io.BytesIO(image)).convert('RGB')
            cvImage = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
            pre_hand = cv2.resize(cvImage,(224,224))
            imgHand = Image.fromarray(pre_hand, 'RGB')
            imgHandArray = np.array(imgHand)
            imgHandArray = np.expand_dims(imgHandArray, axis=0)
            pred = model.predict(imgHandArray)[0]
            ans = labelSign[pred.argmax()]
            data['success'] = True

        except:
            ans='Lỗi'
        return jsonify({"success":str(data["success"]),"predict": ans})

    else :
        return jsonify({"success":str(data["success"]),"predict": "Lỗi không nhận được ảnh"})
        


if __name__ == "__main__":
    print("app running....")    
    model = load_model('my_model.h5')
    app.run(debug=False, host="192.168.31.81", threaded=False)
