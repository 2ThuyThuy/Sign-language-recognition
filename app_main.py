
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import sys
import cv2
import mediapipe as mp
from keras.models import load_model 
from PIL import Image
import numpy as np

global model

class Ui_MainWindow(object):
    def __init__(self):
       
        self.labelSign = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']

    def setupUi(self, MainWindow):

        self.linkImg = ''

        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1277, 898)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.btnOpenVideo = QtWidgets.QPushButton(self.centralwidget)
        self.btnOpenVideo.setGeometry(QtCore.QRect(50, 560, 191, 91))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.btnOpenVideo.setFont(font)
        self.btnOpenVideo.setObjectName("btnOpenVideo")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(50, 30, 371, 331))
        self.label.setText("")
        self.label.setPixmap(QtGui.QPixmap("../data/sign/asl_alphabet_test/asl_alphabet_test/A_test.jpg"))
        self.label.setOpenExternalLinks(False)
        self.label.setObjectName("label")
        self.btnSelectImg = QtWidgets.QPushButton(self.centralwidget)
        self.btnSelectImg.setGeometry(QtCore.QRect(490, 70, 291, 121))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.btnSelectImg.setFont(font)
        self.btnSelectImg.setObjectName("btnSelectImg")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(1000, 80, 191, 91))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.textEdit = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit.setGeometry(QtCore.QRect(950, 190, 271, 161))
        self.textEdit.setObjectName("textEdit")
        self.btnPredict = QtWidgets.QPushButton(self.centralwidget)
        self.btnPredict.setGeometry(QtCore.QRect(490, 270, 291, 151))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.btnPredict.setFont(font)
        self.btnPredict.setObjectName("btnPredict")
        self.txtURL_Img = QtWidgets.QLabel(self.centralwidget)
        self.txtURL_Img.setGeometry(QtCore.QRect(30, 370, 441, 61))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(10)
        self.txtURL_Img.setFont(font)
        self.txtURL_Img.setText("")
        self.txtURL_Img.setObjectName("txtURL_Img")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        #even click
        self.btnSelectImg.clicked.connect(self.SelectImg)
        self.btnPredict.clicked.connect(self.PredictImg)
        self.btnOpenVideo.clicked.connect(self.OpenVideo)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.btnOpenVideo.setText(_translate("MainWindow", "Mở Video"))
        self.btnSelectImg.setText(_translate("MainWindow", "Chọn Ảnh"))
        self.label_2.setText(_translate("MainWindow", "Kết quả"))
        self.btnPredict.setText(_translate("MainWindow", "Dịch"))




    def SelectImg(self) :
        linkFile = App().initUIOpen()
        if not linkFile == '':
            self.linkImg =linkFile
            self.label.setPixmap(QtGui.QPixmap(linkFile))
            #self.textEdit.setText(linkFile)

    def PredictImg(self):
        if self.linkImg == '':
            self.textEdit.setText('Lỗi link')
        else:
            img = cv2.imread(self.linkImg)
            img = cv2.resize(img,(224,224))  
            img = np.reshape(img,[1,224,224,3])
            pre = model.predict(img)
            self.textEdit.setText(self.labelSign[np.argmax(pre[0])])

    def OpenVideo(self):
        try:
            cap = cv2.VideoCapture(0)
            detector = HandDetect(detectionCon=0.8,maxHands=2)
            while True:
                success, img = cap.read()
                hands, img = detector.findHands(img)
            
                cv2.imshow("Image", img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break
        except:
            print('lỗi')
class AnotherWindow(QWidget):
    """
    This "window" is a QWidget. If it has no parent, it
    will appear as a free-floating window as we want.
    """
    def __init__(self):
        super().__init__()
        self.title = 'Video'
        self.VBL = QVBoxLayout()

        self.FeedLabel = QLabel()
        self.VBL.addWidget(self.FeedLabel)

        self.CancelBTN = QPushButton("Cancel")
        self.CancelBTN.clicked.connect(self.CancelFeed)
        self.VBL.addWidget(self.CancelBTN)

        self.Worker1 = Worker1()

        self.Worker1.start()
        self.Worker1.ImageUpdate.connect(self.ImageUpdateSlot)
        self.setLayout(self.VBL)

    def ImageUpdateSlot(self, Image):
        self.FeedLabel.setPixmap(QPixmap.fromImage(Image))

    def CancelFeed(self):
        self.Worker1.stop()

class Worker1(QThread):
    ImageUpdate = pyqtSignal(QImage)
    def run(self):
        self.ThreadActive = True
        Capture = cv2.VideoCapture(0)
        while self.ThreadActive:
            ret, frame = Capture.read()
            if ret:
                Image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                FlippedImage = cv2.flip(Image, 1)
                ConvertToQtFormat = QImage(FlippedImage.data, FlippedImage.shape[1], FlippedImage.shape[0], QImage.Format_RGB888)
                Pic = ConvertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
                self.ImageUpdate.emit(Pic)
    def stop(self):
        self.ThreadActive = False
        self.quit()

class App(QWidget):
    def __init__(self):
        super().__init__()
        self.title = 'App'
    
    def initUIOpen(self):
        self.setWindowTitle(self.title)
        filename = self.openFileNameDialog()
        return filename

    def initUISave(self):
        self.setWindowTitle(self.title)
        filename = self.saveFileDialog()
        return filename
    
    def openFileNameDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"OpenFile", "","All Files (*);;Text Files (*.txt)", options=options)
        return fileName
    
    def saveFileDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getSaveFileName(self,"QFileDialog.getSaveFileName()","","All Files (*);;Text Files (*.txt)", options=options)
        return fileName
        

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
                        pred = model.predict(imgHandArray)[0]
                        
                        labelPred = self.labelSign[pred.argmax()]
                        #labelPred += str(100*pred[pred.argmax()])+"%"
                     
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
        


if __name__ == "__main__":
    model = load_model('my_model.h5')
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
