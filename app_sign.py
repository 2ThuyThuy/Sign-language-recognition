from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import sys

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1075, 743)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.btnOpenVideo = QtWidgets.QPushButton(self.centralwidget)
        self.btnOpenVideo.setGeometry(QtCore.QRect(50, 460, 151, 61))
        self.btnOpenVideo.setObjectName("btnOpenVideo")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(70, 40, 321, 281))
        self.label.setText("")
        self.label.setPixmap(QtGui.QPixmap("../data/sign/asl_alphabet_test/asl_alphabet_test/A_test.jpg"))
        self.label.setOpenExternalLinks(False)
        self.label.setObjectName("label")
        self.btnSelectImg = QtWidgets.QPushButton(self.centralwidget)
        self.btnSelectImg.setGeometry(QtCore.QRect(500, 90, 151, 61))
        self.btnSelectImg.setObjectName("btnSelectImg")
        self.btnPredictImg = QtWidgets.QPushButton(self.centralwidget)
        self.btnPredictImg.setGeometry(QtCore.QRect(500, 200, 151, 61))
        self.btnPredictImg.setObjectName("btnPredictImg")
        self.txtResult = QtWidgets.QPlainTextEdit(self.centralwidget)
        self.txtResult.setGeometry(QtCore.QRect(750, 90, 231, 71))
        self.txtResult.setObjectName("txtResult")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        
        #event click
        self.btnSelectImg.clicked.connect(self.SelectImg)



    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.btnOpenVideo.setText(_translate("MainWindow", "Mở Video"))
        self.btnSelectImg.setText(_translate("MainWindow", "Chọn Ảnh"))
        self.btnPredictImg.setText(_translate("MainWindow", "Kiểm Tra"))

    def SelectImg(self):
        linkFile = App().initUIOpen()
        if not linkFile == '':
            f = open(linkFile,"r",encoding="utf-8")
            #readFile = f.read()
            self.txtResult.setText(f.read())
            f.close()

class App(QWidget):
    def __init__(self):
        super().__init__()
        self.title = 'App'
    
    def initUIOpen(self):
        self.setWindowTitle(self.title)
        filename = self.openFileNameDialog()
        #self.show()
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
        


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
