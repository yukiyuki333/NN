import sys
from PyQt5.QtWidgets import*
from PyQt5 import QtGui
from UI import Ui_Form
from Perceptron import draw
class AppWindow(QDialog):
    def __init__(self):
        super().__init__()
        self.filename=""
        self.Epoch=200
        self.Training_rate=0.01
        self.ui = Ui_Form() #新增剛剛拉的前端介面
        self.ui.setupUi(self)
        self.ui.dataset.clicked.connect(self.load_data) #選dataset
        self.ui.training.clicked.connect(self.show_plot) #畫圖
        self.ui.exit.clicked.connect(self.closeEvent)   #關視窗
        self.show() #show

    def load_data(self):
        fileName1, filetype = QFileDialog.getOpenFileName(self, "(*.txt)")
        if fileName1 != '':
            self.ui.data_name.setText(fileName1)
            self.filename=fileName1
    def show_plot(self):
        if self.ui.epoch.text()!='':
            self.Epoch=int(self.ui.epoch.text())
        if self.ui.training_rate.text()!='':
            self.Training_rate=float(self.ui.training_rate.text())
        p = draw(self.filename, self.Epoch, self.Training_rate)
        Traac,Testac,Weight=p.back()
        font = QtGui.QFont()
        font.setPointSize(16)
        self.ui.Taccuracy.setText("training accuracy: "+str(Traac))
        self.ui.Taccuracy.setFont(font)
        self.ui.Taccuracy_2.setText("test accuracy: " + str(Testac))
        self.ui.Taccuracy_2.setFont(font)
        self.ui.Taccuracy_3.setText("weight: " + str(Weight))
        self.ui.Taccuracy_3.setFont(font)

app = QApplication(sys.argv)
w = AppWindow()
w.show()
sys.exit(app.exec_())