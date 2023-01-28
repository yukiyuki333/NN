#-*-coding:utf-8-*-
from PyQt5 import QtWidgets, uic
import sys
import numpy as np
from UI import Ui_MainWindow

import matplotlib
matplotlib.use("Qt5Agg")  # 聲明使用QT5
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QFileDialog
import sys



class MatplotlibWidget(QtWidgets.QMainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setup_control()

        self.filename = ""
        self.points = np.empty([0, 2], float)
        self.pclass = np.array([], int)
        self.allclass = np.array([], int)
        self.pred = np.array([], int)
        self.epoch = 0
        self.learning_rate = 0
        self.train_accuracy = 0
        self.test_accuracy = 0


    def reset(self):
        self.points = np.empty([0, 2], float)
        self.pclass = np.array([], int)
        self.allclass = np.array([], int)
        self.pred = np.array([], int)
        self.epoch = 0
        self.learning_rate = 0
        self.train_accuracy = 0
        self.test_accuracy = 0

    def setup_control(self):  # botton 連接加在這裡
        self.ui.dataset_button.clicked.connect(self.open_file)
        self.ui.exit_button.clicked.connect(self.exit)
        self.ui.training_button.clicked.connect(self.train_control)

    def open_file(self):
        self.filename, filetype = QFileDialog.getOpenFileName(self,
                                                              "Open file",
                                                              "./")  # start path
        self.ui.show_file_path.setText(self.filename)
        font = QtGui.QFont()
        font.setFamily("Adobe 繁黑體 Std B")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.ui.show_file_path.setFont(font)

    def start_train(self):

        with open(self.filename) as f:
            lines = f.readlines()
            for line in lines:
                x, y, d = line.split(" ")
                self.pclass = np.append(self.pclass, int(d[0]))
                self.points = np.append(self.points, np.array([[x, y]], dtype="float64"), axis=0)
                if int(d[0]) not in self.allclass:
                    self.allclass = np.append(self.allclass, int(d[0]))
        np.sort(self.allclass)

    def split_data(self):
        shuffled_indices = np.random.permutation(len(self.points))
        train_data_size = int(len(self.points) * (2 / 3))

        self.train_indices = shuffled_indices[:train_data_size]
        self.test_indices = shuffled_indices[train_data_size:]

    def train_predict(self):
        unit_step = lambda x: self.allclass[0] if x < 0 else self.allclass[1]  # 活化函數
        w = np.random.rand(len(self.points[0]))
        bias = 0

        for i in range(self.epoch):
            for j in self.train_indices:
                input_i = self.points[j]
                label = self.pclass[j]
                result = input_i * w + bias
                result = float(sum(result))
                y_pred = float(unit_step(result))
                w = w + self.learning_rate * \
                    (label - y_pred) * np.array(input_i)  # 更新權重
                bias = bias + self.learning_rate * (label - y_pred)  # 更新 bias

        return w, bias

    def test_predict(self, w, bias):  # 预测函数
        """Return class label after unit step"""
        unit_step = lambda x: self.allclass[0] if x < 0 else self.allclass[1]  # 活化函數

        for i in self.train_indices:
            input_i = self.points[i]
            result = input_i * w + bias
            result = float(sum(result))
            y_pred = float(unit_step(result))
            self.pred = np.append(self.pred, y_pred)

        for i in self.test_indices:
            input_i = self.points[i]
            result = input_i * w + bias
            result = float(sum(result))
            y_pred = float(unit_step(result))
            self.pred = np.append(self.pred, y_pred)

        return

    def train_plot(self, w, bias):
        self.ui.widget.canvas.train.cla()
        for i in range(len(self.train_indices)):
            j = self.train_indices[i]
            if self.pred[i] == 1:
                self.ui.widget.canvas.train.scatter(self.points[j, 0], self.points[j, 1], s=5, color='r')
            else:
                self.ui.widget.canvas.train.scatter(self.points[j, 0], self.points[j, 1], s=5, color='b')

            if self.pred[i] == self.pclass[j]:
                self.train_accuracy += 1

        self.train_accuracy = self.train_accuracy / len(self.train_indices)

        x1 = -bias / w[0]

        self.ui.widget.canvas.train.axline([x1, 0], slope=-w[0] / w[1], lw=3, color='k')
        self.ui.widget.canvas.train.axis(xmin=min(self.points[:, 0]) - 1, xmax=max(self.points[:, 0]) + 1)  # 設定x軸顯示範圍
        self.ui.widget.canvas.train.axis(ymin=min(self.points[:, 1]) - 1, ymax=max(self.points[:, 1]) + 1)  # 設定y軸顯示範圍
        self.ui.widget.canvas.train.set_xlabel("X")
        self.ui.widget.canvas.train.set_ylabel("Y")
        self.ui.widget.canvas.train.set_title("Train")


    def test_plot(self, w, bias):
        self.ui.widget.canvas.test.cla()
        for i in range(len(self.test_indices)):
            j = self.test_indices[i]
            if self.pred[len(self.train_indices) + i] == 1:
                self.ui.widget.canvas.test.scatter(self.points[j, 0], self.points[j, 1], s=5, color='r')
            else:
                self.ui.widget.canvas.test.scatter(self.points[j, 0], self.points[j, 1], s=5, color='b')

            if self.pred[len(self.train_indices) + i] == self.pclass[j]:
                self.test_accuracy += 1

        self.test_accuracy = self.test_accuracy / len(self.test_indices)

        x1 = -bias / w[0]

        self.ui.widget.canvas.test.axline([x1, 0], slope=-w[0] / w[1], lw=3, color='k')
        self.ui.widget.canvas.test.axis(xmin=min(self.points[:, 0]) - 1, xmax=max(self.points[:, 0]) + 1)  # 設定x軸顯示範圍
        self.ui.widget.canvas.test.axis(ymin=min(self.points[:, 1]) - 1, ymax=max(self.points[:, 1]) + 1)  # 設定y軸顯示範圍
        self.ui.widget.canvas.test.set_xlabel("X")
        self.ui.widget.canvas.test.set_ylabel("Y")
        self.ui.widget.canvas.test.set_title("Test")

    def original_plot(self, w, bias):
        self.ui.widget.canvas.original.cla()

        for i in range(len(self.points)):
            if self.pclass[i] == 1:
                self.ui.widget.canvas.original.scatter(self.points[i, 0], self.points[i, 1], s=5, color='r')
            else:
                self.ui.widget.canvas.original.scatter(self.points[i, 0], self.points[i, 1], s=5, color='b')

        x1 = -bias / w[0]

        self.ui.widget.canvas.original.axline([x1, 0], slope=-w[0] / w[1], lw=3, color='k')
        self.ui.widget.canvas.original.axis(xmin=min(self.points[:, 0]) - 1, xmax=max(self.points[:, 0]) + 1)  # 設定x軸顯示範圍
        self.ui.widget.canvas.original.axis(ymin=min(self.points[:, 1]) - 1, ymax=max(self.points[:, 1]) + 1)  # 設定y軸顯示範圍
        self.ui.widget.canvas.original.set_xlabel("X")
        self.ui.widget.canvas.original.set_ylabel("Y")
        self.ui.widget.canvas.original.set_title("Original")
        self.ui.widget.canvas.figure.subplots_adjust(wspace=0.75)


    def train_control(self):
        self.reset()
        self.epoch = self.ui.epoch_spinbox.value()
        self.learning_rate = self.ui.learning_rate_spinbox.value()

        self.start_train()
        self.split_data()
        w, bias = p.train_predict()
        self.test_predict(w, bias)
        bias *=2
        self.train_plot(w, bias)
        self.test_plot(w, bias)
        self.original_plot(w, bias)
        self.ui.widget.canvas.draw()

        self.ui.show_train_accuracy.setText(str(p.train_accuracy))
        font = QtGui.QFont()
        font.setFamily("Adobe 繁黑體 Std B")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.ui.show_train_accuracy.setFont(font)

        self.ui.show_test_accuracy.setText(str(p.test_accuracy))
        font = QtGui.QFont()
        font.setFamily("Adobe 繁黑體 Std B")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.ui.show_test_accuracy.setFont(font)

        self.ui.show_weight.setText(str(w))
        font = QtGui.QFont()
        font.setFamily("Adobe 繁黑體 Std B")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.ui.show_weight.setFont(font)

        self.ui.show_bias.setText(str(bias))
        font = QtGui.QFont()
        font.setFamily("Adobe 繁黑體 Std B")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.ui.show_bias.setFont(font)


    def exit(self):
        app.quit()



if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    p = MatplotlibWidget()
    p.show()
    sys.exit(app.exec())