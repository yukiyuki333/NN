
from PyQt5 import QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import sys
import matplotlib.pyplot as plt
import numpy as np


class MplWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)

        # 將matplotlib 帶進 Qt 接口
        self.canvas = FigureCanvas(Figure())

        self.toolbar = NavigationToolbar(self.canvas, self)  # 加入matplotlib 的 toolbar
        layout = QtWidgets.QVBoxLayout()  # 在視窗上增加放toolbar和畫布的地方
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.canvas.original = self.canvas.figure.add_subplot(131)
        self.canvas.original.set_xlabel("X")
        self.canvas.original.set_ylabel("Y")
        self.canvas.original.set_title("Original")
        self.canvas.train = self.canvas.figure.add_subplot(132)
        self.canvas.train.set_xlabel("X")
        self.canvas.train.set_ylabel("Y")
        self.canvas.train.set_title("Train")
        self.canvas.test = self.canvas.figure.add_subplot(133)
        self.canvas.test.set_xlabel("X")
        self.canvas.test.set_ylabel("Y")
        self.canvas.test.set_title("Test")
        self.canvas.figure.subplots_adjust(wspace=0.75)
        self.setLayout(layout)

