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

        #self.toolbar = NavigationToolbar(self.canvas, self)  # 加入matplotlib 的 toolbar
        layout = QtWidgets.QVBoxLayout()  # 在視窗上增加放toolbar和畫布的地方
        #layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.canvas.ax = self.canvas.figure.add_subplot(111)
        self.canvas.ax.set_title("please choose dataset and start")
        self.setLayout(layout)
