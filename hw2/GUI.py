# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'UI.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(1054, 799)
        self.t4D = QtWidgets.QPushButton(Form)
        self.t4D.setGeometry(QtCore.QRect(950, 80, 81, 41))
        self.t4D.setIconSize(QtCore.QSize(20, 20))
        self.t4D.setObjectName("t4D")
        self.label = QtWidgets.QLabel(Form)
        self.label.setGeometry(QtCore.QRect(910, 20, 121, 41))
        self.label.setObjectName("label")
        self.t6D = QtWidgets.QPushButton(Form)
        self.t6D.setGeometry(QtCore.QRect(950, 130, 81, 41))
        self.t6D.setIconSize(QtCore.QSize(20, 20))
        self.t6D.setObjectName("t6D")
        self.widget = MplWidget(Form)
        self.widget.setGeometry(QtCore.QRect(10, 20, 881, 761))
        self.widget.setObjectName("widget")
        self.front = QtWidgets.QLabel(Form)
        self.front.setGeometry(QtCore.QRect(900, 300, 41, 41))
        self.front.setObjectName("front")
        self.left = QtWidgets.QLabel(Form)
        self.left.setGeometry(QtCore.QRect(900, 400, 31, 31))
        self.left.setObjectName("left")
        self.right = QtWidgets.QLabel(Form)
        self.right.setGeometry(QtCore.QRect(900, 490, 41, 41))
        self.right.setObjectName("right")
        self.left_2 = QtWidgets.QLabel(Form)
        self.left_2.setGeometry(QtCore.QRect(900, 360, 31, 31))
        self.left_2.setObjectName("left_2")
        self.right_2 = QtWidgets.QLabel(Form)
        self.right_2.setGeometry(QtCore.QRect(900, 450, 41, 41))
        self.right_2.setObjectName("right_2")
        self.front_2 = QtWidgets.QLabel(Form)
        self.front_2.setGeometry(QtCore.QRect(900, 260, 41, 41))
        self.front_2.setObjectName("front_2")

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.t4D.setText(_translate("Form", "train4d"))
        self.label.setText(_translate("Form", "<html><head/><body><p><span style=\" font-size:12pt;\">choose train data</span></p></body></html>"))
        self.t6D.setText(_translate("Form", "train6d"))
        self.front.setText(_translate("Form", "<html><head/><body><p><br/></p></body></html>"))
        self.left.setText(_translate("Form", "<html><head/><body><p><br/></p></body></html>"))
        self.right.setText(_translate("Form", "<html><head/><body><p><br/></p></body></html>"))
        self.left_2.setText(_translate("Form", "<html><head/><body><p><span style=\" font-size:12pt;\">left</span></p></body></html>"))
        self.right_2.setText(_translate("Form", "<html><head/><body><p><span style=\" font-size:12pt;\">right</span></p></body></html>"))
        self.front_2.setText(_translate("Form", "<html><head/><body><p><span style=\" font-size:12pt;\">front</span></p></body></html>"))
from mplwidget import MplWidget