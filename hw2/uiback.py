from PyQt5 import QtWidgets, uic
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Qt5Agg")  # 聲明使用QT5
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtCore import QTimer
from train import rbfn4d,rbfn6d
import sys
from GUI import Ui_Form
import matplotlib.patches as pc

def o_basis(data, m, sigma):  # 輸入向量轉換為徑向基底函數，m 是該群群聚中心
    difftmp =data-m
    squtmp = np.square(difftmp.sum(axis=0))
    phi = np.exp(-squtmp / (2 * (sigma) ** 2))
    return phi

class car():
    def __init__(self, cx,cy, degree):
        self.put_ini = pc.Circle((cx,cy), 3, fill=False, color='r')
        self.cx=cx
        self.cy=cy
        self.degree=degree
        self.wall1=np.array([[-6, 6], [6, 6], [6, 30], [30, 30], [18, 30], [18, 18], [-6, 18], [-6, -6]])
        self.wall2=np.array([[-3, -3], [-3, 10], [10, 10], [10, 50], [50, 50], [22, 50], [22, 22], [-3, 22]])
    def back_ini(self):
        return self.put_ini

    def GetEquation(self,p1x, p1y, p2x, p2y):
        A = p2y - p1y
        B = p1x - p2x
        C = p2x * p1y - p1x * p2y
        return A, B, C

    def GetLinesIP(self,A1, B1, C1, A2, B2, C2):
        m = A1 * B2 - A2 * B1
        if m != 0:
            x = (C2 * B1 - C1 * B2) / m
            y = (C1 * A2 - C2 * A1) / m
            ip = np.array([x, y], dtype="float64")
            return ip
        else:
            return np.array([np.infty, np.infty])
    def sensor(self, angel):#回傳該角度到牆壁最短距離
        min_distance = np.Inf
        radian_angel = np.array((self.degree-angel)*np.pi/180)  # 轉成弧度以計算三角函數
        cx2 = self.cx+3*round(np.cos(radian_angel), 6)
        cy2 = self.cy+3*round(np.sin(radian_angel), 6)
        cp1=np.array([self.cx,self.cy])
        cp2=np.array([cx2,cy2])
        a1, b1, c1 = self.GetEquation(self.cx,self.cy,cx2,cy2)
        for i in range(len(self.wall1)):
            a2,b2,c2=self.GetEquation(self.wall1[i][0],self.wall1[i][1],self.wall2[i][0],self.wall2[i][1])
            temp = self.GetLinesIP(a1, b1, c1,a2,b2,c2)
            d1=np.linalg.norm(temp-self.wall1[i])+np.linalg.norm(temp-self.wall2[i])
            d2=np.linalg.norm(self.wall1[i]-self.wall2[i])
            dir_d1=np.linalg.norm(temp-cp2)+np.linalg.norm(cp2-cp1)
            dir_d2 = np.linalg.norm(temp - cp1)
            if (temp[0] != np.infty and temp[1] != np.infty) and min_distance>np.linalg.norm(cp1-temp) \
                    and (round(d1, 6)-round(d2, 6))<=0.01 and round(dir_d1, 6) >= round(dir_d2, 6):
                min_distance = round(np.linalg.norm(cp1-temp),6)#取道小數後第6位

        return min_distance
    def carmove(self,dir_degree):
        theta=dir_degree*np.pi/180
        Theta=self.degree*np.pi/180
        self.cx=self.cx+np.cos(theta+Theta)+np.sin(theta)*np.sin(Theta)
        self.cy=self.cy+np.sin(theta+Theta)-np.sin(theta)*np.cos(Theta)
        Theta=Theta-np.arcsin(2*np.sin(theta)/6)
        self.degree=Theta*180/np.pi
        return self.cx,self.cy,self.degree



class MatplotlibWidget(QtWidgets.QMainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        self.ui.t4D.clicked.connect(self.t4)
        self.ui.t6D.clicked.connect(self.t6)
        self.w,self.c,self.s=np.array([]),np.array([]),0
        self.mytimer=QTimer(self)
        self.stopflag=0
        self.mytimer.timeout.connect(self.car_run)
        self.choose_train=0
    def t4(self):
        self.choose_train=4
        self.init1()
        model=rbfn4d()
        self.w,self.c,self.s=model.train()
        self.car_run()
        self.mytimer.start(100)
    def t6(self):
        self.choose_train=6
        self.init1()
        model = rbfn6d()
        self.w,self.c,self.s=model.train()
        #print("model OK")
        self.car_run()
        self.mytimer.start(100)
    def init1(self):
        self.mytimer.stop()
        self.stopflag = 0
        # 手工連線
        self.x=[[-6, 6], [6, 6], [6, 30], [30, 30], [18, 30], [18, 18], [-6, 18], [-6, -6]]
        self.y=[[-3, -3], [-3, 10], [10, 10], [10, 50], [50, 50], [22, 50], [22, 22], [-3, 22]]
        #初始化一些參數
        self.cx, self.cy, self.cdegree = 0, 0, 90
        self.sensorline_x,self.sensorline_ymin,self.sensorline_ymax=0,-3,5
        self.sen = np.empty([0, 3], float)
        self.deg = np.array([])
        self.xy = np.empty([0, 2], float)
        self.xy=np.append(self.xy,[0,0])

    def car_run(self):
        if self.cx >= 18 and self.cx <= 30 and self.cy >= 37 and self.cy <= 40:
            self.stopflag = 1
        if (self.cx <= -20 or self.cx >= 45) or (self.cy <= -10 or self.cy >= 55):
            self.stopflag = 1
        if self.stopflag!=1:
            front_dis = car(self.cx, self.cy, self.cdegree).sensor(0)
            left_dis = car(self.cx, self.cy, self.cdegree).sensor(45)
            right_dis = car(self.cx, self.cy, self.cdegree).sensor(-45)
            sentmp=np.array([front_dis,left_dis,right_dis])
            sentmp6=np.array([self.cx,self.cy,front_dis,left_dis,right_dis])
            self.sen=np.append(self.sen,np.array([sentmp], dtype="float64"),axis=0)
            #4
            output = 0.0  # 應轉的角度
            if self.choose_train==4:
                for i in range(len(self.c)):
                    tmp = o_basis(sentmp, self.c[i], self.s)
                    output += tmp * self.w[i]
            elif self.choose_train==6:
                for i in range(len(self.c)):
                    tmp = o_basis(sentmp6, self.c[i], self.s)
                    output += tmp * self.w[i]
            if output >= 0:
                output %= 180
                if output > 40:
                    output = 40
            elif output < 0:
                output *= -1
                output %= 180
                output *= -1
                if output < -40:
                    output = -40
            self.deg=np.append(self.deg,output)
            self.cx, self.cy, self.cdegree = car(self.cx,self.cy, self.cdegree).carmove(output)
            self.xy=np.append(self.xy,[self.cx,self.cy])
            #print("car_run OK")
            self.play_animation()
            font = QtGui.QFont()
            font.setPointSize(12)
            self.ui.front.setText(str(front_dis))
            self.ui.front.setFont(font)
            self.ui.left.setText(str(left_dis))
            self.ui.left.setFont(font)
            self.ui.right.setText(str(right_dis))
            self.ui.right.setFont(font)
            #print("Text OK")
        else:
            self.mytimer.stop()
            self.writefile()
    def play_animation(self):
        self.ui.widget.canvas.ax.cla()
        self.ui.widget.canvas.ax.axis(xmin=-20, xmax=45)  # 設定x軸顯示範圍
        self.ui.widget.canvas.ax.axis(ymin=-10, ymax=55)  # 設定y軸顯示範圍
        self.ui.widget.canvas.ax.xaxis.set_major_locator(plt.MultipleLocator(10))
        self.ui.widget.canvas.ax.yaxis.set_major_locator(plt.MultipleLocator(10))
        index=0
        for i in range(len(self.xy)):
            if index>=len(self.xy):
                break
            self.ui.widget.canvas.ax.scatter(self.xy[index], self.xy[index+1], color='b')
            index=index+2
        #print("play_animation OK")
        self.plot_pic()

    def plot_pic(self):
        # 畫邊界、終點
        for i in range(len(self.x)):
            self.ui.widget.canvas.ax.plot(self.x[i], self.y[i], color='b')
            self.ui.widget.canvas.ax.scatter(self.x[i], self.y[i], color='r')
        rect = pc.Rectangle((18, 37), 12, 3, color='r')
        self.ui.widget.canvas.ax.add_patch(rect)
        #車
        nowcar = car(self.cx,self.cy,self.cdegree).back_ini()
        self.ui.widget.canvas.ax.add_patch(nowcar)
        self.ui.widget.canvas.draw()
        #print("plot_pic OK")
    def writefile(self):
        if self.choose_train==4:
            track4D = open(".txt", "w+")
            path = 'track4D.txt'
            with open(path, 'w') as f:
                for i in range(len(self.sen)):
                    lines = str(self.sen[i][0])+' '+str(self.sen[i][1])+' '+str(self.sen[i][2])+' '+str(self.deg[i])+'\n'
                    f.write(lines)
        elif self.choose_train==6:
            track6D = open(".txt", "w+")
            path = 'track6D.txt'
            index=0
            with open(path, 'w') as f:
                for i in range(len(self.sen)):
                    lines = str(self.xy[index])+' '+str(self.xy[index+1])+' '+str(self.sen[i][0])+' '+str(self.sen[i][1])+' '+str(self.sen[i][2])+' '+str(self.deg[i])+'\n'
                    index=index+2
                    f.write(lines)
app = QtWidgets.QApplication(sys.argv)
p = MatplotlibWidget()
p.show()
sys.exit(app.exec())
