import matplotlib.pyplot as plt
import numpy as np

class draw():
    def __init__(self, data ,epoch,trate):
        filename = data
        self.training_ac=float(0.0)
        self.test_ac = float(0.0)
        Epoch=epoch
        Trate=trate
        fig = plt.figure(figsize=(18, 6))
        ax1 = fig.add_subplot(131)
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133)
        Point = np.empty([0, 2], float)
        Point_c = np.array([])
        Point_c_pre = np.array([])
        has_zero=0
        with open(filename, 'r') as f:
            lines = f.readlines()
            for line in lines:
                value = [float(s) for s in line.split()]
                Point = np.append(Point, np.array([[value[0],value[1]]],dtype="float64"), axis=0)
                Point_c = np.append(Point_c, value[2])
                if value[2]==0.0:
                    has_zero=1
        if has_zero==1:
            Point_c+=1.0
        #print(X1) #print(Y1) #print(X2) #print(Y2)
        indices = np.random.permutation(len(Point))
        rand_data = Point[indices]
        rand_data_z = Point_c[indices]
        L = (len(rand_data_z)//3)*2

        #training
        self.w = np.random.rand(len(rand_data[0]))
        bias=0.0
        phi=0
        for i in range(Epoch):
            for j in range(L):
                v=rand_data[j]*self.w+bias
                v=float(sum(v))  #加法器
                if(v<0):         #活化函數
                    phi=1
                else:
                    phi=2
                if phi!=rand_data_z[j]:  #調鍵結值
                    if (v < 0):
                        self.w=self.w+Trate*rand_data[j]
                        bias+=Trate
                    else:
                        self.w=self.w-Trate*rand_data[j]
                        bias-=Trate
        #training end

        #準確率計算
        for i in range(L):
            v = rand_data[i] * self.w + bias
            v = float(sum(v))
            if (v < 0):
                phi = 1
                Point_c_pre=np.append(Point_c_pre,1)
            else:
                phi = 2
                Point_c_pre = np.append(Point_c_pre,2)
            if phi==rand_data_z[i]:
                self.training_ac+=float(1.0)
        self.training_ac/=L
        for i in range(L,len(rand_data_z)):
            v = rand_data[i] * self.w + bias
            v = float(sum(v))
            if (v < 0):
                phi = 1
                Point_c_pre = np.append(Point_c_pre, 1)
            else:
                phi = 2
                Point_c_pre = np.append(Point_c_pre, 2)
            if phi == rand_data_z[i]:
                self.test_ac += float(1.0)
        tmp=(len(rand_data_z)-L)
        self.test_ac /= float(tmp)
        #結束


        #draw
        for i in range(len(rand_data_z)):
            if rand_data_z[i]==1:
                ax1.plot(rand_data[i][0],rand_data[i][1], "o",color='b')
            else:
                ax1.plot(rand_data[i][0],rand_data[i][1], "o",color='r')
        for i in range(0,L):
            if Point_c_pre[i]==1:
                ax2.plot(rand_data[i][0],rand_data[i][1], "o",color='b')
            else:
                ax2.plot(rand_data[i][0],rand_data[i][1], "o",color='r')
        for i in range(L,len(rand_data_z)):
            if Point_c_pre[i]==1:
                ax3.plot(rand_data[i][0],rand_data[i][1], "o",color='b')
            else:
                ax3.plot(rand_data[i][0],rand_data[i][1], "o",color='r')
        ax2.axline([0,-2*bias/self.w[1]],slope=-1*(self.w[0]/self.w[1]))
        ax3.axline([0,-2*bias/self.w[1]],slope=-1*(self.w[0]/self.w[1]))
        plt.show()
    def back(self):
        return self.training_ac,self.test_ac,self.w



#pyinstaller -F uiback.py -p UI.py -p Perceptron.py --noconsole