import sys
from PyQt5.QtWidgets import*
from PyQt5 import QtGui
from GUI import Ui_Form
import numpy as np
from hopfield import hopfield

class AppWindow(QDialog):
    def __init__(self):
        super().__init__()
        self.filename=""
        self.filename_test=""
        self.trained=0
        self.ui = Ui_Form() #新增剛剛拉的前端介面
        self.ui.setupUi(self)
        self.ui.data_not_exist.setVisible(False)
        self.ui.dataset.clicked.connect(self.load_data) #選dataset
        self.ui.show_butt.clicked.connect(self.show_pic)
        self.ui.train.clicked.connect(self.train_model)
        self.show() #show

    def load_data(self):
        fileName1, filetype = QFileDialog.getOpenFileName(self, "(*.txt)")  #取得路徑才能開test
        ### 要傳入 model，照圖的尺寸拆出二維陣列，再計算
        self.data_num = 1
        self.h = 0
        self.w = 0
        self.epoch=1000
        self.test_data = np.empty([0, 1], int)
        self.oritest = np.empty([0, 1], int)
        ###
        if fileName1 != '':
            self.trained=0
            self.ui.recall.setText('')
            self.ui.Test_data.setText('')
            ###切割路徑與檔名
            path = list(fileName1)
            self.tmpname = ""
            for i in range(len(fileName1)):
                if fileName1[len(fileName1)-1-i]=='/':
                    break
                path.pop()
                self.tmpname=fileName1[len(fileName1)-1-i]+self.tmpname
            pathstr=""
            for i in range(len(path)):
                pathstr+=path[i]
            self.ui.data_name.setText(fileName1)
            self.filename=fileName1
            if self.tmpname=="Basic_Training.txt":
                self.filename_test=pathstr+"Basic_Testing.txt"  #train、test 要放在同一個資料夾內(資料集的位置不必和code相同)
            elif self.tmpname=="Bonus_Training.txt":
                self.filename_test=pathstr+"Bonus_Testing.txt"
            with open(fileName1, 'r') as f:
                lines = f.readlines()
                #計算讀入的資料集之圖像尺寸
                for line in lines:
                    if line=='\n':
                        break
                    else:
                        if self.w==0:
                            for c in line:
                                self.w+=1
                        self.h+=1
                self.w-=1  #把句尾換行扣掉
                for line in lines:
                    if line=='\n':
                        self.data_num += 1
            font = QtGui.QFont()
            font.setPointSize(16)
            self.ui.how_many_pic_label.setText("The dataset has "+str(self.data_num)+" data.")
            self.ui.how_many_pic_label.setFont(font)
            self.open_test()
    def open_data(self,fileName1):
        print(self.noise_rate)
        self.noise = np.random.random(size=(self.data_num * self.h * self.w))
        index = -1
        with open(fileName1, 'r') as f:
            # 存資料
            lines = f.readlines()
            for line in lines:
                for c in line:
                    if c == ' ':
                        self.data = np.append(self.data, -1)
                        index += 1
                        if self.noise[index] < self.noise_rate:
                            self.data_noise=np.append(self.data_noise,1)
                        else:
                            self.data_noise = np.append(self.data_noise, -1)
                    elif c == '1':
                        self.data = np.append(self.data, 1)
                        index += 1
                        if self.noise[index] >= self.noise_rate:
                            self.data_noise = np.append(self.data_noise, 1)
                        else:
                            self.data_noise = np.append(self.data_noise, -1)
                    elif c == '\n':
                        break
        self.data = self.data.reshape(-1, self.w * self.h)
        self.data_noise=self.data_noise.reshape(-1,self.w*self.h)
        self.writefile()
    def writefile(self):
        noise_train = open("noise_train.txt", "w+")
        path = 'noise_train.txt'
        with open(path, 'w') as f:
            for i in range(self.data_num):
                    counter=0
                    lines = ''
                    for j in range(self.w*self.h):
                        counter+=1
                        if self.data_noise[i][j]==-1:
                            lines+=' '
                        elif self.data_noise[i][j]==1:
                            lines+='1'
                        if counter==self.w:
                            lines +='\n'
                            f.write(lines)
                            counter = 0
                            lines = ''
                    f.write('\n')
    def open_test(self):
        # 存測試資料
        with open(self.filename_test, 'r') as f:
            lines = f.readlines()
            # 存資料
            for line in lines:
                if line == '\n':
                    continue
                else:
                    for c in line:
                        if c == ' ':
                            self.test_data = np.append(self.test_data, -1)
                            self.oritest = np.append(self.oritest, -1)
                        elif c == '1':
                            self.test_data = np.append(self.test_data, 1)
                            self.oritest = np.append(self.oritest, 1)
                        elif c == '\n':
                            break
        self.test_data = self.test_data.reshape(-1, self.w*self.h)
        self.ans=self.test_data
        self.oritest=self.oritest.reshape(-1, self.w*self.h)
        #print(self.test_data)
    def show_pic(self):
        if self.ui.which_data.text()=='':
            self.ui.data_not_exist.setVisible(True)
            self.ui.Test_data.setText('')
            self.ui.recall.setText('')
            return
        number = int(self.ui.which_data.text())
        if number>=1 and number<=self.data_num:
            self.ui.data_not_exist.setVisible(False)
            char=np.empty([0,1],int)
            char=np.append(char,self.oritest[number-1])
            char=char.reshape(-1,self.w)
            #print(char)
            str = "\n\n\n\n\n\n"   #排版用
            for i in range(self.h):
                for b in range(9):#排版用
                    str+='　'#全形空白
                for j in range(self.w):
                    if char[i][j]==-1:
                        str+='　'#全形空白
                    else:
                        str+='１'#全形字
                str+='\n'
            self.ui.Test_data.setText(str)
            #展示回想結果
            if self.trained == 1:
                char_ans = np.empty([0, 1], int)
                char_ans = np.append(char_ans, self.ans[number-1])
                char_ans = char_ans.reshape(-1, self.w)
                # print(char)
                str_ans = "\n\n\n\n\n\n"  # 排版用
                for i in range(self.h):
                    for b in range(9):  # 排版用
                        str_ans += '　'  # 全形空白
                    for j in range(self.w):
                        if char_ans[i][j] == -1:
                            str_ans += '　'  # 全形空白
                        else:
                            str_ans += '１'  # 全形字
                    str_ans += '\n'
                self.ui.recall.setText(str_ans)

        else:
            self.ui.data_not_exist.setVisible(True)
            self.ui.Test_data.setText('')
            self.ui.recall.setText('')
    def train_model(self):
        if self.filename!='':
            self.noise_rate = self.ui.noise.value()
            self.data = np.empty([0, 1], int)
            self.data_noise = np.empty([0, 1], int)
            self.open_data(self.filename)
            if self.ui.epoch.text() != '':
                self.epoch=int(self.ui.epoch.text())
            self.ans=hopfield(self.data_noise,self.test_data,self.w,self.h,self.data_num,self.epoch).Return()
            print((self.ans == self.data).all())
            print((self.ans == self.data_noise).all())
            self.trained=1
app = QApplication(sys.argv)
w = AppWindow()
w.show()
sys.exit(app.exec_())