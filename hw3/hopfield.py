import random
import numpy as np

class hopfield():
    def __init__(self, data ,test_data,width,height,data_num,epoch):
        # 創建W
        W = np.zeros([width*height,width*height])
        for L in range(data_num):
            x = data[L]
            for i in range(width*height):
                for j in range(i,width*height):
                    if i == j:
                        W[i, j] = 0
                    else:
                        W[i, j] += x[i] * x[j]
                        W[j, i] = W[i, j]
        W = W /width*height
        theta = np.zeros(width*height)
        for i in range(width*height):
            for j in range(width*height):
                theta[i] += W[i][j]
        self.ans_data=test_data
        for e in range(epoch):
            for i in range(data_num):
                x=self.ans_data[i]
                lenth=len(x)
                update=random.randint(0,lenth-1)
                next_value=0
                for j in range(lenth):
                    next_value+=W[update][j]*x[j]
                next_value-=theta[update] #內積
                if next_value>=0:
                    self.ans_data[i][update]=1
                if next_value<0:
                    self.ans_data[i][update]=-1
    def Return(self):
        return self.ans_data

