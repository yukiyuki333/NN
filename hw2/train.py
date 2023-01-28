import numpy as np

class rbfn4d():
    def __init__(self):
        self.filename="train4dAll.txt"
        self.distance = np.empty([0, 3], float)
        self.angel=np.array([])
        with open(self.filename, 'r') as f:
            lines = f.readlines()
            for line in lines:
                value = [float(s) for s in line.split()]
                self.distance = np.append(self.distance, np.array([[value[0], value[1], value[2]]], dtype="float64"),axis=0)
                self.angel = np.append(self.angel, value[3])
        self.K_num=7
    def o_basis(self,data,m,sigma): #輸入向量轉換為徑向基底函數，m 是該群群聚中心
        difftmp = data-m
        squtmp = np.square(difftmp.sum(axis=0))
        phi=np.exp(-squtmp/(2*(sigma)**2))
        return phi
    def cal_dis(self,x,y):
        diff=x-y
        dis = np.sqrt(np.square(diff).sum(axis=1))
        return dis
    def train(self):
        #找初始中心點
        first=np.random.choice(len(self.distance))
        select=[first]
        for i in range(1,self.K_num):
            all_dis=np.empty([len(self.distance),0],float)
            for j in select:
                tmpdis=self.cal_dis(self.distance,self.distance[j]).reshape(-1,1)
                all_dis=np.c_[all_dis,tmpdis]
            min_d=all_dis.min(axis=1).reshape(-1,1)
            index=np.argmax(min_d)
            select.append(index)
        self.center=self.distance[select]
        #find m
        totallen,singlelen = self.distance.shape
        while True:
            dict_y={}
            for j in range(self.K_num):
                dict_y[j]=np.empty([0,singlelen])
            for i in range(totallen):
                distmp=self.cal_dis(self.distance[i],self.center)
                label=np.argsort(distmp)[0]
                dict_y[label]=np.r_[dict_y[label],self.distance[i].reshape(1,-1)]
            censtmp=np.empty([0,singlelen])
            for i in range(self.K_num):
                centmp=np.mean(dict_y[i],axis=0).reshape(1,-1)
                censtmp=np.r_[censtmp,centmp]
            if np.all(censtmp==self.center)==True:
                break
            else:
                self.center=censtmp
        #find sigma
        cmax=0.0
        for i in range(len(self.center)):
            x=self.center[i]
            for j in range(len(self.center)):
                y=self.center[j]
                if(i!=j):
                    diff=x-y
                    tmp=np.sqrt(np.square(diff).sum(axis=0))
                    if cmax<tmp:
                        cmax=tmp
        self.sigmall=cmax/np.sqrt(2*(self.K_num))
        #input 轉換
        output=np.array([])
        epoch=600
        trate=0.03
        self.w=np.random.randn(len(self.center))
        input_basis=np.empty([0,self.K_num],float)
        for i in range(len(self.distance)):
            anstmp=0.0
            input_tmp=np.array([])
            for j in range(len(self.center)):
                tmp=self.o_basis(self.distance[i],self.center[j],self.sigmall)
                tmp = round(tmp, 6)
                input_tmp=np.append(input_tmp,tmp)
                anstmp+=tmp*self.w[j]
            if anstmp >= 0:
                anstmp%=180
                if anstmp>40:
                    anstmp=40
            elif anstmp < 0:
                anstmp *= -1
                anstmp %= 180
                anstmp *= -1
                if anstmp<-40:
                    anstmp=-40

            output=np.append(output,anstmp)
            input_basis=np.append(input_basis,np.array([input_tmp],dtype=float),axis=0)
        #調鍵結值
        theta=0
        for num in range(epoch):
            for i in range(len(self.distance)):
                if(self.angel[i]!=output[i]):
                    delta=trate*(self.angel[i]-output[i]-theta)
                    self.w=self.w+delta*input_basis[i]
                    theta+=delta
            for k in range(len(self.distance)):
                anstmp = 0.0
                for l in range(len(self.center)):
                    tmp = self.o_basis(self.distance[k], self.center[l], self.sigmall)
                    anstmp += tmp * self.w[l]
                if anstmp >= 0:
                    anstmp %= 180
                    if anstmp > 40:
                        anstmp = 40
                elif anstmp < 0:
                    anstmp *= -1
                    anstmp %= 180
                    anstmp *= -1
                    if anstmp < -40:
                        anstmp = -40
                output[k] = anstmp
        #for i in range(len(self.distance)):
            #print(output[i])
        return self.w,self.center,self.sigmall


class rbfn6d():
    def __init__(self):
        self.filename="train6dAll.txt"
        self.distance = np.empty([0, 5], float)
        self.angel=np.array([])
        with open(self.filename, 'r') as f:
            lines = f.readlines()
            for line in lines:
                value = [float(s) for s in line.split()]
                self.distance = np.append(self.distance, np.array([[value[0], value[1], value[2],value[3],value[4]]], dtype="float64"),axis=0)
                self.angel = np.append(self.angel, value[5])
        self.K_num=10
    def o_basis(self,data,m,sigma): #輸入向量轉換為徑向基底函數，m 是該群群聚中心
        difftmp = data-m
        squtmp = np.square(difftmp.sum(axis=0))
        phi=np.exp(-squtmp/(2*(sigma)**2))
        return phi
    def cal_dis(self,x,y):
        diff=x-y
        dis = np.sqrt(np.square(diff).sum(axis=1))
        return dis

    def train(self):
        #找初始中心點
        first=np.random.choice(len(self.distance))
        select=[first]
        for i in range(1,self.K_num):
            all_dis=np.empty([len(self.distance),0],float)
            for j in select:
                tmpdis=self.cal_dis(self.distance,self.distance[j]).reshape(-1,1)
                all_dis=np.c_[all_dis,tmpdis]
            min_d=all_dis.min(axis=1).reshape(-1,1)
            index=np.argmax(min_d)
            select.append(index)
        self.center=self.distance[select]
        #find m
        totallen,singlelen = self.distance.shape
        while True:
            dict_y={}
            for j in range(self.K_num):
                dict_y[j]=np.empty([0,singlelen])
            for i in range(totallen):
                distmp=self.cal_dis(self.distance[i],self.center)
                label=np.argsort(distmp)[0]
                dict_y[label]=np.r_[dict_y[label],self.distance[i].reshape(1,-1)]
            censtmp=np.empty([0,singlelen])
            for i in range(self.K_num):
                centmp=np.mean(dict_y[i],axis=0).reshape(1,-1)
                censtmp=np.r_[censtmp,centmp]
            if np.all(censtmp==self.center)==True:
                break
            else:
                self.center=censtmp
        #find sigma
        cmax=0.0
        for i in range(len(self.center)):
            x=self.center[i]
            for j in range(len(self.center)):
                y=self.center[j]
                if(i!=j):
                    diff=x-y
                    tmp=np.sqrt(np.square(diff).sum(axis=0))
                    if cmax<tmp:
                        cmax=tmp
        self.sigmall=cmax/np.sqrt(2*(self.K_num))
        #input 轉換
        output=np.array([])
        epoch=600
        trate=0.03
        self.w=np.random.randn(len(self.center))
        input_basis=np.empty([0,self.K_num],float)
        for i in range(len(self.distance)):
            anstmp=0.0
            input_tmp=np.array([])
            for j in range(len(self.center)):
                tmp=self.o_basis(self.distance[i],self.center[j],self.sigmall)
                tmp=round(tmp,6)
                input_tmp=np.append(input_tmp,tmp)
                anstmp+=tmp*self.w[j]
            if anstmp >= 0:
                anstmp %= 180
                if anstmp > 40:
                    anstmp = 40
            elif anstmp < 0:
                anstmp *= -1
                anstmp %= 180
                anstmp *= -1
                if anstmp < -40:
                    anstmp = -40
            output=np.append(output,anstmp)
            input_basis=np.append(input_basis,np.array([input_tmp],dtype=float),axis=0)
        #調鍵結值
        theta=0
        for num in range(epoch):
            for i in range(len(self.distance)):
                if(self.angel[i]!=output[i]):
                    delta=trate*(self.angel[i]-output[i]-theta)
                    self.w=self.w+delta*input_basis[i]
                    theta+=delta
            for k in range(len(self.distance)):
                anstmp = 0.0
                for l in range(len(self.center)):
                    tmp = self.o_basis(self.distance[k], self.center[l], self.sigmall)
                    anstmp += tmp * self.w[l]
                if anstmp >= 0:
                    anstmp %= 180
                    if anstmp > 40:
                        anstmp = 40
                elif anstmp < 0:
                    anstmp *= -1
                    anstmp %= 180
                    anstmp *= -1
                    if anstmp < -40:
                        anstmp = -40
                output[k] = anstmp
        #for i in range(len(self.distance)):
            #print(output[i])
        return self.w,self.center,self.sigmall



'''
t=rbfn6d()
t1,t2,t3=t.train()
print(t1)
print(t2)
print(t3)
'''