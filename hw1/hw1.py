import sys
import numpy as np
import pandas as pd
from pandas import Series, DataFrame

data = pd.read_csv(sys.argv[1],encoding = "big5")
temp = data[list(range(3,27))]
Data = []
for i in range(18):
    Data.append([])

for i in range(240):
    for j in range(24):
        for k in range(18):
            if temp.loc[i*18+k][j] == 'NR':
                Data[k].append(float(0))
            else:
                Data[k].append(float(temp.loc[i*18+k][j]))


train_x = []
train_y = []
 
for i in range(200):
    train_x.append([1])
    for s in range(7):
        train_x[i].append( Data[9][i*28+s] )
    train_y.append(Data[9][i*28+7])    

# gradient descent
# y = b + w * x
w = np.ones(8)
w /= 20
lr = 0.01
iteration = 70000
pre_gra = np.zeros(8)
train_xT = np.transpose(train_x)

for i in range(iteration):
    y = np.dot(train_x,w)
    L = y - train_y
    gra = 2*np.dot(train_xT,L)
    pre_gra += gra**2
    ada = np.sqrt(pre_gra)
    w -= lr*gra/ada
    

t = pd.read_csv(sys.argv[2],header = None, encoding = "big5")
test_x = []
test_y = []
for i in range(240):
    test_x.append([1])
    for s in range(4,11):
        test_x[i].append(float(t.loc[i*18+9][s]))
    temp = np.dot(w,test_x[i])
    if temp > 0:
        test_y.append(temp)
    else:
        test_y.append(0)


idlist = []
for i in range(240):
    idlist.append("id_"+repr(i))
idli = pd.DataFrame(idlist,columns=['id'])
te = pd.DataFrame(test_y,columns=['value'])
result = pd.concat([idli,te], axis =1)


result.to_csv(sys.argv[3],index = False)
