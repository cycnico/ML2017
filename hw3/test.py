import os
import sys
import numpy as np
from keras.utils import np_utils
from keras.models import load_model
import pandas as pd
from pandas import Series, DataFrame
model = load_model(sys.argv[1])

#predict
Test = np.delete(np.genfromtxt(sys.argv[2], delimiter=',',dtype= np.str),0,0)
X_test = np.array(Test[:,1], dtype= np.str)

X_test_temp = np.zeros((7178,2304))
for i in range (7178):
    X_test_temp[i] = np.fromstring(X_test[i], dtype='float32', sep=' ')

X_test = X_test_temp
X_test = X_test.reshape(X_test.shape[0],48,48,1)
X_test_temp = []

Y_test = model.predict_classes(X_test)

#output
idlist = []
for i in range(7178):
    idlist.append(repr(i))
idli = pd.DataFrame(idlist,columns=['id'])
y = pd.DataFrame(Y_test,columns=['label'])
result = pd.concat([idli,y], axis =1)
result.to_csv(sys.argv[3],index = False)
