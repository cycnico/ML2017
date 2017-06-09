import os
import sys
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from keras.models import Sequential, Model, load_model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.merge import Add, Dot, Concatenate
from keras.layers import Input, Embedding, Reshape
from keras.optimizers import SGD, Adam, Adadelta
from keras.utils import np_utils

max_userid = 6040
max_movieid = 3952

def MFmodel(n_users, n_items, latent_dim):

    user_input = Input(shape=[1])
    item_input = Input(shape=[1])
    user_vector = Embedding(n_users, latent_dim, embeddings_initializer='random_normal')(user_input)
    user_vector = Flatten()(user_vector)
    item_vector = Embedding(n_items, latent_dim, embeddings_initializer='random_normal')(item_input)
    item_vector = Flatten()(item_vector)
    user_bias = Embedding(n_users, 1, embeddings_initializer='zeros')(user_input)
    user_bias = Flatten()(user_bias)
    item_bias = Embedding(n_items, 1, embeddings_initializer='zeros')(item_input)
    item_bias = Flatten()(item_bias)
    r_hat = Dot(axes=1)([user_vector,item_vector])
    r_hat = Add()([r_hat,user_bias,item_bias])
    model = Model([user_input,item_input],r_hat)
    model.compile(loss='mse', optimizer='adamax')
    return model


trained_model = MFmodel(max_userid, max_movieid, 15)
trained_model.load_weights('MFmodel.h5')
Test = np.delete(np.genfromtxt(sys.argv[1]+'test.csv', delimiter=',',dtype= np.str),0,0)

Ans = trained_model.predict([Test[:,1],Test[:,2]])

#output
idlist = []
for i in range(100336):
    idlist.append(repr(i+1))
idli = pd.DataFrame(idlist,columns=['TestDataID'])
y = pd.DataFrame(Ans,columns=['Rating'])
result = pd.concat([idli,y], axis =1)
result.to_csv(sys.argv[2], index = False)
