import os
from keras.models import load_model
import keras.backend as K
from keras.utils import np_utils
import numpy as np
import matplotlib.pyplot as plt

emotion_classifier = load_model('model.h5')
input_img = emotion_classifier.input
img_ids = [777]

Train = np.delete(np.genfromtxt('train.csv', delimiter=',',dtype= np.str),0,0)
X_train = np.array(Train[:,1], dtype= np.str)
Y_train = np.array(Train[:,0], dtype= 'int16')

X_train_temp = np.zeros((28709,2304))
for i in range (28709):
    X_train_temp[i] = np.fromstring(X_train[i], dtype='float32', sep=' ')

X_train = X_train_temp
X_train = X_train.reshape(X_train.shape[0],1,48,48,1)
X_train_temp = []
Y_train = np.reshape(Y_train,(1,28709))
Y_train = np_utils.to_categorical(Y_train, 7)

X_train = X_train.reshape(X_train.shape[0],1,48,48,1)

val_proba = emotion_classifier.predict(X_train[777])
pred = val_proba.argmax(axis=-1)
target = K.mean(emotion_classifier.output[:, pred])
grads = K.gradients(target, input_img)[0]
fn = K.function([input_img, K.learning_phase()], [grads])

grads_value = fn([X_train[777], 0])
temp = np.array(grads_value[0][0])
temp -= temp.min()
temp /= temp.max()
grads_normed = temp
heatmap = grads_normed
heatmap.reshape(48, 48)

thres = 0.5
ori = X_train[777].reshape(48, 48)
see = X_train[777].reshape(48, 48)
s = np.where(heatmap<=thres)
t = np.zeros((48,48))
for i in range(48):
    for j in range (48):
        t[i][j]=heatmap[i][j]*255
        if(heatmap[i][j]<=thres):
            see[i][j] = 0

plt.figure()
plt.imshow(ori,cmap='gray')
plt.colorbar()
plt.tight_layout()
fig = plt.gcf()
plt.draw()
fig.savefig('origin.png', dpi=100)

plt.figure()
plt.imshow(t, cmap=plt.cm.jet)
plt.colorbar()
plt.tight_layout()
fig = plt.gcf()
plt.draw()
fig.savefig('heatmap.png', dpi=100)

plt.figure()
plt.imshow(see,cmap='gray')
plt.colorbar()
plt.tight_layout()
fig = plt.gcf()
plt.draw()
fig.savefig('mask.png', dpi=100)

plt.figure()
plt.imshow(t,cmap='gray')
plt.colorbar()
plt.tight_layout()
fig = plt.gcf()
plt.draw()
fig.savefig('filter.png', dpi=100)
