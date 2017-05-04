import os
import sys
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD, Adam, Adadelta
from keras.utils import np_utils
from keras.callbacks import Callback, ModelCheckpoint
import matplotlib.pyplot as plt

class History(Callback):
    def on_train_begin(self,logs={}):
        self.tr_losses=[]
        self.val_losses=[]
        self.tr_accs=[]
        self.val_accs=[]

    def on_epoch_end(self,epoch,logs={}):
        self.tr_losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.tr_accs.append(logs.get('acc'))
        self.val_accs.append(logs.get('val_acc'))

def dump_history(store_path,logs):
    with open(os.path.join(store_path,'train_loss'),'a') as f:
        for loss in logs.tr_losses:
            f.write('{}\n'.format(loss))
    with open(os.path.join(store_path,'train_accuracy'),'a') as f:
        for acc in logs.tr_accs:
            f.write('{}\n'.format(acc))
    with open(os.path.join(store_path,'valid_loss'),'a') as f:
        for loss in logs.val_losses:
            f.write('{}\n'.format(loss))
    with open(os.path.join(store_path,'valid_accuracy'),'a') as f:
        for acc in logs.val_accs:
            f.write('{}\n'.format(acc))

#read data
Train = np.delete(np.genfromtxt(sys.argv[1], delimiter=',',dtype= np.str),0,0)
#Valid = np.genfromtxt('valid2.csv', delimiter=',',dtype= np.str)
X_train = np.array(Train[:,1], dtype= np.str)
Y_train = np.array(Train[:,0], dtype= 'int16')
#X_valid = np.array(Valid[:,1], dtype= np.str)
#Y_valid = np.array(Valid[:,0], dtype= 'int16')

X_train_temp = np.zeros((28709,2304))
for i in range (28709):
    X_train_temp[i] = np.fromstring(X_train[i], dtype='float32', sep=' ')
X_train = X_train_temp
X_train = X_train.reshape(X_train.shape[0],48,48,1)
X_train_temp = []
Y_train = np.reshape(Y_train,(1,28709))
Y_train = np_utils.to_categorical(Y_train, 7)

"""
X_valid_temp = np.zeros((3710,2304))
for i in range (3710):
    X_valid_temp[i] = np.fromstring(X_valid[i], dtype='float32', sep=' ')
X_valid = X_valid_temp
X_valid = X_valid.reshape(X_valid.shape[0],48,48,1)
X_valid_temp = []
Y_valid = np.reshape(Y_valid,(1,3710))
Y_valid = np_utils.to_categorical(Y_valid, 7)
"""

#build model
model = Sequential()
model.add(Convolution2D(64, 3, padding='same', input_shape=(48,48,1)))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, padding='same'))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.5))

model.add(Convolution2D(64, 3,  padding='same'))
model.add(Activation('relu'))
model.add(Convolution2D(128, 3,  padding='same'))
model.add(Activation('relu'))
model.add(Convolution2D(128, 3, padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.5))

model.add(Convolution2D(128, 3,  padding='same'))
model.add(Activation('relu'))
model.add(Convolution2D(256, 3,  padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(7))
model.add(Activation('softmax'))

opt = Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
model.summary()

#callbacks
history = History()
"""
callback = [
    history,
    ModelCheckpoint('model.h5', monitor='val_acc', save_best_only=True, mode='auto', verbose=0),
]
"""

#train
model.fit(X_train, Y_train, batch_size = 128, epochs = 100, callbacks=history)

model.save('model.h5')

"""
#dump_history('./history',history)
plt.plot(history.tr_accs)
plt.plot(history.val_accs)
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.show()
"""
