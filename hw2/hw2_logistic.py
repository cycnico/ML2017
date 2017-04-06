import sys
import numpy as np
from math import log, floor

def sigmoid(z):
	res = 1 / (1.0 + np.exp(-z))
	return np.clip(res, 0.00000000000001, 0.99999999999999)

X_train = np.delete(np.genfromtxt(sys.argv[1], delimiter=','), 0, 0)
Y_train = np.genfromtxt(sys.argv[2], delimiter=',')
X_test = np.delete(np.genfromtxt(sys.argv[3], delimiter=','), 0, 0)

#add square terms
X_train_temp = X_train[:,0:6]
X_train_temp = np.delete(X_train_temp,(2),axis=1)
X_train = np.concatenate((X_train,X_train_temp*X_train_temp),axis=1)
X_test_temp = X_test[:,0:6]
X_test_temp = np.delete(X_test_temp,(2),axis=1)
X_test = np.concatenate((X_test,X_test_temp*X_test_temp),axis=1)

#feature normalization
X_all = np.concatenate((X_train, X_test))
mu = np.mean(X_all, axis=0)
sigma = np.std(X_all, axis=0)

index = [0, 1, 3, 4, 5, 106, 107, 108, 109, 110]
mean_vec = np.zeros(X_all.shape[1])
std_vec = np.ones(X_all.shape[1])
mean_vec[index] = mu[index]
std_vec[index] = sigma[index]

X_all_normed = (X_all - mean_vec) / std_vec
X_train_normed = X_all_normed[0:X_train.shape[0]-1]
X_test_normed = X_all_normed[X_train.shape[0]:]

#initiallize
w = np.zeros((111,))
b = np.zeros((1,))
pre_w_grad = np.zeros((111,))
pre_w_grad += 0.0001
pre_b_grad= np.zeros((1,))
pre_b_grad += 0.0001
train_data_size = X_train_normed.shape[0]
lr = 0.000036
iteration = 1400
lamb = 0.05
batch_size = 512
display_num = 20
batch_num = int(floor(train_data_size / batch_size))

for i in range(iteration):

    L = 0.0
    for idx in range(batch_num):
        X_batch = X_train_normed[idx*batch_size:(idx+1)*batch_size]
        Y_batch = Y_train[idx*batch_size:(idx+1)*batch_size]

        z = np.dot(X_batch, np.transpose(w)) + b
        y = sigmoid(z)

        w_grad = np.sum(-1 * X_batch * (Y_batch - y).reshape((batch_size,1)), axis=0)
        b_grad = np.sum(-1 * (Y_batch - y))

        """"
        pre_w_grad += w_grad**2
        pre_b_grad += b_grad**2
        w_ada = np.sqrt(pre_w_grad)
        b_ada = np.sqrt(pre_b_grad)
        pos = np.sign(w_grad/w_ada)

        w = w - lr*((w_grad+lamb*w)/w_ada)
        b = b - lr*((b_grad+lamb*b)/b_ada)
        """""

        w = w - lr * (w_grad + lamb*w)
        b = b - lr * b_grad
		

#prediction
z = (np.dot(X_test_normed,np.transpose(w)) + b)
y = sigmoid(z)
for i in range (16281):
    if(y[i]>0.5):
        y[i]=1
    if(y[i]<0.5):
        y[i]=0
    if(X_test[i][0]<20):
        y[i]=0

#output
with open(sys.argv[4], 'w') as f:
    f.write('id,label\n')
    for i, v in  enumerate(y):
        f.write('%d,%d\n' %(i+1, v))
