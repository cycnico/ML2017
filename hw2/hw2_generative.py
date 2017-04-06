import sys
import numpy as np
import math
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

# feature normalization
X_all = np.concatenate((X_train, X_test))
mean = np.mean(X_all, axis=0)
sigma = np.std(X_all, axis=0)

index = [0, 1, 3, 4, 5, 106, 107, 108, 109, 110]
mean_vec = np.zeros(X_all.shape[1])
std_vec = np.ones(X_all.shape[1])
mean_vec[index] = mean[index]
std_vec[index] = sigma[index]

X_all_normed = (X_all - mean_vec) / std_vec
X_train_normed = X_all_normed[0:X_train.shape[0]-1]
X_test_normed = X_all_normed[X_train.shape[0]:]

# gaussian distribution parameters
train_data_size = X_train_normed.shape[0]
cnt1 = 0
cnt2 = 0

mu1 = np.zeros((111,))
mu2 = np.zeros((111,))
for i in range(train_data_size):
    if Y_train[i] == 1:
        mu1 += X_train_normed[i]
        cnt1 += 1
    else:
        mu2 += X_train_normed[i]
        cnt2 += 1
mu1 /= cnt1
mu2 /= cnt2

sigma1 = np.zeros((111,111))
sigma2 = np.zeros((111,111))
for i in range(train_data_size):
    if Y_train[i] == 1:
        sigma1 += np.dot(np.transpose([X_train_normed[i] - mu1]), [(X_train_normed[i] - mu1)])
    else:
        sigma2 += np.dot(np.transpose([X_train_normed[i] - mu2]), [(X_train_normed[i] - mu2)])
sigma1 /= cnt1
sigma2 /= cnt2
shared_sigma = (float(cnt1) / train_data_size) * sigma1 + (float(cnt2) / train_data_size) * sigma2

sigma_inverse = np.linalg.inv(shared_sigma)
w = np.dot( (mu1-mu2), sigma_inverse)
b = (-0.5) * np.dot(np.dot([mu1], sigma_inverse), mu1) + (0.5) * np.dot(np.dot([mu2], sigma_inverse), mu2) + np.log(float(cnt1)/cnt2)

#prediction
z = np.dot(X_test_normed,np.transpose(w)) + b
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
