import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

X = np.zeros((100,4096))
for i in range (100):
    sr = "./face/"+str(i+1)+".bmp"
    im = Image.open(sr)
    im = im.convert("L")
    data = im.getdata()
    data = np.array(data)
    X[i] = data

X_mean = X.mean(axis=0, keepdims=True)
X_ctr = X - X_mean
u, s, v = np.linalg.svd(X_ctr)

k = 59
x_mean = np.reshape(X_mean,(4096,))
x_reduced = np.zeros((100,k))
x_recovered = np.zeros((100,4096))
sum_ = 0
rmse = 0
for i in range(100):
    for j in range(k):
        temp = np.transpose(v[j])
        x_reduced[i][j] = np.dot(X_ctr[i],temp)
        sum_ = sum_ + x_reduced[i][j] * v[j]
    x_recovered[i] = sum_ + x_mean
    sum_ = 0
for i in range(100):
    temp = X[i] - x_recovered[i]
    s = np.square(temp)
    for j in range(4096):
        rmse = rmse + s[j]
rmse = rmse / 4096 / 100
rmse = (rmse**0.5) / 256

x_recovered = np.array(x_recovered)

#plot average
avg = np.reshape(X_mean,(64,64))
plt.imshow(avg, cmap='gray')
plt.show()

#plot eigenfaces
plt.figure(figsize=(8,8))

plt.subplot(3,3,1)
plt.title('eigenface 1')
plt.axis('off')
eig = np.reshape(v[0],(64,64))
plt.imshow(eig, cmap='gray')

plt.subplot(3,3,2)
plt.title('eigenface 2')
plt.axis('off')
eig = np.reshape(v[1],(64,64))
plt.imshow(eig, cmap='gray')

plt.subplot(3,3,3)
plt.title('eigenface 3')
plt.axis('off')
eig = np.reshape(v[2],(64,64))
plt.imshow(eig, cmap='gray')

plt.subplot(3,3,4)
plt.title('eigenface 4')
plt.axis('off')
eig = np.reshape(v[3],(64,64))
plt.imshow(eig, cmap='gray')

plt.subplot(3,3,5)
plt.title('eigenface 5')
plt.axis('off')
eig = np.reshape(v[4],(64,64))
plt.imshow(eig, cmap='gray')

plt.subplot(3,3,6)
plt.title('eigenface 6')
plt.axis('off')
eig = np.reshape(v[5],(64,64))
plt.imshow(eig, cmap='gray')

plt.subplot(3,3,7)
plt.title('eigenface 7')
plt.axis('off')
eig = np.reshape(v[6],(64,64))
plt.imshow(eig, cmap='gray')

plt.subplot(3,3,8)
plt.title('eigenface 8')
plt.axis('off')
eig = np.reshape(v[7],(64,64))
plt.imshow(eig, cmap='gray')

plt.subplot(3,3,9)
plt.title('eigenface 9')
plt.axis('off')
eig = np.reshape(v[8],(64,64))
plt.imshow(eig, cmap='gray')

plt.show()


#plot origin image
plt.figure(figsize=(8,8))
for i in range (100):
    sr = "./face/"+str(i+1)+".bmp"
    im = Image.open(sr)
    im = im.convert("L")
    data = im.getdata()
    data = np.array(data)

    plt.subplot(10,10,i+1)
    plt.axis('off')
    data = np.reshape(data,(64,64))
    plt.imshow(data, cmap='gray')

plt.show()


#plot reconstruct
plt.figure(figsize=(8,8))
for i in range (100):
    data = x_recovered[i]
    plt.subplot(10,10,i+1)
    plt.axis('off')
    data = np.reshape(data,(64,64))
    plt.imshow(data, cmap='gray')

plt.show()
