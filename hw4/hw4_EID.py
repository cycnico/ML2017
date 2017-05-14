import os
import sys
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from sklearn import manifold
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

data = np.load(sys.argv[1])

Y = np.zeros(200)
threshold = 320
for i in range(200):
    x = data[str(i)]
    y = 0
    pca = PCA()
    pca.fit(x)
    PCA(copy=True)
    for j in range(100):
        if(pca.explained_variance_[j]>threshold):
            y = y + 1
    Y[i] = y

# plot PCA features
"""
x = data[str(40)]
pca = PCA()
pca.fit(x)
PCA(copy=True)
features = range(pca.n_components_)
plt.bar(features, pca.explained_variance_)
plt.xticks(features)
plt.ylabel('variance')
plt.xlabel('PCA feature')
plt.show()
"""

idlist = []
Y_test = np.log(Y)
for i in range(200):
    idlist.append(repr(i))
idli = pd.DataFrame(idlist,columns=['SetId'])
y = pd.DataFrame(Y_test,columns=['LogDim'])
result = pd.concat([idli,y], axis =1)
result.to_csv(sys.argv[2],index = False)
