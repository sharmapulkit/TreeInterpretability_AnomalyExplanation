import numpy as np
import csv 
import pandas as pd
import os

from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

PATH='~/Documents/PDF/umass/sem2/cs696ds/code/'

dataset = os.path.join(PATH, 'datasets/networking-results.csv')

data = pd.read_csv(dataset)

target_columns = ['html_attrs', 'html_tags', 'elapsed', 'decompressed_content_length', \
        'raw_content_length']
feature_columns = ['record.count', 'mobile_user_agent', 'proxy', 'compression', \
        'trial', 'ping_time']

train_fraction = 0.7
train_size = int(data.shape[0]*train_fraction)
X_train = data[feature_columns][:train_size]
Y_train = data[target_columns][:train_size]


X = X_train
distortions = []
#K = range(45,50)
#for k in K:
#        kmeanModel = KMeans(n_clusters=k).fit(X)
#        kmeanModel.fit(X)
#        distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])
#
## Plot the elbow
#plt.plot(K, distortions, 'bx-')
#plt.xlabel('k')
#plt.ylabel('Distortion')
#plt.title('The Elbow Method showing the optimal k')
#plt.show()

data.columns
