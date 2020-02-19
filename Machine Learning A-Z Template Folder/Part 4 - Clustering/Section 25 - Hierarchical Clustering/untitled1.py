#import all libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import data
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:,[3,4]].values

#choosing the right number of clusters using dendrogram
import scipy.cluster.hierarchy as sch
dendogram = sch.dendrogram(sch.linkage(X,method = 'ward'))
plt.title('dendrogram')
plt.xlabel('data points')
plt.ylabel('Euclidean distance')
plt.show()

#so optimal cluster is 5
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean',linkage = 'ward')
y = hc.fit_predict(X)

#visualising the clusters
plt.scatter(X[y == 0,0],X[y==0,1],c = 'blue',label = 'cluster1')
plt.scatter(X[y == 1,0],X[y==1,1],c = 'red',label = 'cluster2')
plt.scatter(X[y == 2,0],X[y==2,1],c = 'green',label = 'cluster3')
plt.scatter(X[y == 3,0],X[y==3,1],c = 'violet',label = 'cluster4')
plt.scatter(X[y == 4,0],X[y==4,1],c = 'cyan',label = 'cluster5')
plt.title('Hierarchy clustering')
plt.xlabel('income')
plt.ylabel('spending score')
plt.legend()
plt.show()