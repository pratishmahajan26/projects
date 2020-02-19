# import the libs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:,[3,4]].values

#choosing the number of clusters
from sklearn.cluster import KMeans
WCSS = []
for i in range(1,11):
    kmeans = KMeans(n_clusters= i, init = 'k-means++', n_init = 10, max_iter =300)
    kmeans.fit(X)
    WCSS.append(kmeans.inertia_)
plt.plot(range(1,11),WCSS)
plt.title('The elbow method')
plt.xlabel('no of clusters')
plt.ylabel('WCSS')
plt.show()

# so .. ideal no of cluster is 5

#building the model
kmeans = KMeans(n_clusters= 5, init = 'k-means++', n_init = 10, max_iter =300)
y = kmeans.fit_predict(X)

#visualising the clusters
plt.scatter(X[y == 0,0],X[y== 0,1],c='red', label='cluster1')
plt.scatter(X[y == 1,0],X[y== 1,1],c='blue', label='cluster2')
plt.scatter(X[y == 2,0],X[y== 2,1],c='green', label='cluster3')
plt.scatter(X[y == 3,0],X[y== 3,1],c='violet', label='cluster4')
plt.scatter(X[y == 4,0],X[y== 4,1],c='cyan', label='cluster5')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],c='yellow')
plt.title('Kmeans cluster')
plt.xlabel('salary')
plt.ylabel('credit score')
plt.legend()
plt.show()
    