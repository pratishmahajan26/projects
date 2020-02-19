# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 17:38:18 2019

@author: spriyadarshini
"""

# import the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset =pd.read_csv('Wine.csv')
dataset.head()
dataset.isnull().sum()  # no missing data
dataset.Customer_Segment.value_counts()
# check categorical values
dataset.select_dtypes(include='object').columns  # 0
dataset.select_dtypes(include='number').columns

y = dataset['Customer_Segment']
X = dataset.drop(columns = ['Customer_Segment'],axis = 1)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 42)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train =scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

# apply PCA
from sklearn.decomposition import PCA
pca = PCA(n_components= None,random_state = 42)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
per_var=np.round(pca.explained_variance_ratio_*100,decimals=1)

# Scree plot  
labels = ['PCA' + str(i) for i in range(0,13)]
plt.bar(x=range(0,len(per_var)),height = per_var,tick_label= labels)
plt.xticks(rotation = 90)
plt.show()

# training the model with only 2 components PCA1 and PCA2
pca = PCA(n_components= 2,random_state = 42)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 42)
classifier.fit(X_train_pca,y_train)
y_pred = classifier.predict(X_test_pca)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

# visualising data

# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train_pca, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test_pca, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()


