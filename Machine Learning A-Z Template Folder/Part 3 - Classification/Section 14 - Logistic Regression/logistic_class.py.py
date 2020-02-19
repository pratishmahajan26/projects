#import libs
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

#import dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X= dataset.iloc[:,2:4].values
y= dataset.iloc[:,4].values

#Separating training and test data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.25,random_state = 0)

#Feature scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.fit_transform(X_test)

#Fitting Logistic regression to the model
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train,y_train)

#Prediction
y_pred = classifier.predict(X_test)

#getting the confusion metrics
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

#graph for training set
from matplotlib.colors import ListedColormap
X_set,y_set = X_train,y_train
# this will create a rectangular grid
X1,X2 = np.meshgrid(np.arange(start=min(X_set[:,0]-1),stop=max(X_set[:,0]+1),step=0.01),
                    np.arange(start=min(X_set[:,1]-1),stop=max(X_set[:,1]+1) ,step=0.01))
 
y1 = classifier.predict(np.array([X1.ravel(),X2.ravel()]).T)
plt.contourf(X1,X2,y1.reshape(X1.shape),cmap=ListedColormap(('red','green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j,0],X_set[y_set == j,1] ,
                c = ListedColormap(('blue','yellow'))(i),label =j)

plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

#graph for test set
from matplotlib.colors import ListedColormap
X_set,y_set = X_test,y_test
X1,X2 = np.meshgrid(np.arange(start = min(X_test[:,0]-1),stop = max(X_test[:,0])+1,step = 0.01),
                    np.arange(start =min(X_test[:,1])-1,stop=max(X_test[:,1])+1,step=0.01)
        )
y1 = classifier.predict(np.array([X1.ravel(),X2.ravel()]).T)
plt.contourf(X1,X2,y1.reshape(X1.shape),cmap=ListedColormap(('red','green')))
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())

for i,j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j,0],X_set[y_set ==j,1],
                c=ListedColormap(('blue','yellow'))(i),label=j)

plt.title('Logistic Regression (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()


