import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:,2:4].values
y = dataset.iloc[:,-1].values

#scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

#separating test and train set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 0)

#training the model with SVM
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf',random_state = 0)
classifier.fit(X_train,y_train)

#prediction
y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test,y_pred)

accuracy_score(y_test,y_pred)

#graph for training set
from matplotlib.colors import ListedColormap 
X_set,y_set = X_train,y_train
X1,X2 = np.meshgrid(np.arange(start =min(X_set[:,0])-1 , stop =max(X_set[:,0]) + 1,step =0.01),
        np.arange(start =min(X_set[:,1])-1 , stop =max(X_set[:,1]) + 1, step =0.01))
#y  with same scale as _y_test
y1 = classifier.predict(np.array([X1.ravel(),X2.ravel()]).T)
plt.contourf(X1,X2,y1.reshape(X1.shape),cmap = ListedColormap(('red','green')))
#scatter the plots
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j,0],X_set[y_set ==j ,1], 
                cmap = ListedColormap(('blue','yellow'))(i),label = j)
plt.title('SVN(training set)')
plt.xlabel('age')
plt.ylabel('salary')
plt.legend()
plt.show()


#graph for test set
from matplotlib.colors import ListedColormap 
X_set,y_set = X_test,y_test
X1,X2 = np.meshgrid(np.arange(start =min(X_set[:,0])-1 , stop =max(X_set[:,0]) + 1,step =0.01),
        np.arange(start =min(X_set[:,1])-1 , stop =max(X_set[:,1]) + 1, step =0.01))
#y  with same scale as _y_test
y1 = classifier.predict(np.array([X1.ravel(),X2.ravel()]).T)
plt.contourf(X1,X2,y1.reshape(X1.shape),cmap = ListedColormap(('red','green')))
#scatter the plots
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j,0],X_set[y_set ==j ,1], 
                cmap = ListedColormap(('blue','yellow'))(i),label = j)
plt.title('SVN(test set)')
plt.xlabel('age')
plt.ylabel('salary')
plt.legend()
plt.show()