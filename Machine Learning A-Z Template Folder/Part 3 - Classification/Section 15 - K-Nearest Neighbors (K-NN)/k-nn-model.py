import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Social_Network_Ads.csv')
y = dataset.iloc[:,-1].values
X = dataset.iloc[:,2:4].values

#scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state =0)

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5, metric = 'minkowski', p=2)
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test,y_pred)

accuracy_score(y_test,y_pred)

#plotting the graph(training set)
from matplotlib.colors import ListedColormap
X_set,y_set = X_train,y_train
#create the meshgrid
X1,X2 = np.meshgrid(np.arange(start =min(X_set[:,0]) - 1,stop=max(X_set[:,0]) +1,step=0.01),
            np.arange(start =min(X_set[:,1]) - 1,stop=max(X_set[:,1]) + 1,step=0.01))
#contouring
#now we have to make prediction on X1 and X2 but classifier is created for x_train.
#so need to convert X1,X2 to shape of X_train to pass it to classifier
y1= classifier.predict( np.array([X1.ravel(),X2.ravel()]).T)
#again have to convert to the shape of X1 or X2
plt.contourf(X1,X2,y1.reshape(X1.shape),cmap = ListedColormap(('red','green')))

# scatter the y_train as small circles
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j,0], X_set[y_set == j,1] , color= ListedColormap(('blue','yellow'))(i), label = j)
plt.title('training set')
plt.xlabel('age')
plt.ylabel('salary')
plt.legend()
plt.show()



#plotting the graph (test set)
from matplotlib.colors import ListedColormap
X_set,y_set = X_test,y_test
#create the meshgrid
X1,X2 = np.meshgrid(np.arange(start =min(X_set[:,0]) - 1,stop=max(X_set[:,0]) +1,step=0.01),
            np.arange(start =min(X_set[:,1]) - 1,stop=max(X_set[:,1]) + 1,step=0.01))
#contouring
#now we have to make prediction on X1 and X2 but classifier is created for x_train.
#so need to convert X1,X2 to shape of X_train to pass it to classifier
y1= classifier.predict( np.array([X1.ravel(),X2.ravel()]).T)
#again have to convert to the shape of X1 or X2
plt.contourf(X1,X2,y1.reshape(X1.shape),cmap = ListedColormap(('red','green')))

# scatter the y_train as small circles
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j,0], X_set[y_set == j,1] , color= ListedColormap(('blue','yellow'))(i), label = j)
plt.title('test set')
plt.xlabel('age')
plt.ylabel('salary')
plt.legend()
plt.show()