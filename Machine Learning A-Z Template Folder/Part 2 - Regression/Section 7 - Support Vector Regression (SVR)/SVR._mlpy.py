#import the libs
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

#import dataset
dataset = pd.read_csv('Position_Salaries.csv')
X= dataset.iloc[:,1:2].values
y=dataset.iloc[:,2:].values

#SVM model doesn't do scalling  automatically
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
X=sc_x.fit_transform(X)
y=sc_y.fit_transform(y)

# Fitting model with SVM
from sklearn.svm import SVR
regressor = SVR(kernel='rbf',degree = 3)
regressor.fit(X,y)


#Visualising
plt.scatter(X,y,color = 'red')
plt.plot(X,regressor.predict(X),color = 'blue')
plt.title('level vs salary(SVM)')
plt.xlabel('level')
plt.ylabel('salary')
plt.show()

#Visualising with grid
X_grid = np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape(len(X_grid),1)
plt.scatter(X,y,color = 'red')
plt.plot(X_grid,regressor.predict(X_grid),color = 'blue')
plt.title('level vs salary(SVM GRID)')
plt.xlabel('level')
plt.ylabel('salary')
plt.show()

# predicting salary of 6.5 level
y_pred = regressor.predict(sc_x.transform(6.5))
sc_y.inverse_transform(y_pred)

# predicting all the values
y_score = regressor.predict(X)
y_score = sc_y.inverse_transform(y_score)


from sklearn.metrics import accuracy_score
accuracy_score(sc_y.inverse_transform(y),y_score.reshape(len(y_score),1))
#accuracy_score(sc_y.inverse_transform(y).reshape(len(y_score),1),y_score.reshape(len(y_score),1))
