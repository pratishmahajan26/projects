#import libs
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

#import datasaet
dataset = pd.read_csv('Position_Salaries.csv')
X= dataset.iloc[:,1:2].values
y= dataset.iloc[:,2:].values

#Fitting random forest regressor to model
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 300,random_state = 0)
regressor.fit(X,y)

# predict the salary of level 6.5
y_pred = regressor.predict([[6.5]])

#visualising the graph
X_grid = np.arange(min(X),max(X),0.01)
X_grid = X_grid.reshape(len(X_grid),1)
plt.scatter(X,y,color = 'r')
plt.plot(X_grid,regressor.predict(X_grid),color = 'blue')
plt.title('RFR')
plt.xlabel('position')
plt.ylabel('salary')
plt.show()



